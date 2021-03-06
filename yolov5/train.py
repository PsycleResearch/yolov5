import json
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.cuda import amp
from tqdm import tqdm

from test import test
from yolov5.models.yolo import Model
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (
    labels_to_class_weights, check_anchors, labels_to_image_weights,
    compute_loss, fitness, strip_optimizer, get_latest_run)
from yolov5.utils.torch_utils import init_seeds, ModelEMA, intersect_dicts

logger = logging.getLogger(__name__)
import streamlit as st


def train(hyperparameters: dict, weights, metric_weights=None, epochs=2, batch_size=1,
          logging_directory='runs/',
          cfg: str = None, resume=False, img_size=640, workers=8, name='', train_list_path='train.txt',
          test_list_path='text.txt', classes=[], augment=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_directory = f'{logging_directory}/weights'
    os.makedirs(weights_directory, exist_ok=True)
    last_weights_directory = weights_directory + '/last.pt'
    best_weights_directory = weights_directory + '/best.pt'
    results_file = f'{logging_directory}/results.txt'

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1)

    nb_classes = len(classes)

    if weights is not None:
        # Load pretrained model
        checkpoint = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(checkpoint['model'].yaml, channels=3, nb_classes=nb_classes).to(device)
        exclude = []  # exclude keys
        state_dict = checkpoint['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        # Create model
        model = Model(cfg, channels=3, nb_classes=nb_classes).to(device)  # create

    # Freeze
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

    # Optimizer
    nominal_batch_size = 64
    accumulate = max(round(nominal_batch_size / batch_size), 1)  # accumulate loss before optimizing
    hyperparameters['weight_decay'] *= batch_size * accumulate / nominal_batch_size  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyperparameters['lr0'], betas=(hyperparameters['beta1'], 0.999))

    optimizer.add_param_group(
        {'params': pg1, 'weight_decay': hyperparameters['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if weights is not None:
        # Optimizer
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness = checkpoint['best_fitness']

        # Results
        if checkpoint.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(checkpoint['training_results'])  # write results.txt

        # Epochs
        start_epoch = checkpoint['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, checkpoint['epoch'], epochs))
            epochs += checkpoint['epoch']  # finetune additional epochs

        del checkpoint, state_dict

    # Image sizes
    grid_size = int(max(model.stride))  # max stride
    assert img_size == math.ceil(
        img_size / grid_size) * grid_size, f'img_size ({img_size}) must be a multiple of max stride ({grid_size})'

    # Exponential moving average
    exponential_moving_average = ModelEMA(model)

    # Trainloader
    train_dataloader, train_dataset = create_dataloader(train_list_path, img_size, batch_size, grid_size,
                                                        hyperparameters=hyperparameters,
                                                        augment=augment,
                                                        workers=workers)
    nb_batches = len(train_dataloader)

    # Testloader
    exponential_moving_average.updates = start_epoch * nb_batches // accumulate  # set EMA updates
    test_dataloader, _ = create_dataloader(test_list_path, img_size, batch_size, grid_size,
                                           hyperparameters=hyperparameters,
                                           augment=False,
                                           workers=workers)

    # Model parameters
    hyperparameters['cls_loss_gain'] *= nb_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nb_classes  # attach number of classes to model
    model.hyp = hyperparameters  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, nb_classes).to(device)  # attach class weights
    model.names = classes

    # Check anchors
    check_anchors(train_dataset, model=model, thr=hyperparameters['anchor_multiple_threshold'], img_size=img_size)

    # Start training
    t0 = time.time()
    nw = max(3 * nb_batches, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nb_classes)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Starting training for %g epochs...' % epochs)
    my_bar = st.progress(0)
    results_dict = {'Epoch': '', 'Precision': '', 'Recall': '', 'mAP': '', 'F1': ''}
    results_df = pd.DataFrame([results_dict])
    results_table = st.table(results_df)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        my_bar.progress(int(epoch / epochs * 100))
        if epoch == epochs - 1:
            my_bar.progress(100)
        # Update image weights (optional)
        if train_dataset.image_weights:
            # Generate indices
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(train_dataset.labels, nc=nb_classes, class_weights=w)
            train_dataset.indices = random.choices(range(train_dataset.n), weights=image_weights,
                                                   k=train_dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = tqdm(enumerate(train_dataloader), total=nb_batches)  # progress bar
        optimizer.zero_grad()

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        print(f'TRAINING Epoch: {epoch}/{epochs - 1} \tgpu_mem: {mem}')
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb_batches * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nominal_batch_size / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyperparameters['momentum']])

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                # if not torch.isfinite(loss):
                #     logger.info('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if exponential_moving_average:
                    exponential_moving_average.update(model)
            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # mAP
        if exponential_moving_average:
            exponential_moving_average.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        results, maps, times = test(
            model=exponential_moving_average.ema.module if hasattr(exponential_moving_average.ema,
                                                                   'module') else exponential_moving_average.ema,
            classes=classes,
            batch_size=batch_size,
            img_size=img_size,
            dataloader=test_dataloader,
            save_dir=logging_directory)
        results_dict_2 = {'Epoch': str(epoch), 'Precision': str(round(results[0], 2)),
                          'Recall': str(round(results[1], 2)),
                          'mAP': str(round(results[2], 2)), 'F1': str(round(results[3], 2))}
        results_df_2 = pd.DataFrame([results_dict_2])
        results_table.add_rows(results_df_2)
        # Write
        with open(results_file, 'a') as f:
            f.write('%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1),
                     weights=metric_weights)  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        with open(results_file, 'r') as f:  # create checkpoint
            checkpoint = {'epoch': epoch,
                          'best_fitness': best_fitness,
                          'training_results': f.read(),
                          'model': exponential_moving_average.ema.module if hasattr(exponential_moving_average,
                                                                                    'module') else exponential_moving_average.ema,
                          'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last, best and delete
        torch.save(checkpoint, last_weights_directory)
        if best_fitness == fi:
            torch.save(checkpoint, best_weights_directory)
        del checkpoint
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    n = ('_' if len(name) and not name.isnumeric() else '') + name
    fresults, flast, fbest = f'results{n}.txt', f'{weights_directory}/last{n}.pt', f'{weights_directory}/best{n}.pt'
    for f1, f2 in zip([last_weights_directory, best_weights_directory, results_file], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
    # Finish
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    weights = 'weights/yolov5s.pt'  # pre-trained weights
    cfg = None  # In case we resume
    train_list_path = 'train.txt'
    test_list_path = 'test.txt'
    classes = ['not ok', 'ok']
    hyperparameters_path = 'data/hyp.json'
    epochs = 8
    batch_size = 8
    img_size = 640
    resume = False  # can also be a string for the desired checkpoint
    name = ''
    logging_directory = 'runs/'
    workers = 8
    augment = True

    if resume:
        assert cfg is not None
        checkpoint = resume if isinstance(resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(checkpoint), 'ERROR: --resume checkpoint does not exist'
        weights = checkpoint
    else:
        assert weights is not None

    with open(hyperparameters_path) as f:
        hyperparameters = json.load(f)

    train(hyperparameters, weights, cfg=cfg, train_list_path=train_list_path,
          test_list_path=test_list_path, classes=classes, epochs=epochs, batch_size=batch_size,
          img_size=img_size, resume=resume, name=name, logging_directory=logging_directory, workers=workers,
          augment=augment)
