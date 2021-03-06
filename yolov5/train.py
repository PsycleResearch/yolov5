import logging
import math
import os
import random
import test  # import test.py to get mAP after each epoch
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from yolov5.models.yolo import Model
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_img_size, increment_dir, set_logging)
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)
import streamlit as st


def train(hyp, device, weights, tb_writer=None, metric_weights=None, epochs=2, batch_size=1, logdir='runs/',
          data='dataset.yaml', cfg='', resume=False, img_size=[640, 640], workers=8, name=''):
    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(logdir) / 'evolve'  # logging directory
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')

    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    with torch_distributed_zero_first(-1):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(-1):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if cfg else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Exponential moving average
    ema = ModelEMA(model)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, hyp=hyp, augment=True, workers=workers)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)

    # Testloader
    ema.updates = start_epoch * nb // accumulate  # set EMA updates
    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, hyp=hyp, augment=False, workers=workers)[0]  # only runs on process 0

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    plot_labels(labels, save_dir=log_dir)
    if tb_writer:
        # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    logger.info('Using %g dataloader workers' % dataloader.num_workers)
    logger.info('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
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
        if dataset.image_weights:
            # Generate indices
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                             k=dataset.n)  # rand weighted idx

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

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
                if ema:
                    ema.update(model)
            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            # Plot
            if ni < 3:
                f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        # mAP
        if ema:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        results, maps, times = test.test(data,
                                         batch_size=batch_size,
                                         imgsz=imgsz_test,
                                         model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                         dataloader=testloader,
                                         save_dir=log_dir)
        results_dict_2 = {'Epoch': str(epoch), 'Precision': str(round(results[0], 2)),
                          'Recall': str(round(results[1], 2)),
                          'mAP': str(round(results[2], 2)), 'F1': str(round(results[3], 2))}
        results_df_2 = pd.DataFrame([results_dict_2])
        results_table.add_rows(results_df_2)
        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1),
                     weights=metric_weights)  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        with open(results_file, 'r') as f:  # create checkpoint
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': f.read(),
                    'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                    'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last, best and delete
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    n = ('_' if len(name) and not name.isnumeric() else '') + name
    fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
    for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
    # Finish
    plot_results(save_dir=log_dir)  # save as results.png
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    weights = 'weights/yolov5s.pt'
    cfg = ''
    data = 'dataset.yaml'
    hyp = ''
    epochs = 8
    batch_size = 16
    img_size = [640, 640]
    resume = False
    name = ''
    device = ''
    logdir = 'runs/'
    workers = 8

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    # parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    # parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # opt = parser.parse_args()

    # Set DDP variables
    # opt.world_size = 1
    set_logging(-1)

    # Resume
    if resume:  # resume an interrupted run
        ckpt = resume if isinstance(resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
        #     opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        cfg, weights, resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        hyp = hyp or ('data/hyp.finetune.yaml' if weights else 'data/hyp.scratch.yaml')
        data, cfg, hyp = check_file(data), check_file(cfg), check_file(hyp)  # check files
        assert len(cfg) or len(weights), 'either --cfg or --weights must be specified'
        img_size.extend([img_size[-1]] * (2 - len(img_size)))  # extend to 2 sizes (train, test)

    device = select_device(device, batch_size=batch_size)

    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % logdir)
    tb_writer = SummaryWriter(log_dir=increment_dir(Path(logdir) / 'exp', name))  # runs/exp

    train(hyp, device, weights, tb_writer=tb_writer, cfg=cfg, data=data, epochs=epochs, batch_size=batch_size,
          img_size=img_size, resume=resume, name=name, logdir=logdir, workers=workers)
