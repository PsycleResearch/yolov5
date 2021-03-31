import json
import logging
import math
import os
import albumentations as A
import cv2
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch import nn
from torch.cuda import amp
from tqdm import tqdm

from test import test
from yolov5.models.yolo import Model
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (
    labels_to_class_weights, check_anchors, strip_optimizer, get_latest_run)
from yolov5.utils.torch_utils import init_seeds, intersect_dicts
from yolov5.utils.loss import ComputeLoss

logger = logging.getLogger(__name__)


def fitness(precision, recall, map50, map, metric_weights: list):
    """
    Weighted combination of precision, recall, mAP50 and mAP
    """
    return np.array([precision, recall, map50, map] * np.array(metric_weights)).sum()


def train(hyperparameters: dict, weights: str, metric_weights: list = None, epochs: int = 2, batch_size: int = 1,
          logging_directory: str = 'runs/', accumulate: int = 1,
          resume: bool = False, img_size: int = 640, workers: int = 8,
          train_list_path: str = 'train.txt',
          test_list_path: str = 'text.txt', classes: list = [], augmentations: list = []):
    is_cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda_available else 'cpu')
    weights_directory = f'{logging_directory}/weights'
    os.makedirs(weights_directory, exist_ok=True)
    last_weights_directory = weights_directory + '/last.pt'
    best_weights_directory = weights_directory + '/best.pt'
    init_seeds(1)
    nb_classes = len(classes)

    # Load pretrained model
    checkpoint = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(checkpoint['model'].yaml, input_channels=3, nb_classes=nb_classes).to(device)
    state_dict = checkpoint['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
    model.load_state_dict(state_dict, strict=False)

    # Freeze
    freeze = []
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.Adam(pg0, lr=hyperparameters['lr0'], betas=(hyperparameters['momentum'], 0.999))
    optimizer.add_param_group(
        {'params': pg1, 'weight_decay': hyperparameters['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch, best_fitness = 0, 0.0
    # Optimizer
    if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_fitness = checkpoint['best_fitness']

    # Epochs
    start_epoch = checkpoint['epoch'] + 1
    if resume:
        assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
    if epochs < start_epoch:
        logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (weights, checkpoint['epoch'], epochs))
        epochs += checkpoint['epoch']

    del checkpoint, state_dict

    # Image sizes
    grid_size = max(int(model.stride.max()), 32)   # max stride
    assert img_size == math.ceil(
        img_size / grid_size) * grid_size, f'img_size ({img_size}) must be a multiple of max stride ({grid_size})'

    # Trainloader
    train_dataloader, train_dataset = create_dataloader(train_list_path, img_size, batch_size, grid_size,
                                                        hyperparameters=hyperparameters,
                                                        workers=workers,
                                                        augment=True,
                                                        augmentations=augmentations)
    nb_batches = len(train_dataloader)

    # Model parameters
    hyperparameters['cls_loss_gain'] *= nb_classes / 80.  # scale coco-tuned cls_loss_gain to current dataset
    model.nb_classes = nb_classes
    model.hyperparameters = hyperparameters
    model.giou_loss_ratio = 1.0
    model.class_weights = labels_to_class_weights(train_dataset.labels, nb_classes).to(device)
    model.classes = classes

    # Testloader
    test_dataloader, _ = create_dataloader(test_list_path, img_size, batch_size, grid_size,
                                           hyperparameters=hyperparameters,
                                           workers=workers)

    # Check anchors
    check_anchors(train_dataset, model=model, thr=hyperparameters['anchor_multiple_threshold'], img_size=img_size)

    # TODO: hyp['box'] *= 3. / nl  # scale to layers
    nb_detection_layers = model.model[-1].nb_detection_layers
    hyperparameters['cls_loss_gain'] *= nb_classes / 80. * 3. / nb_detection_layers  # scale to classes and layers
    hyperparameters['obj_loss_gain'] *= (img_size / 640) ** 2 * 3. / nb_detection_layers  # scale to image size and layers
    model.nb_classes = nb_classes
    model.hyperparameters = hyperparameters
    model.iou_loss_ratio = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.classes = classes

    # Start training
    nb_warmup_iterations = max(3 * nb_batches, 1e3)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=is_cuda_available)
    compute_loss = ComputeLoss(model)
    logger.info('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if is_cuda_available else 0)  # (GB)
        print(f'TRAINING Epoch: {epoch}/{epochs - 1} \tgpu_mem: {mem}')
        for i, (images, targets, paths) in tqdm(enumerate(train_dataloader), total=nb_batches):
            nb_integrated_batches = i + nb_batches * epoch
            images = images.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if nb_integrated_batches <= nb_warmup_iterations:
                xi = [0, nb_warmup_iterations]  # x interp
                accumulate = max(1, np.interp(nb_integrated_batches, xi, [1, nb_batches / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(nb_integrated_batches, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(nb_integrated_batches, xi, [0.9, hyperparameters['momentum']])

            # Autocast
            with amp.autocast(enabled=is_cuda_available):
                predictions = model(images)
                loss, _ = compute_loss(predictions, targets.to(device))

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if nb_integrated_batches % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Scheduler
        scheduler.step()

        # mAP
        final_epoch = epoch + 1 == epochs
        test_precision, test_recall, test_mAP50, test_mAP = test(
            model=model,
            dataloader=test_dataloader,
            save_dir=logging_directory,
            conf_thres=0.1,
            iou_thres=0.6)

        # Update best mAP
        fitness_i = fitness(test_precision, test_recall, test_mAP50, test_mAP, metric_weights)
        if fitness_i > best_fitness:
            print(f"IMPROVING ACCORDING TO METRIC: from {best_fitness} to {fitness_i}")
            best_fitness = fitness_i

        # Save model
        checkpoint = {'epoch': epoch,
                      'best_fitness': best_fitness,
                      'model': model,
                      'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last, best and delete
        torch.save(checkpoint, last_weights_directory)
        if best_fitness == fitness_i:
            torch.save(checkpoint, best_weights_directory)
        del checkpoint
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    flast, fbest = f'{weights_directory}/last.pt', f'{weights_directory}/best.pt'
    for f1, f2 in zip([last_weights_directory, best_weights_directory], [flast, fbest]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # weights = 'weights/yolov5s_baptiste.pt'  # pre-trained weights
    weights = 'weights/new_yolov5s_baptiste.pt'  # pre-trained weights
    train_list_path = 'train.txt'
    test_list_path = 'test.txt'
    classes = ['pli']
    hyperparameters_path = 'data/hyp.json'
    epochs = 50
    batch_size = 1
    accumulate = 1  # number of batches before optimizing
    img_size = 640
    resume = False  # can also be a string for the desired checkpoint
    logging_directory = 'runs/'
    workers = 8
    metric_weights = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]

    if resume:
        checkpoint = resume if isinstance(resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(checkpoint), 'ERROR: --resume checkpoint does not exist'
        weights = checkpoint
    else:
        assert weights is not None

    with open(hyperparameters_path) as f:
        hyperparameters = json.load(f)

    augmentations = [
        A.Blur(blur_limit=3, p=0.5),
        A.CLAHE(clip_limit=4, p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=cv2.INTER_NEAREST, p=0.5),
        A.GaussNoise(var_limit=25, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.GlassBlur(sigma=0.7, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
        A.MultiplicativeNoise(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.5),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
        A.Flip(p=0.5),
    ]

    train(hyperparameters, weights, train_list_path=train_list_path,
          test_list_path=test_list_path, classes=classes, epochs=epochs, batch_size=batch_size,
          accumulate=accumulate,
          img_size=img_size, resume=resume, logging_directory=logging_directory, workers=workers,
          metric_weights=metric_weights, augmentations=augmentations)

# TODO: check new weights from master (new architecture?)