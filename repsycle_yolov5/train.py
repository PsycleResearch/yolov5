import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model import Model
from dataset import YoloDataset
from loss import Loss
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from tqdm import tqdm
from utils import pred2bboxes, non_max_suppression, mean_average_precision, cell_to_coordinates

def train(model, epochs):

    scaled_anchors = torch.tensor(config.anchors) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    scaled_anchors = scaled_anchors.to(config.device)

    model = model.to(config.device)
    model.train()

    img_dir = './images/'
    training_labels = './datas/training_set.json'
    validation_labels = './datas/validation_set.json'

    training_dataset = YoloDataset(img_dir, validation_labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)
    validation_dataset = YoloDataset(img_dir, validation_labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)

    loader = DataLoader(training_dataset, batch_size=6, num_workers=0, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    compute_loss = Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):

        losses = []
        bbox_losses = []
        noobj_losses = []
        obj_losses = []
        class_losses = []

        model.train()

        for img, target, bboxe in tqdm(loader):

            x = model(img)

            with torch.cuda.amp.autocast():
                loss, bbox_loss, noobj_loss, obj_loss, class_loss = compute_loss.forward(x, target, scaled_anchors)

            losses.append(loss)
            bbox_losses.append(bbox_loss)
            noobj_losses.append(noobj_loss)
            obj_losses.append(obj_loss)
            class_losses.append(class_loss)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print('###')
        print(f'epochs : {e + 1} / {epochs} | loss : {sum(losses) / len(losses)} '
              f'| bbox_loss : {sum(bbox_losses) / len(bbox_losses)} '
              f'| noobj_losses : {sum(noobj_losses) / len(noobj_losses)} '
              f'| obj_losses : {sum(obj_losses) / len(obj_losses)} '
              f'| class_losses : {sum(class_losses) / len(class_losses)}')

        predictions = {}
        annotations = {}

        model.eval()

        if e % 10 == 0 and e != 0:

            for i, (img, label, bboxe) in enumerate(validation_dataset):

                img = img.unsqueeze(dim=0)

                with torch.no_grad():
                    prediction = model(img)

                # prediction = pred2bboxes(prediction, threshold=0.5, scaled_anchors=scaled_anchors)
                prediction = non_max_suppression(prediction, scaled_anchors, iou_threshold=0.6, threshold=0.5)

                predictions[str(i)] = prediction
                annotations[str(i)] = bboxe

            print(f'mAP : {mean_average_precision(predictions, annotations, iou_threshold=0.6, box_format="midpoint", num_classes=1)}')

if __name__ == '__main__':

    model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)
    epochs = 1000
    train(model, epochs)
    torch.save(model.state_dict(), 'test.pt')
