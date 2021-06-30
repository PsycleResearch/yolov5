import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model import Model, create
from dataset import YoloDataset
from loss import Loss
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from tqdm import tqdm
from utils import non_max_suppression, mean_average_precision

def autoanchors(training_dataset):
    from scipy.cluster.vq import kmeans
    import json
    with open('datas/training_set.json', 'r') as f:
        datas = json.load(f)
    datas = np.asarray(list(datas.values()))
    datas = np.squeeze(datas)[:,2:4]
    k, dist = kmeans(datas, 9, iter=100)
    idx = np.argsort(k[:,0] * k[:, 1])
    k = k[idx]
    return k.reshape((3,3,2))

def eval(model, eval_dataset, scaled_anchors):
    predictions = {}
    annotations = {}

    model.eval()

    with open('datas/validation_set.json', 'r') as f:
        import json
        f = json.load(f)

    for i, (img, label, img_id) in enumerate(eval_dataset):
        img = img.unsqueeze(dim=0)

        with torch.no_grad():
            prediction = model(img)

        prediction = non_max_suppression(prediction, scaled_anchors, iou_threshold=0.2, threshold=0.5)
        predictions[str(i)] = prediction
        annotations[str(i)] = f[img_id]

    mAP, precision, recall = mean_average_precision(predictions, annotations, iou_threshold=0.6, num_classes=config.nb_classes)

    print(f'mAP : {mAP} | precision {precision} | recall {recall}')

    model.train()

def train(model, epochs):

    model = model.to(config.device)
    model.train()

    img_dir = './images/'
    training_labels = './datas/validation_set.json'
    validation_labels = './datas/validation_set.json'

    #anchors = autoanchors(training_labels)

    training_dataset = YoloDataset(img_dir, training_labels, config.anchors, config.image_size, C=config.nb_classes, augmentation=False)
    validation_dataset = YoloDataset(img_dir, validation_labels, config.anchors, config.image_size, C=config.nb_classes, augmentation=False)

    scaled_anchors = torch.tensor(config.anchors) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    scaled_anchors = scaled_anchors.to(config.device).float()

    loader = DataLoader(training_dataset, batch_size=6, num_workers=0, shuffle=True)

    #optimizer = optim.Adam(model.parameters(), lr=0.0032)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    compute_loss = Loss()

    for e in range(epochs):

        losses = []
        bbox_losses = []
        noobj_losses = []
        obj_losses = []
        class_losses = []

        model.train()

        for img, target, img_id in tqdm(loader):

            with torch.cuda.amp.autocast():
                x = model(img)
                loss, bbox_loss, noobj_loss, obj_loss, class_loss = compute_loss.forward(x, target, scaled_anchors)

            losses.append(loss)
            bbox_losses.append(bbox_loss)
            noobj_losses.append(noobj_loss)
            obj_losses.append(obj_loss)
            class_losses.append(class_loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('\n###')
        print(f'epochs : {e + 1} / {epochs} | loss : {sum(losses) / len(losses)} '
              f'| bbox_loss : {sum(bbox_losses) / len(bbox_losses)} '
              f'| noobj_losses : {sum(noobj_losses) / len(noobj_losses)} '
              f'| obj_losses : {sum(obj_losses) / len(obj_losses)} '
              f'| class_losses : {sum(class_losses) / len(class_losses)}')

        eval(model, validation_dataset, scaled_anchors)

if __name__ == '__main__':

    #model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)
    model = create('yolov5s.pt', channels=3, classes=config.nb_classes, anchors=config.anchors)
    epochs = 150
    train(model, epochs)
    # torch.save(model.state_dict(), 'test.pt')
