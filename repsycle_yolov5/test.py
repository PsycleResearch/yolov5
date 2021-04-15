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
from processing import letterbox
import cv2

def test(model):

    img_dir = './images/'
    labels = './datas/labels.json'

    scaled_anchors = torch.tensor(config.anchors) * torch.tensor(config.scales). \
        unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

    dataset = YoloDataset(img_dir, labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)
    model.eval()
    model.to(config.device)

    #for img, target in dataset:

    img = cv2.imread('000000391895.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img, _, _ = letterbox(img)
    img = img / 255.
    img = img.reshape((1,3,640,640))
    img = torch.tensor(img)
    # plt.imshow(img)
    # plt.show()
    img = img.to(config.device).float()
    y_ = model(img)

    for i in range(3):
        for j in range(3):

            to = torch.flatten(y_[i][..., 4:5].sigmoid())

            for t in to:
                print(t)

            obj = to > 0.1
            # obj[0][2].to('cpu')
            # print(obj[0][j].to('cpu'))
            # plt.matshow(obj[0][j].to('cpu'))
            # plt.show()

        # txy = (y_[0][...,i,j,0:2] * 2 - 0.5).sigmoid() * 2. - 0.5
        # twh = (y_[0][...,i,j, 2:4].sigmoid() * 2) ** 2 * scaled_anchors[0]
        # tc = y_[0][...,i,j,5:]
        # tbox = torch.cat((txy, twh), 4).to('cuda:0')
        # print(y_[0].shape)

if __name__ == '__main__':

    #model = create('yolov5s.pt', pretrained=True, channels=3, classes=config.nb_classes)
    #model = torch.jit.load('./yolov5s.pt')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    test(model)
