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
import json
from utils import plot_images

def rescale_prediction():
    return

def test(model):

    img_dir = './images/'
    labels = './datas/temp.json'

    with open(labels, 'r') as f:
        datas = json.load(f)

    #img_size = image_size
    image_id = list(datas.keys())[:256]
    annotations = list(datas.values())[:256]

    scaled_anchors = torch.tensor(config.anchors).to(config.device) * torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.device)

    dataset = YoloDataset(img_dir, labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)
    model.eval()
    model.to(config.device)

    for i in range(32):

        img = cv2.imread(f'./images/{image_id[i]}.jpeg')
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (640, 640))

        plt.imshow(img)
        plt.show()

        img = img / 255.
        img = img.reshape((1,3,640,640))
        img = torch.tensor(img)
        img = img.to(config.device).float()

        y_ = model(img)

        predictions = []
        coordinates = []

        for p in range(3):
            for i in range(y_[p].shape[-2]):
                for j in range(y_[p].shape[-2]):
                    for s in range(3):
                        objectness = torch.sigmoid(y_[p][0,s,i,j,4])
                        if objectness > 0.7:
                            x = ( j + y_[p][0,s,i,j,0].sigmoid().detach() ) / y_[p].shape[-2]
                            y = ( i + y_[p][0,s,i,j,1].sigmoid().detach() ) / y_[p].shape[-2]
                            w = (y_[p][0,s,i,j,2].exp().detach()) * scaled_anchors[p][s][0] / y_[p].shape[-2]
                            h = (y_[p][0,s,i,j,3].exp().detach()) * scaled_anchors[p][s][1] / y_[p].shape[-2]
                            c = torch.argmax(y_[p][0,s,i,j,5:]).detach()

                            predictions.append([x, y, w, h, c])
                            #coordinates.append()

        plot_images(image, predictions)

    # coordinates
    # for bboxe in predictions:
        # plt.scatter(x1, y1)
        # plt.scatter(x2, y2)
        # plt.imshow(img.reshape((640,640,3)).to('cpu'))
        # plt.show()

    # mask = (y_[2][...,4].sigmoid() > 0.3)
    #
    # wh = torch.sigmoid(y_[2][...,2:4][mask] * 2) ** 2 #* scaled_anchors
    # xy = y_[2][...,:2].sigmoid()[mask] * 2 - 0.5
    # print(xy)
    # print(wh * 640)

        # txy = (y_[0][...,i,j,0:2] * 2 - 0.5).sigmoid() * 2. - 0.5
        # twh = (y_[0][...,i,j, 2:4].sigmoid() * 2) ** 2 * scaled_anchors[0]
        # tc = y_[0][...,i,j,5:]
        # tbox = torch.cat((txy, twh), 4).to('cuda:0')
        # print(y_[0].shape)

if __name__ == '__main__':

    model = create('yolov5s.pt', pretrained=True, channels=3, classes=config.nb_classes)
    model.load_state_dict(torch.load('./test.pt'))
    test(model)
