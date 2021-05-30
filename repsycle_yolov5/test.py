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
from processing import letterbox
import cv2
import json
from utils import plot_images, non_max_suppression

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

    scaled_anchors = torch.tensor(config.anchors).to(config.device) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.device)

    model.eval()
    model.to(config.device)

    for z in range(32):

        img = cv2.imread(f'./images/{image_id[z]}.jpeg')
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

        for p in range(3):
            for i in range(y_[p].shape[-2]):
                for j in range(y_[p].shape[-2]):
                    for s in range(3):
                        objectness = torch.sigmoid(y_[p][0,s,i,j,4]).data.tolist()
                        if objectness > 0.5:
                            x = ( ( j + y_[p][0,s,i,j,0].sigmoid().detach() ) / y_[p].shape[-2] ).data.tolist()
                            y = ( ( i + y_[p][0,s,i,j,1].sigmoid().detach() ) / y_[p].shape[-2] ).data.tolist()
                            w = ( (y_[p][0,s,i,j,2].exp().detach()) * scaled_anchors[p][s][0] / y_[p].shape[-2] ).data.tolist()
                            h = ( (y_[p][0,s,i,j,3].exp().detach()) * scaled_anchors[p][s][1] / y_[p].shape[-2] ).data.tolist()
                            c = ( torch.argmax(y_[p][0,s,i,j,5:]).detach() ).data.tolist()
                            predictions.append([x, y, w, h, objectness, c])

        # predictions = non_max_suppression(predictions, iou_threshold = 0.6, threshold = 0.45)
        print(predictions)
        plot_images(image, predictions)

if __name__ == '__main__':
    model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)

    # model = create('yolov5s.pt', pretrained=True, channels=3, classes=config.nb_classes)
    model.load_state_dict(torch.load('./test.pt'))
    test(model)
