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
from utils import plot_images, non_max_suppression, pred2bboxes, mean_average_precision
import time

def rescale_prediction():
    return

def test(model):

    img_dir = './images/'
    labels = './datas/temp.json'

    with open(labels, 'r') as f:
        datas = json.load(f)

    image_id = list(datas.keys())[0:200]
    annotations = list(datas.values())[0:200]
    predictions = {}

    scaled_anchors = torch.tensor(config.anchors).to(config.device) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.device)

    average_inference_time = []

    model.eval()
    model.to(config.device)

    for idx in range(len(image_id)):

        start_time = time.time()

        img = cv2.imread(f'./images/{image_id[idx]}.jpeg')
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (640, 640))

        img = img / 255.
        img = img.reshape((1,3,640,640))
        img = torch.tensor(img)
        img = img.to(config.device).float()

        prediction = model(img)
        prediction = pred2bboxes(prediction, threshold=0.6, scaled_anchors=scaled_anchors)
        prediction = non_max_suppression(prediction, iou_threshold=0.6, threshold=None)

        end_time = time.time()

        # plot_images(image, prediction)

        average_inference_time.append(end_time - start_time)

        predictions[image_id[idx]] = prediction

    average_inference_time = sum(average_inference_time) / len(average_inference_time)

    # print(annotations)
    # print(ordered_annotations)

    map = mean_average_precision(predictions, datas, iou_threshold=0.5, box_format="midpoint", num_classes=1)

    # print(f'* Average inference time : {average_inference_time} '
    #       f'\n* Average number of inferences per seconde : {1 / average_inference_time}')

if __name__ == '__main__':

    model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)
    model.load_state_dict(torch.load('./test.pt'))
    test(model)
