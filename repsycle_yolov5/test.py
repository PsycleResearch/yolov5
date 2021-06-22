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
from utils import plot_images, non_max_suppression, pred2bboxes__, mean_average_precision, cell_to_coordinates
from dataset import YoloDataset
import time

def rescale_prediction():
    return

def evaluate_model(model, scaled_anchors, image_dir, labels_path):

    return

def test(model):

    img_dir = './images/'
    training_labels = './datas/training_set.json'

    training_dataset = YoloDataset(img_dir, training_labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)

    scaled_anchors = torch.tensor(config.anchors).to(config.device) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.device)

    average_inference_time = []

    model.eval()
    model.to(config.device)

    annotations = {}
    predictions = {}

    for idx, (img, label, bboxes) in enumerate(training_dataset):

        start_time = time.time()
        img = img.unsqueeze(0)
        prediction = model(img)
        prediction = non_max_suppression(prediction, scaled_anchors, iou_threshold=0.2, threshold=0.5)

        annotations[str(idx)] = bboxes
        predictions[str(idx)] = prediction

        end_time = time.time()

        # plot_images(np.asarray(img.to('cpu')).reshape((640, 640, 3)), prediction)
        average_inference_time.append(end_time - start_time)

    average_inference_time = sum(average_inference_time) / len(average_inference_time)
    start_time = time.time()
    mAP = mean_average_precision(predictions, annotations, iou_threshold=0.2, num_classes=1)
    end_time = time.time()
    print(end_time - start_time)
    print(f'* mAP@0.5 : {mAP}'
          f'\n* Average inference time : {average_inference_time} '
          f'\n* Average number of inferences per seconde : {1 / average_inference_time}')

if __name__ == '__main__':

    model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)
    model.load_state_dict(torch.load('./test.pt'))
    model.eval()
    test(model)
