import numpy as np
import os
import pandas as pd
from PIL import Image
import config
from utils import (iou_width_height as iou,
                   non_max_suppression as nms)
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import matplotlib.pyplot as plt
from utils import letterbox, plot_images, cell_to_coordinates

class YoloDataset(Dataset):
    def __init__(self, img_dir, labels, anchors, image_size, S=[80, 40, 20], C=1):

        with open(labels, 'r') as f:
            self.datas = json.load(f)

        self.img_size = image_size
        self.image_id = list(self.datas.keys())[:400]
        self.annotations = list(self.datas.values())[:400]

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.nb_anchors = self.anchors.shape[0]
        self.C = C
        self.S = S
        self.ignore_iou_tresh = 0.5
        self.nb_anchors_per_scale = self.nb_anchors // 3

        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_path = self.img_dir + self.image_id[idx] + '.jpeg'
        bboxes = self.annotations[idx] # [x, y, w, h, class]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.img_size)
        image = image / 255.
        image = image.reshape((3, image.shape[0], image.shape[1]))
        image = torch.tensor(image).float().to(config.device)

        targets = [torch.zeros(self.nb_anchors // 3, S, S, 5 + self.C) for S in self.S]

        # Choose which anchors is responsible for each cells following highest IOU
        for bboxe in bboxes:

            if bboxe is None:
                continue

            iou_anchors = iou(torch.tensor(bboxe[2:4]), self.anchors) # Get IOU for each anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = bboxe
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:

                scale_idx = anchor_idx // self.nb_anchors_per_scale
                anchor_on_scale = anchor_idx % self.nb_anchors_per_scale
                S = self.S[scale_idx]

                # In Yolo, eac
                # h coordinates are relative to each cells:
                i, j = int(S*y), int(S*x)

                # It's possible but rare that two objects with same bbox are taken
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = width * S, height * S

                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 0:4] = box_coordinates
                    one_hot_class = torch.zeros(self.C)
                    one_hot_class[int(class_label)] = 1
                    targets[scale_idx][anchor_on_scale, i, j, 5:] = one_hot_class
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_tresh:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = -1  # ignore prediction

        # print(bboxes)
        # plt.imshow(image)
        # plt.show()

        # debug
        # for i in range(3):
        #     for j in range(3):
        #         plt.matshow(targets[i][j, :, :, 4])
        #         plt.show()
        # print(bboxe)

        #print(targets[0][1,...,5:].shape)

        return image, tuple(targets), bboxes

def test():

    img_dir = './images/'
    labels = './datas/temp.json'
    ds = YoloDataset(img_dir, labels, anchors=config.anchors, image_size=(config.image_size, config.image_size))

    for img, targets in ds:
        labels = cell_to_coordinates(img, targets)
        plot_images(img, labels)

if __name__ == '__main__':
    test()