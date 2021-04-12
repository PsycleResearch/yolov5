import numpy as np
import os
import pandas as pd
from PIL import Image
from utils import (iou_width_height as iou,
                   non_max_suppression as nms)
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2

class YoloDataset(Dataset):
    def __init__(self,
                 img_dir,
                 labels,
                 anchors,
                 image_size,
                 S=[20, 40, 80],
                 C=20):

        with open(labels, 'r') as f:
            self.datas = json.load(f)

        self.image_id = list(self.datas.keys())
        self.annotations = list(self.datas.values())

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
        image = cv2.resize(image, (640, 640))

        import matplotlib.pyplot as plt

        targets = [torch.zeros(self.nb_anchors // 3, S, S, 6) for S in self.S] #[prob, x, y, w, h, c]

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

                # In Yolo, each coordinates are relative to each cells:
                i, j = int(S*y), int(S*x)

                # It's possible but rare that two objects with same bbox are taken
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = width * S, height * S

                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 0:4]= box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

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

        image = image.reshape((3, 640, 640))
        image = torch.tensor(image).float()#.to('cuda:0')

        return image, tuple(targets[::-1])

if __name__ == '__main__':

    img_dir = './images/'
    labels = './datas/labels.json'
    image_size = (0, 0)

    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    ds = YoloDataset(img_dir, labels, anchors, image_size)
