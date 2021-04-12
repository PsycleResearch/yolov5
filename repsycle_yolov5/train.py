import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model import Model, create
from dataset import YoloDataset
from loss import Loss
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == '__main__':

    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    IMAGE_SIZE = 640
    S = [IMAGE_SIZE // 80, IMAGE_SIZE // 40, IMAGE_SIZE // 20]
    scaled_anchors = torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    # print(scaled_anchors)

    model = create('yolov5s.pt', pretrained=True, channels=3, classes=1)
    model = model.to("cuda:0")
    model.train()

    img_dir = './images/'
    labels = './datas/labels.json'
    image_size = (640, 640)

    dataset = YoloDataset(img_dir, labels, anchors, image_size)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.9, weight_decay=0.001)
    loss = Loss()
    scaler = torch.cuda.amp.GradScaler()

    epochs = 10

    for e in range(epochs):

        ctr = 0
        losses = []

        for img, target in loader:
            img = img.to('cuda:0')

            x = model(img)

            with torch.cuda.amp.autocast():
                loss1 = loss.forward(x[0], target[0].to('cuda:0'), scaled_anchors[0].to('cuda:0'))
                loss2 = loss.forward(x[1], target[1].to('cuda:0'), scaled_anchors[1].to('cuda:0'))
                loss3 = loss.forward(x[2], target[2].to('cuda:0'), scaled_anchors[2].to('cuda:0'))
                full_loss = loss1 + loss2 + loss3
                #print(full_loss)

            losses.append(full_loss.item())
            optimizer.zero_grad()

            scaler.scale(full_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
        mean_loss = sum(losses) / len(losses)
        print(mean_loss)

