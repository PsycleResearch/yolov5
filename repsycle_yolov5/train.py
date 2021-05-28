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
from utils import preprocessing


def train(model, epochs):

    # Convert proportion anchors

    # print(config.scales)

    scaled_anchors = torch.tensor(config.anchors) * torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

    # print(scaled_anchors)

    model = model.to(config.device)
    model.train()

    img_dir = './images/'
    labels = './datas/temp.json'

    dataset = YoloDataset(img_dir, labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)
    loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):

        losses = []

        for img, target in loader:

            # img = preprocessing(img)
            # img = img.to(config.device)

            target0 = target[0].to(config.device)  # P 80
            target1 = target[1].to(config.device)  # P 40
            target2 = target[2].to(config.device)  # P 20

            x = model(img)

            with torch.cuda.amp.autocast():

                loss1 = loss.forward(x[0], target0, scaled_anchors[0].to('cuda:0'))
                loss2 = loss.forward(x[1], target1, scaled_anchors[1].to('cuda:0'))
                loss3 = loss.forward(x[2], target2, scaled_anchors[2].to('cuda:0'))
                full_loss = loss1 + loss2 + loss3

            losses.append(full_loss.item())
            optimizer.zero_grad()

            scaler.scale(full_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print('_____________________')


if __name__ == '__main__':

    model = create('yolov5s.pt', pretrained=True, channels=3, classes=config.nb_classes)

    epochs = 300
    train(model, epochs)

    torch.save(model.state_dict(), 'test.pt')

    # model.load_state_dict(torch.load('test.pt'))
    # test(model)

    # model.eval()
    # for img, target in loader:
    #     img = img.to('cuda:0')
    #     x = model(img)
    #     #print(x)