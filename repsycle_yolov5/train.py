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
from tqdm import tqdm
from utils import preprocessing


def train(model, epochs):

    scaled_anchors = torch.tensor(config.anchors) * \
                     torch.tensor(config.scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

    model = model.to(config.device)
    model.train()

    img_dir = './images/'
    labels = './datas/temp.json'

    dataset = YoloDataset(img_dir, labels, config.anchors, (config.image_size, config.image_size), C=config.nb_classes)
    loader = DataLoader(dataset, batch_size=6, num_workers=0, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    compute_loss = Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):

        losses = []

        # for step in ['train', 'val']:

        for img, target in tqdm(loader):

            x = model(img)

            with torch.cuda.amp.autocast():
                loss = compute_loss.forward(x, target, scaled_anchors)

            losses.append(loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print('###')
        print(f'epochs : {e} / {epochs} | loss : {sum(losses) / len(losses)}')

if __name__ == '__main__':

    model = Model(anchors=config.anchors, nb_classes=config.nb_classes, nb_channels=3)
    epochs = 1000
    train(model, epochs)
    torch.save(model.state_dict(), 'test.pt')
