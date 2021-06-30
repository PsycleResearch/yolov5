import torch.nn as nn
import torch
from utils import bbox_iou
from config import device

class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.lambda_class = 0.03125
        self.lambda_noobj = 0.5
        self.lambda_obj = 0.15
        self.lambda_bbox = 0.1
        self.balance = [4.0, 1.0, 0.4]

    def forward(self, predictions, targets, anchors):

        no_object_loss = torch.tensor(0, device=device).float()
        object_loss = torch.tensor(0, device=device).float()
        box_loss = torch.tensor(0, device=device).float()
        class_loss = torch.tensor(0, device=device).float()
        bs = targets[0].shape[0]

        for i, target in enumerate(targets):

            prediction = predictions[i].to(device)
            anchor = anchors[i].to(device)
            anchor = anchor.reshape((1, 3, 1, 1, 2))
            target = target.to(device)

            obj = target[..., 4] == 1
            noobj = target[..., 4] == 0

            ### noobj loss:
            no_object_loss += self.bce(
                prediction[..., 4:5][noobj],
                target[..., 4:5][noobj]
            ) * self.balance[i]

            pxy = prediction[..., 0:2].sigmoid() #* 2 - 0.5
            pwh = (prediction[..., 2:4].sigmoid() * 2) ** 2 * anchor
            pbox = torch.cat((pxy, pwh), dim=-1)
            ciou = bbox_iou(pbox[obj].T, target[..., 0:4][obj].T, x1y1x2y2=False, CIoU=True)
            box_loss += (1.0 - ciou).mean()

            ### objectness loss:
            object_loss += self.bce(
                prediction[..., 4:5][obj],
                target[..., 4:5][obj] * ciou.detach().clamp(0).unsqueeze(dim=1)
            ) * self.balance[i]

            ### class loss
            class_loss += self.bce(
                prediction[..., 5:][obj],
                target[...,5:][obj]
            )

            # print('___________________')
            # print(no_object_loss)
            # print(object_loss)
            # print(box_loss)
            # print(class_loss)
            # print('\n')

        box_loss *= self.lambda_bbox * 1/3
        no_object_loss *= self.lambda_noobj * 1/3
        object_loss *= self.lambda_obj * 1/3
        class_loss *= self.lambda_class * 1/3

        loss = box_loss + no_object_loss + object_loss + class_loss

        return (
            loss * bs, box_loss.item(),
            no_object_loss.item(),
            object_loss.item(),
            class_loss.item()
        )