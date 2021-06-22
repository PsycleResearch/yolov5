import torch.nn as nn
import torch
from utils import bbox_iou
from config import device

class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.lambda_class = 0.243
        self.lambda_noobj = 0.3
        self.lambda_obj = 0.3
        self.lambda_bbox = 0.0296

    def forward(self, predictions, targets, anchors):

        no_object_loss = torch.zeros(1, device=device)
        object_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        class_loss = torch.zeros(1, device=device)
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
            )

            ### bbox loss
            # pxy = prediction[..., 0:2].sigmoid()  # x,y coordinates
            # pwh = prediction[...,2:4].exp() * anchor
            # twh = target[..., 2:4] # torch.log(1e-16 + target[..., 2:4] / anchor)
            # box_loss += self.mse(twh[obj], pwh[obj])
            # coordinates_loss += self.mse(target[...,0:2][obj], pxy[obj])

            pxy = prediction[..., 0:2].sigmoid() * 2 - 0.5
            pwh = (prediction[..., 2:4].sigmoid() * 2) ** 2 * anchor
            pbox = torch.cat((pxy, pwh), dim=-1)
            ciou = bbox_iou(pbox[obj].T, target[..., 0:4][obj].T, x1y1x2y2=False, CIoU=True)
            box_loss += (1.0 - ciou).mean()

            ### objectness loss:
            object_loss += self.mse(
                prediction[..., 4:5][obj].sigmoid(),
                target[..., 4:5][obj] * ciou.detach().clamp(0).unsqueeze(dim=1)
            )

            ### class loss
            class_loss += self.bce(
                prediction[..., 5:][obj],
                target[...,5:][obj]
            )

            # print('___________________')
            # print(coordinates_loss)
            # print(no_object_loss)
            # print(object_loss)
            # print(box_loss)
            # print(class_loss)
            # print('\n')

            box_loss *= self.lambda_bbox
            no_object_loss *= self.lambda_noobj
            object_loss *= self.lambda_obj
            class_loss *= self.lambda_class

        loss = box_loss + no_object_loss + object_loss + class_loss

        return (
            loss * bs, box_loss.item(),
            no_object_loss.item(),
            object_loss.item(),
            class_loss.item()
        )