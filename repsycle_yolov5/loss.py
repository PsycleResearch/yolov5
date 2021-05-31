import torch.nn as nn
import torch
from utils import intersection_over_union, box_giou
from config import device

class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 0.1
        self.lambda_noobj = 1
        self.lambda_obj = 0.1
        self.lambda_bbox = 1

    def forward(self, predictions, targets, anchors):

        no_object_loss = 0
        object_loss = 0
        box_loss = 0
        coordinates_loss = 0
        class_loss = 0

        for i, target in enumerate(targets):

            prediction = predictions[i].to(device)
            anchor = anchors[i].to(device)
            anchor = anchor.reshape((1, 3, 1, 1, 2))
            target = target.to(device)

            obj = target[..., 4] == 1
            noobj = target[..., 4] == 0

            ### noobj loss:
            no_object_loss += self.bce(prediction[..., 4:5][noobj], target[..., 4:5][noobj])

            ### objectness loss:
            pbox = torch.cat((prediction[..., 0:2].sigmoid(), prediction[..., 2:4].exp() * anchor), dim=-1)
            ious = intersection_over_union(pbox[obj], target[...,0:4][obj])
            object_loss += self.mse(prediction[..., 4:5][obj].sigmoid(), ious.detach() * target[..., 4:5][obj])

            ### bbox loss
            pxy = prediction[..., 0:2].sigmoid()  # x,y coordinates
            twh = torch.log(1e-16 + target[..., 2:4] / anchor)
            box_loss += self.mse(twh[obj], prediction[...,2:4][obj])
            coordinates_loss += self.mse(target[...,0:2][obj], pxy[obj])

            ### class loss
            class_loss += self.bce(prediction[..., 5:][obj], target[...,5:][obj])

            # print('___________________')
            # print(coordinates_loss)
            # print(no_object_loss)
            # print(object_loss)
            # print(box_loss)
            # print(class_loss)
            # print('\n')

        loss = self.lambda_bbox * (coordinates_loss + box_loss) + \
               self.lambda_noobj * no_object_loss + \
               self.lambda_obj * object_loss + \
               self.lambda_class * class_loss

        return (loss,
                self.lambda_bbox * (coordinates_loss + box_loss),
                self.lambda_noobj * no_object_loss,
                self.lambda_obj * object_loss,
                self.lambda_class * class_loss)

    def forward_(self, predictions, target, anchors):

        return

    def build_target(self, predictions, targets):

        return

