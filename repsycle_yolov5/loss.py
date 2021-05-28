import torch.nn as nn
import torch
from utils import intersection_over_union

class BCEWithLogitLoss:

    def __init__(self, input_size, num_classes):
        super(nn.BCEWithLogitsLoss(), self).__init__()

class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_bbox = 10

    def forward(self, predictions, target, anchors):

        anchors = anchors.reshape((1, 3, 1, 1, 2))

        # predictions
        # pxy = predictions[..., 0:2].sigmoid()
        # pwh = predictions[..., 2:4].exp() * anchors
        # po = predictions[..., 4:5]
        # pc = predictions[..., 5:]

        # targets
        # txy = target[..., 0:2]
        # twh = target[..., 2:4]
        # to = target[..., 4:5]
        # tc = target[..., 5:]

        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        ### noobj loss:
        no_object_loss = self.bce(predictions[..., 4:5][noobj], target[..., 4:5][noobj])

        ### objectness loss:
        pbox = torch.cat((predictions[..., 0:2].sigmoid(), predictions[..., 2:4].exp() * anchors), dim=-1)
        ious = intersection_over_union(pbox[obj], target[...,0:4][obj]).detach()
        object_loss = self.mse(predictions[..., 4:5][obj].sigmoid(), ious * target[..., 4:5][obj])

        ### bbox loss
        pxy = predictions[..., 0:2].sigmoid()  # x,y coordinates
        twh = torch.log(1e-16 + target[..., 2:4] / anchors)
        box_loss = self.mse(twh[obj], predictions[...,2:4][obj])
        coordinates_loss = self.mse(target[...,0:2][obj], pxy[obj])

        ### class loss
        class_loss = self.bce(predictions[..., 5:][obj], target[...,5:][obj])

        print('___________________')
        print(coordinates_loss)
        print(no_object_loss)
        print(object_loss)
        print(box_loss)
        print(class_loss)
        print('\n')

        loss = self.lambda_bbox * (coordinates_loss + box_loss) + \
               self.lambda_noobj * no_object_loss + \
               self.lambda_obj * object_loss + \
               self.lambda_class * class_loss

        return loss

