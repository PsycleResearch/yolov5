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

    # Coord XY + Coord WH + Coord

    def forward(self,predictions, target, anchors):

        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.matshow(target[0,0,...,2].detach().numpy())
        # plt.show()

        anchors = anchors.reshape((1, 3, 1, 1, 2))

        # predictions
        pxy = predictions[..., 0:2].sigmoid() * 2. - 0.5
        pwh = (predictions[..., 2:4].sigmoid() * 2) ** 2 * anchors

        po = predictions[..., 4:5]
        pc = predictions[..., 5:]
        pbox = torch.cat((pxy, pwh), 4)

        # targets
        txy = target[..., 0:2]
        twh = target[..., 2:4]
        to = target[..., 4:5]
        tc = target[..., 5:]
        tbox = torch.cat((txy, twh), 4).to('cuda:0')

        # for i in torch.flatten(pwh):
        #     if i > 0 : print(i)

        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        ious = intersection_over_union(pbox[obj], tbox[obj]).detach()

        # noobj loss:
        no_object_loss = self.bce(po[noobj], to[noobj])

        if torch.sum(obj) > 0:
            # objectness loss:
            object_loss = self.mse(po[obj], ious * to[obj])
            # bbox loss
            box_loss = self.mse(pbox[obj], tbox[obj])
            # class loss
            class_loss = self.bce(pc[obj], tc[obj])
        else:
            # objectness loss:
            object_loss = self.mse(torch.zeros(1), torch.zeros(1))
            # bbox loss
            box_loss = self.mse(torch.zeros(1), torch.zeros(1))
            # class loss
            class_loss = self.bce(torch.zeros(1), torch.zeros(1))

        # print(to[noobj])
        # print(no_object_loss) #OK
        # print(object_loss)
        # print(box_loss)
        # print(class_loss)

        loss = self.lambda_obj * object_loss + \
               self.lambda_noobj * no_object_loss + \
               self.lambda_class * class_loss + \
               self.lambda_bbox + box_loss

        #print(loss)

        return loss

