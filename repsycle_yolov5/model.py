import torch
from torch import nn
import math
import torch.nn.functional as F
import os
import platform
from pathlib import Path
from common import *
import config

class Model(torch.nn.Module):

    def __init__(self, nb_classes=1, nb_channels=3):
        super(Model, self).__init__()

        self.nb_classes = nb_classes

        cd = 2
        wd = 3

        # # Backbone : CSPNet : https://arxiv.org/abs/1911.11929
        #
        # self.focus = Focus(nb_channels,  64 // 2, 3, 1)
        # self.conv1 = Conv(64 // 2, 128 // 2, 3, 2)
        # self.csp1 = BottleneckCSP(128, 128, n=1)
        #
        # self.conv2 = Conv(128 // 2, 256 // 2, 3, 2)
        # self.csp2 = BottleneckCSP(256, 256, n=1)
        #
        # self.conv3 = Conv(256 // 2, 512 // 2, 3, 2)
        # self.csp3 = BottleneckCSP(512 // 2, 512 // 2, n=1)
        #
        # self.conv4 = Conv(512, 1024, 3, 2)
        # self.spp = SPP(1024, 1024)
        # self.csp4 = BottleneckCSP(1024, 1024, n=1, shortcut=False)
        #
        # # Neck : PANet : https://arxiv.org/abs/1803.01534
        #
        # self.conv5 = Conv(1024, 512)
        # self.up1 = nn.Upsample(scale_factor=2)
        # self.csp5 = BottleneckCSP(1024, 512, n=1, shortcut=False)
        #
        # self.conv6 = Conv(512, 256)
        # self.up2 = nn.Upsample(scale_factor=2)
        # self.csp6 = BottleneckCSP(512, 256, n=1, shortcut=False)
        #
        # self.conv7 = Conv(256, 256, 3, 2)
        # self.csp7 = BottleneckCSP(512, 512, n=1, shortcut=False)
        #
        # self.conv8 = Conv(512, 512, 3, 2)
        # self.csp8 = BottleneckCSP(1024, 1024, n=1, shortcut=False)

        # Backbone : CSPNet : https://arxiv.org/abs/1911.11929

        self.focus = Focus(nb_channels,  64 // cd, 3 , 1)
        self.conv1 = Conv(64 // cd, 128 // cd, 3, 2)
        self.csp1 = BottleneckCSP(128 // cd, 128 // cd, n = 3 // wd)

        self.conv2 = Conv(128 // cd, 256 // cd, 3, 2)
        self.csp2 = BottleneckCSP(256 // cd, 256 // cd, n= 9 // wd)

        self.conv3 = Conv(256 // cd, 512 // cd, 3, 2)
        self.csp3 = BottleneckCSP(512 // cd, 512 // cd, n = 9 // wd)

        self.conv4 = Conv(512 // cd, 1024 // cd, 3, 2)
        self.spp = SPP(1024 // cd , 1024 // cd)
        self.csp4 = BottleneckCSP(1024 // cd , 1024 // cd, n = 3 // wd, shortcut=False)

        # Neck : PANet : https://arxiv.org/abs/1803.01534

        self.conv5 = Conv(1024 // cd, 512 // cd)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = BottleneckCSP(1024 // cd, 512 // cd, n = 3 // cd, shortcut=False)

        self.conv6 = Conv(512 // cd, 256 // cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = BottleneckCSP(512 // cd, 256 // cd , n = 3 // wd, shortcut=False)

        self.conv7 = Conv(256 // cd, 256 // cd, 3, 2)
        self.csp7 = BottleneckCSP(512 // cd, 512 // cd, n = 3 // wd, shortcut=False)

        self.conv8 = Conv(512 // cd, 512 // cd, 3, 2)
        self.csp8 = BottleneckCSP(1024 // cd, 1024 // cd, n=3 // wd, shortcut=False)

        self.anchors = [[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]]

        self.nb_output = self.nb_classes + 5
        self.nb_layer = len(self.anchors)
        self.nb_anchors = len(self.anchors[0])
        self.grid = [torch.zeros(1)] * self.nb_layer
        self.stride = [8, 16, 32]
        self.batch_size = 1
        self.a = torch.tensor(self.anchors).float()
        self.register_buffer('anchorss', self.a)
        self.register_buffer('anchor_grid', self.a.clone().view(self.nb_layer, 1, -1, 1, 1, 2))

        # Head (classification)

        self.conv9 = nn.Conv2d(128, self.nb_output * self.nb_anchors, kernel_size = (1, 1), stride = (1, 1))
        self.conv10 = nn.Conv2d(256, self.nb_output * self.nb_anchors, kernel_size = (1, 1), stride = (1, 1))
        self.conv11 = nn.Conv2d(512, self.nb_output * self.nb_anchors, kernel_size = (1, 1), stride = (1, 1))

    def _build_backbone(self, x):

        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)

        x_p3 = self.conv2(x) #P3
        x = self.csp2(x_p3)

        x_p4 = self.conv3(x) #P4
        x = self.csp3(x_p4)

        x_p5 = self.conv4(x) #P5
        x = self.spp(x_p5)

        x = self.csp4(x)

        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, x):

        h_p5 = self.conv5(x)  # head P5
        x = self.up1(h_p5)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x_concat)

        return x_small, x_medium, x_large

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _build_detect(self, xs, xm, xl):

        convs = [self.conv9, self.conv10, self.conv11]
        x_ = [xs, xm, xl]
        z = []
        i = 0

        for x, conv in zip(x_, convs):

            x = conv(x)

            # Reformat X (batch_size,nb_classes * nb_anchors, grid_y, grid_x) to x(bs,3,20,20,85)
            batch_size, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x = x.view(batch_size, self.nb_anchors, self.nb_output, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:
            #     if self.grid[i].shape[2:4] != x.shape[2:4]:
            #         self.grid[i] = self._make_grid(nx, ny).to(x.device)
            #
            #     y = x.sigmoid()
            #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x.device)) * self.stride[i]  # xy
            #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            #     z.append(y.view(batch_size, -1, self.nb_output))

            x_[i] = x

            i+=1

        return x_ #if self.training else (torch.cat(z, 1), x_)

    def forward(self, x):
        p3, p4, p5, x = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, x)
        x_ = self._build_detect(xs, xm, xl)
        return x_

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    file = Path(weights).name
    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models
    if file in models and not os.path.isfile(weights):
        print(file)
        # Google Drive
        # d = {'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',
        #      'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',
        #      'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',
        #      'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS'}
        # r = gdrive_download(id=d[file], name=weights) if file in d else 1
        # if r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6:  # check
        #    return

        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            if platform.system() == 'Darwin':  # avoid MacOS python requests certificate error
                r = os.system('curl -L %s -o %s' % (url, weights))
            else:
                torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return

def create(name, pretrained, channels, classes):

    model = Model(classes, channels)

    # state_dict = torch.load(name, map_location=torch.device('cuda:0'))['model'].float().state_dict()  # to FP32
    #
    # dict_load = list(state_dict.items())
    # dict_model = list(model.state_dict().items())
    # new_state_dict = {}
    #
    # for i in range(len(dict_model)):
    #     if dict_model[i][1].shape == dict_load[i][1].shape:
    #         key=dict_model[i][0]
    #         value=dict_load[i][1]
    #         new_state_dict[key] = value
    #     else :
    #         key = dict_model[i][0]
    #         value = dict_model[i][1]
    #         new_state_dict[key] = value
    #
    # model.load_state_dict(new_state_dict)

    return model

if __name__ == '__main__':

    model = create('yolov5s.pt', pretrained=True, channels=3, classes=2)

    model.eval()
    img = torch.rand((1, 3, 640, 640))
    x_ = model(img)

    for out in x_:
        print(out.shape)

    model.train()
    img = torch.rand((1, 3, 640, 640))
    o = model(img)
    for out in o:
        print(out.shape)