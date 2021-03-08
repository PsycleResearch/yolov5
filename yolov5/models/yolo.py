import logging
import math
from copy import deepcopy

import torch
import torch.nn as nn

from yolov5.models.common import Conv, Bottleneck, SPP, Focus, BottleneckCSP, Concat
from yolov5.utils.general import check_anchor_order, make_divisible
from yolov5.utils.torch_utils import (
    fuse_conv_and_bn, model_info, initialize_weights)

logger = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nb_classes=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nb_classes = nb_classes
        self.nb_outputs_per_anchor = nb_classes + 5
        self.nb_detection_layers = len(anchors)
        self.nb_anchors = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nb_detection_layers  # init grid
        a = torch.tensor(anchors).float().view(self.nb_detection_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nb_detection_layers, nb_anchors, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nb_detection_layers, 1, -1, 1, 1, 2))  # shape(nb_detection_layers, 1, nb_anchors, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.nb_outputs_per_anchor * self.nb_anchors, 1) for x in ch)  # output conv

    def forward(self, x):
        inference_output = []
        for i in range(self.nb_detection_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.nb_anchors, self.nb_outputs_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                inference_output.append(y.view(bs, -1, self.nb_outputs_per_anchor))

        return x if self.training else (torch.cat(inference_output, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg: dict, channels=3, nb_classes=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        # Loading pretrained weights
        self.yaml = cfg  # model dict

        # Define model
        self.yaml['nb_classes'] = nb_classes  # override pretrained weights nb classes
        self.model, self.save = parse_model(deepcopy(self.yaml), input_channels=[channels])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, channels, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def forward(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.nb_anchors, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nb_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(model_dict, input_channels):
    ##########
    # Replace "nc" in dict by "nb_classes"
    assert model_dict['head'][-1][3][0] == 'nc'
    model_dict['head'][-1][3][0] = 'nb_classes'
    ###########

    anchors, nb_classes, depth_multiple, width_multiple = model_dict['anchors'], model_dict['nb_classes'], model_dict['depth_multiple'], model_dict['width_multiple']
    nb_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    nb_outputs = nb_anchors * (nb_classes + 5)

    layers, save, output_channels = [], [], input_channels[-1]
    for i, (_from, _number, _module, _args) in enumerate(model_dict['backbone'] + model_dict['head']):
        _module = eval(_module) if isinstance(_module, str) else _module  # eval strings
        for j, a in enumerate(_args):
            try:
                _args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        _number = max(round(_number * depth_multiple), 1) if _number > 1 else _number  # depth gain
        if _module in [nn.Conv2d, Conv, Bottleneck, SPP, Focus, BottleneckCSP]:
            channels, output_channels = input_channels[_from], _args[0]
            output_channels = make_divisible(output_channels * width_multiple, 8) if output_channels != nb_outputs else output_channels

            _args = [channels, output_channels, *_args[1:]]
            if _module is BottleneckCSP:
                _args.insert(2, _number)
                _number = 1
        elif _module is nn.BatchNorm2d:
            _args = [input_channels[_from]]
        elif _module is Concat:
            output_channels = sum([input_channels[-1 if x == -1 else x + 1] for x in _from])
        elif _module is Detect:
            _args.append([input_channels[x + 1] for x in _from])
            if isinstance(_args[1], int):  # number of anchors
                _args[1] = [list(range(_args[1] * 2))] * len(_from)
        else:
            output_channels = input_channels[_from]

        m_ = nn.Sequential(*[_module(*_args) for _ in range(_number)]) if _number > 1 else _module(*_args)  # module
        t = str(_module)[8:-2].replace('__main__.', '')  # module type
        nb_params = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, _from, t, nb_params  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([_from] if isinstance(_from, int) else _from) if x != -1)  # append to savelist
        layers.append(m_)
        input_channels.append(output_channels)
    return nn.Sequential(*layers), sorted(save)
