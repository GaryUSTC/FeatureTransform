import torch
import torch.nn as nn
import math
from collections import OrderedDict
# from torch.nn import SyncBatchNorm2d
import sys

import time

BN = None
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BN(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BN(oup),
        nn.ReLU(inplace=True)
    )


def deconv_bn(inp, oup, kernel, strd, pad):
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, kernel, stride=strd, padding=pad),
        BN(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidualNoGroup(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualNoGroup, self).__init__()
        # added by gaowei
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BN(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        for name in self.names:
            module = self.conv._modules[name]
            x = module(x)
        ##            print ("zz size ",x.size())
        ##            print ('module', module)

        if self.use_res_connect:
            return t + x
        else:
            return x


class BumblebeeNaiveNoGroup(nn.Module):
    def __init__(self, input_dim, label_dim, group_size, group, sync_stats, deconv_setting,
                 interverted_residual_setting, input_size=224, width_mult=1.):
        super(BumblebeeNaiveNoGroup, self).__init__()
        # setting of sync BN
        global BN

        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs)
            # return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=sync_stats, eps=1e-5,
            #                        momentum=0.05)

        BN = BNFunc

        self.deconv_setting = deconv_setting
        self.interverted_residual_setting = interverted_residual_setting

        # add feature map from input img

        ##        self.addition_mid_map_layer =[]

        ##        self.addition_mid_map_layer.append(InvertedResidualV2(3, 3 , 2, 4,1))
        ##        self.addition_mid_map_layer.append(InvertedResidualV2(3, 3 , 2, 4,1))
        ##        self.addition_mid_map_layer.append(InvertedResidualV2(3, 3 , 1, 1,1))

        ##        self.addition_mid_map_layer = nn.Sequential(*self.addition_mid_map_layer)

        # building first layer
        assert input_size % 32 == 0
        self.first_deconv_channel = input_dim
        self.first_conv_channel = 16
        self.mid_channel = 3
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        input_channel = int(self.first_deconv_channel * width_mult)

        self.deconv_features = []

        for c, k, s, p in self.deconv_setting:
            output_channel = int(c * width_mult)
            self.deconv_features.append(deconv_bn(input_channel, output_channel, k, s, p))
            input_channel = output_channel

        self.deconv_features.append(nn.ConvTranspose2d(input_channel, self.mid_channel, 4, stride=2, padding=1))
        self.deconv_features = nn.Sequential(*self.deconv_features)

        self.features = []
        self.features.append(nn.Tanh())
        output_channel = int(self.first_conv_channel * width_mult)
        self.features.append(nn.Conv2d(self.mid_channel * 2, output_channel, 1, stride=1))
        self.features.append(BN(output_channel))
        input_channel = output_channel

        # self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidualNoGroup(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidualNoGroup(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(8, ceil_mode=True))  # such that easily converted to caffemodel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.last_channel, label_dim),
        )

        self._initialize_weights()

    def forward(self, img, x):

        ##        mid = self.addition_mid_map_layer(img)
        ##        print ("mid size",mid.size())

        x = x.view(-1, self.first_deconv_channel, 1, 1)
        x = self.deconv_features(x)

        ##        print  ("deconv size",x.size())

        x = torch.cat((x, img), 1)

        ##        print  ("concate size", x.size())

        x = self.features(x)
        x = x.view(-1, self.last_channel)
        ##        print ("output size", x.size())
        x = self.classifier(x)
        ##        print ("last size", x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def BumblebeeNaiveV5NoGroup(load_pretrain=True, T=6, input_dim=128, label_dim=128, group_size=1, group=None,
                            sync_stats=False, input_size=224, width_mult=1.):
    deconv_setting = []
    # setting of deconv blocks
    deconv_setting = [
        # c, k, s, p
        [128, 4, 1, 0],
        [64, 4, 2, 1],
        [32, 4, 2, 1],
        [8, 4, 2, 1],
    ]
    # setting of inverted residual blocks
    interverted_residual_setting = [
        # t, c, n, s
        [T, 8, 3, 2],
        [T, 16, 4, 2],
        [T, 24, 3, 1],
        [T, 40, 3, 2],
        [T, 80, 1, 1],
    ]

    model = BumblebeeNaiveNoGroup(input_dim, label_dim, group_size, group, sync_stats, deconv_setting,
                                  interverted_residual_setting)
    return model


BumblebeeNaiveV5NoGroup()
