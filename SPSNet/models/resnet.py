#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
from modules import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
resnet101_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

class BasicBlock(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            stride = 1,
            stride_at_1x1 = False,
            dilation = 1,
            *args, **kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = int(out_chan / 1)

        self.conv1 = nn.Conv2d(in_chan,
                mid_chan,
                kernel_size = 3,
                stride = stride1x1,
                padding = dilation,
                dilation = dilation,
                bias = False)
        self.bn1 = BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan,
                out_chan,
                kernel_size = 3,
                stride = stride3x3,
                padding = dilation,
                dilation = dilation,
                bias = False)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan, activation='none'))
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        if self.downsample == None:
            inten = x
        else:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Bottleneck(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            stride = 1,
            stride_at_1x1 = False,
            dilation = 1,
            *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = int(out_chan / 4)

        self.conv1 = nn.Conv2d(in_chan,
                mid_chan,
                kernel_size = 1,
                stride = stride1x1,
                bias = False)
        self.bn1 = BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan,
                mid_chan,
                kernel_size = 3,
                stride = stride3x3,
                padding = dilation,
                dilation = dilation,
                bias = False)
        self.bn2 = BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan,
                out_chan,
                kernel_size=1,
                bias=False)
        self.bn3 = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan, activation='none'))
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.downsample == None:
            inten = x
        else:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1):
    assert out_chan % 4 == 0
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, out_chan, stride=1, dilation=dilation))
    return nn.Sequential(*blocks)

def create_block(in_chan, out_chan, b_num, stride=1, dilation=1):
    assert out_chan % 4 == 0
    blocks = [BasicBlock(in_chan, out_chan, stride=stride, dilation=dilation),]
    for i in range(1, b_num):
        blocks.append(BasicBlock(out_chan, out_chan, stride=1, dilation=dilation))
    return nn.Sequential(*blocks)

class Resnet101(nn.Module):
    def __init__(self, inchannel=3, stride=32, *args, **kwargs):
        super(Resnet101, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el==1 else 1 for el in dils]

        self.conv1 = nn.Conv2d(
                inchannel,
                64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1,
                dilation = 1,
                ceil_mode = False)
        self.layer1 = create_stage(64, 256, 3, stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, 4, stride=2, dilation=1)
        self.layer3 = create_stage(512, 1024, 23, stride=strds[0], dilation=dils[0])
        self.layer4 = create_stage(1024, 2048, 3, stride=strds[1], dilation=dils[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet101_url)
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        return bn_params, non_bn_params

class Resnet50(nn.Module):
    def __init__(self, inchannel=3, stride=32, *args, **kwargs):
        super(Resnet50, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el==1 else 1 for el in dils]

        self.conv1 = nn.Conv2d(
                inchannel,
                64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1,
                dilation = 1,
                ceil_mode = False)
        self.layer1 = create_stage(64, 256, 3, stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, 4, stride=2, dilation=1)
        self.layer3 = create_stage(512, 1024, 6, stride=strds[0], dilation=dils[0])
        self.layer4 = create_stage(1024, 2048, 3, stride=strds[1], dilation=dils[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet50_url)
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        return bn_params, non_bn_params

class Resnet18(nn.Module):
    def __init__(self, inchannel=3, stride=32, *args, **kwargs):
        super(Resnet18, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el==1 else 1 for el in dils]

        self.conv1 = nn.Conv2d(
                inchannel,
                64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1,
                dilation = 1,
                ceil_mode = False)
        self.layer1 = create_block(64, 64, 2, stride=1, dilation=1)
        self.layer2 = create_block(64, 128, 2, stride=2, dilation=1)
        self.layer3 = create_block(128, 256, 2, stride=strds[0], dilation=dils[0])
        self.layer4 = create_block(256, 512, 2, stride=strds[1], dilation=dils[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        return bn_params, non_bn_params
    
class Resnet101_HDC(nn.Module):
    def __init__(self, inchannel=3, stride=32, *args, **kwargs):
        super(Resnet101_HDC, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el==1 else 1 for el in dils]

        self.conv1 = nn.Conv2d(
                inchannel,
                64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1,
                dilation = 1,
                ceil_mode = False)
        self.layer1 = create_stage(64, 256, 3, stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, 4, stride=2, dilation=1)
        # HDC
        blocks3 = [Bottleneck(512, 1024, stride=2, dilation=1),]
        for i in range(5):
            blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=2))
            blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=5))
            blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=9))
            blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=1))
        blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=2))
        blocks3.append(Bottleneck(1024, 1024, stride=1, dilation=5))
        self.layer3 = nn.Sequential(*blocks3)
        blocks4 = [Bottleneck(1024, 2048, stride=1, dilation=5),]
        blocks4.append(Bottleneck(2048, 2048, stride=1, dilation=9))
        blocks4.append(Bottleneck(2048, 2048, stride=1, dilation=17))
        self.layer4 = nn.Sequential(*blocks4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet101_url)
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        return bn_params, non_bn_params