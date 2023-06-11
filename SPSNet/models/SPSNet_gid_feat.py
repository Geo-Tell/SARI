import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet101_HDC as Resnet101
from modules import InPlaceABNSync as BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=4, padding=4)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan*5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan*4, out_chan, ks=1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan=[1024,512,256], *args, **kwargs):
        super(Decoder, self).__init__()
        self.conv_feat16 = ConvBNReLU(low_chan[0], 256, ks=3, padding=1)
        self.conv_feat8 = ConvBNReLU(low_chan[1], 128, ks=3, padding=1)
        self.conv_feat4 = ConvBNReLU(low_chan[2], 64, ks=3, padding=1)
        self.conv_fuse1 = ConvBNReLU(256, 128, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(128, 64, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(64, 64, ks=3, padding=1)
        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, feat4_low, feat8_low, feat16_low, feat_aspp, feat_map=True):
        H, W = feat16_low.size()[2:]
        feat4_low = self.conv_feat4(feat4_low)
        feat8_low = self.conv_feat8(feat8_low)
        feat16_low = self.conv_feat16(feat16_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',
                align_corners=True)
        feat_out = self.conv_fuse1(feat16_low+feat_aspp_up)
        H, W = feat8_low.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                align_corners=True)
        feat_out = self.conv_fuse2(feat_out+feat8_low)        
        H, W = feat4_low.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                align_corners=True)
        feat_out = self.conv_fuse3(feat_out+feat4_low)   
        logits = self.conv_out(self.fuse(feat_out))
        if feat_map:
            return feat_out, logits
        else:
            return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SPSNet(nn.Module):
    def __init__(self, cfg, bashArgs, *args, **kwargs):
        super(SPSNet, self).__init__()
        if(bashArgs.Backbone=="Resnet101"):
            self.backbone = Resnet101(inchannel=4,stride=16)
            self.aspp = ASPP(in_chan=2048, out_chan=256, with_gp=cfg.aspp_global_feature)
            self.decoder = Decoder(cfg.n_classes, low_chan=[1024,512,256])
        self.init_weight()

    def forward(self, x, feat_map=True):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        if feat_map:
            feature_out, logits = self.decoder(feat4, feat8, feat16, feat_aspp, feat_map)
            feature_out = F.interpolate(feature_out, (H, W), mode='bilinear', align_corners=True)
        else:
            logits = self.decoder(feat4, feat8, feat16, feat_aspp, feat_map)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
        if feat_map:
            return feature_out, logits
        else:
            return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.backbone.get_params()
        tune_wd_params = list(self.aspp.parameters())  \
                + list(self.decoder.parameters())  \
                + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params