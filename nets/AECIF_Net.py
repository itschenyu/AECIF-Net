from this import s
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification, CIF

class AECIF_Net(nn.Module):
    def __init__(self, num_classes=21, backbone='hrnetv2_w18', pretrained=False):
        super(AECIF_Net, self).__init__()
        self.backbone = hrnet_classification(backbone=backbone, pretrained=pretrained)
        last_inp_channels = np.int(np.sum(self.backbone.pre_stage_channels))
        mid_channels = 512
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(last_inp_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(last_inp_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cif = CIF(num_classes = num_classes, last_inp_channels = last_inp_channels)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x[0], x1, x2, x3], 1)  # 720*120*120
        out = self.conv3x3(x)  # 512*120*120
        out_1 = self.conv3x3_1(x) # 512*120*120
        x_fuse, x_1_fuse = self.cif(out, out_1)

        x = F.interpolate(x_fuse, size=(H, W), mode='bilinear', align_corners=True)
        x_1 = F.interpolate(x_1_fuse, size=(H, W), mode='bilinear', align_corners=True)

        return x, x_1