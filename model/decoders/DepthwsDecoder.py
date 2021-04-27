import torch
import torch.nn as nn
import torch.nn.functional as F


class SEnet(nn.Module):
    def __init__(self, in_channels, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(in_channels // reduction)
        self.conv_2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(in_channels)


    def forward(self, inputs):
        x = inputs
        b, c, _, _ = x.shape
        x = self.avg_pool(x).view(b, c, 1, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))   
        x = F.relu(self.bn_2(self.conv_2(x)))
        return inputs * x
        