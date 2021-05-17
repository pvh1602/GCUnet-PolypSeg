import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import math
from torch import einsum
from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0)     # Keep feature's size not change
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        
        return self.conv_block(x) + self.skip_connection(x)
    

class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super(UpSample, self).__init__()

        self.up_sample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        return self.up_sample(x)


class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



if __name__ == '__main__':
    upsample = UpSample(1024, 1024, 2, 2)
    x = torch.randn(1, 1024,16,16)
    y = upsample(x)
    print(y.shape)