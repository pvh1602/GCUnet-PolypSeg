import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *


class ResEncoder(nn.Module):
    def __init__(self, in_channels: int, filters=[64, 128, 256, 512], fmap_size=(256,256)):
        super(ResEncoder, self).__init__()

        self.fmap_size = list(fmap_size)
        self.filters = filters

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.layers = nn.ModuleList()
        for i in range(len(filters)-1):
            self.layers.append(
                ResidualBlock(filters[i], filters[i+1], stride=2, padding=1)
            )
            self.fmap_size[0] = int(self.fmap_size[0] / 2)
            self.fmap_size[1] = int(self.fmap_size[1] / 2)


    def forward(self, x):
        
        features = []
        x = self.input_layer(x) + self.input_skip(x)
        features.append(x)
        for i in range(len(self.filters)-1):
            x = self.layers[i](x)
            features.append(x)

        return features

    

if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    model = ResEncoder(in_channels=3)
    features = model(x)
    for i in range(len(features)):
        print(f"feature {i+1} shape is: {features[i].shape}")