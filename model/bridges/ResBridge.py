import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)     # Keep feature's size not change
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        
        return self.conv_block(x) + self.skip_connection(x)

class ResBridge(nn.Module):
    def __init__(self, in_channels, out_channels, fmap_size=None, num_block=None):
        super(ResBridge, self).__init__()
        self.bridge = ResidualBlock(in_channels, out_channels, stride=2, padding=1)

    def forward(self, x):
        return self.bridge(x)