import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
from arguments import *
args = get_args_training()


class ResDecoder(nn.Module):
    def __init__(self, in_channels=None, filters=[64, 128, 256, 512, 1024], resolution=(256, 256)):
        super(ResDecoder, self).__init__()
        self.filters = list(reversed(filters))  # 1024 512 256 128 64
        # print(self.filters)
        
        self.resolution = list(resolution)      
        self.layers = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        for i in range(len(self.filters)-1):
            self.layers.append(
                ResidualBlock(self.filters[i] + self.filters[i+1], self.filters[i+1], stride=1, padding=1)
            )
            self.up_sample_layers.append(
                UpSample(self.filters[i], self.filters[i], kernel_size=2, stride=2)
            )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.filters[-1], 1, 1, 1)    # Bx1xHxW
        )

    
    def forward(self, x, features):
        features = list(reversed(features))
        for i in range(len(self.layers)):
            # print(f"i = {i}")

            # print(f"x shape is {x.shape}")
            x = self.up_sample_layers[i](x)
            # print(f"After UpSample x shape is {x.shape}")
            x = torch.cat([x, features[i]], dim=1)  
            # print(f"After Concat x shape is {x.shape}")
            x = self.layers[i](x)
            # print(f"After ResBlock x shape is {x.shape}")
        if 'Resnet' in args.backbone or 'res2net50' in args.backbone:
            x = F.interpolate(x, size=args.train_size)


        x = self.output_layer(x)

        return x
        


if __name__ == '__main__':
    x = torch.randn(1,1024,16,16)
    features = []
    fm_size = 256
    for i in [64,128,256,512]:
        features.append(torch.randn(1, i, fm_size, fm_size))
        fm_size = int(fm_size / 2)
        print(f"features {i} has shape {features[-1].shape}")
    
    # print(features)
    decoder = ResDecoder()
    output = decoder(x, features)
    print(output.shape)
