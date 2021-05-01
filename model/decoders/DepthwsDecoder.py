import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from arguments import *
args = get_args_training()


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SEnet(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEnet, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
        

class DepthwsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, stride=1, padding=1):
        super(DepthwsBlock, self).__init__()
        expand_channels = in_channels * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, padding=1, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEnet(out_channels)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.skip_connection(x) + self.block(x)


class DepthwsDecoder(nn.Module):
    def __init__(self, in_channels=None, filters=[64, 128, 256, 512, 1024], resolution=(256, 256)):
        super(DepthwsDecoder, self).__init__()
        self.filters = list(reversed(filters))  # 1024 512 256 128 64
        # print(self.filters)
        
        self.resolution = list(resolution)      
        self.layers = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        for i in range(len(self.filters)-1):
            self.layers.append(
                DepthwsBlock(self.filters[i] + self.filters[i+1], self.filters[i+1], stride=1, padding=1)
            )
            self.up_sample_layers.append(
                nn.Upsample(scale_factor=2)
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
        if 'Resnet' in args.backbone:
            x = F.interpolate(x, size=args.train_size)


        x = self.output_layer(x)

        return x
        


if __name__ == '__main__':
    x = torch.randn(2,1024,16,16)
    features = []
    fm_size = 256
    for i in [64,128,256,512]:
        
        features.append(torch.randn(2, i, fm_size, fm_size))
        fm_size = int(fm_size / 2)
        print(f"features {i} has shape {features[-1].shape}")
    
    # print(features)
    decoder = DepthwsDecoder()
    pytorch_total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    # summary(decoder, (1024, 16, 16))
    output = decoder(x, features)
    print(output.shape)
    print(f"Trainable params: {pytorch_total_params}")
