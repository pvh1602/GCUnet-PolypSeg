import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from arguments import *

args = get_args_training()
if args.backbone == 'ResEncoder':
    from .backbones import ResEncoder as encoder
    
if args.bridge == 'ResBridge':
    from .bridges import ResBridge as bridge
elif args.bridge == 'MHSABridge':
    from .bridges import MHSABridge as bridge

if args.decoder == 'ResDecoder':
    from .decoders import ResDecoder as decoder

import torch
import torch.nn as nn
from core.modules import ResidualConv, Upsample


class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output



class Unet(nn.Module):
    def __init__(self, in_channels, filters=[64, 128, 256, 512], fmap_size=(args.train_size, args.train_size), num_block=args.n_blocks):
        super(Unet, self).__init__()
        self.enc_filters = filters[:-1]
        self.bridge_in_channels = filters[-2]
        self.bridge_out_channels = filters[-1]
        self.dec_filters = filters
        self.fmap_size = list(fmap_size)
        self.bridge_in_fmap_size = [
            int(self.fmap_size[0] / pow(2, len(self.enc_filters))),
            int(self.fmap_size[1] / pow(2, len(self.enc_filters)))
        ]

        self.encoder = encoder(in_channels=in_channels, filters=self.enc_filters, fmap_size=fmap_size)
        self.bridge = bridge(
                            in_channels=self.bridge_in_channels, 
                            out_channels=self.bridge_out_channels, 
                            fmap_size=tuple(self.bridge_in_fmap_size),
                            num_block=1
                            )
        self.decoder = decoder(in_channels=1024, filters=self.dec_filters)

    def forward(self, x):
        features = self.encoder(x)
        # print("Encoder")
        # for i, feature in enumerate(features):
            # print(f"feature {i} shape is: {feature.shape}")
        
        # print("Bridge")
        out_bridge = self.bridge(features[-1])
        # print(f"Bridge feature shape is: {out_bridge.shape}")
        # print("Decoder")
        outp = self.decoder(out_bridge, features)
        # print(f"Output shape is: {outp.shape}")
        return outp


if __name__ == '__main__':
    x = torch.randn(1,3,256,256).to('cuda')
    model = Unet(in_channels=3).to('cuda')
    outp = model(x)
    
    summary(model, (3,256,256))
    # print(outp.shape)

    
