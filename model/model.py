import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from arguments import *


args = get_args_training()
if args.backbone == 'ResEncoder':
    from .backbones import ResEncoder as encoder
elif args.backbone == 'Resnet50':
    from .backbones.ResNet import resnet50
    encoder = resnet50(pretrained=True)
elif args.backbone == 'Resnet34':
    from .backbones.ResNet import resnet34
    encoder = resnet34(pretrained=True)
elif args.backbone == 'Resnet18':
    from .backbones.ResNet import resnet18
    encoder = resnet18(pretrained=True)
    
if args.bridge == 'ResBridge':
    from .bridges import ResBridge as bridge
elif args.bridge == 'MHSABridge':
    from .bridges import MHSABridge as bridge

if args.decoder == 'ResDecoder':
    from .decoders import ResDecoder as decoder


class Unet(nn.Module):
    def __init__(self, in_channels, filters=[64, 128, 256, 512], fmap_size=(args.train_size, args.train_size), num_block=args.n_blocks):
        super(Unet, self).__init__()
        filters = [2**(i+6) for i in range(args.n_filters)]
        filters = [64, 256, 512, 1024, 1024]
        self.enc_filters = filters[:-1]
        self.bridge_in_channels = filters[-2]
        self.bridge_out_channels = filters[-1]
        self.dec_filters = filters
        self.fmap_size = list(fmap_size)
        self.bridge_in_fmap_size = [
            int(self.fmap_size[0] / pow(2, len(self.enc_filters)-1)),
            int(self.fmap_size[1] / pow(2, len(self.enc_filters)-1))
        ]
        if 'Resnet' in args.backbone:
            self.encoder = encoder
        else:
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
        print("Encoder")
        for i, feature in enumerate(features):
            print(f"feature {i} shape is: {feature.shape}")
        
        print("Bridge")
        out_bridge = self.bridge(features[-1])
        print(f"Bridge feature shape is: {out_bridge.shape}")
        print("Decoder")
        outp = self.decoder(out_bridge, features)
        print(f"Output shape is: {outp.shape}")
        return outp


if __name__ == '__main__':
    x = torch.randn(1,3,256,256).to('cuda')
    model = Unet(in_channels=3).to('cuda')
    outp = model(x)
    
    summary(model, (3,256,256))
    # print(outp.shape)

    
