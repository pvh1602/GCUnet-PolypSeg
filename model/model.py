import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from arguments import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = get_args_training()

if 'Resnet' in args.bridge:
    assert args.backbone == args.bridge, f'The bridge {args.bridge} need backbone {args.bridge} but get {args.backbone} backbone!'

if args.backbone == 'ResEncoder':
    from .backbones import ResEncoder as encoder
elif args.backbone == 'Resnet50':
    from .backbones.ResNet import resnet50
    encoder = resnet50(pretrained=args.pretrained)
elif args.backbone == 'Resnet34':
    from .backbones.ResNet import resnet34
    encoder = resnet34(pretrained=args.pretrained)
elif args.backbone == 'Resnet18':
    from .backbones.ResNet import resnet18
    encoder = resnet18(pretrained=args.pretrained)
else:
    assert False, f"Do not exist {args.backbone} backbone!"
    

if args.bridge == 'ResBridge':
    from .bridges import ResBridge as bridge
elif args.bridge == 'MHSABridge':
    from .bridges import MHSABridge as bridge
elif args.bridge == 'Resnet50':
    from .bridges.ResnetBridge import resnet50
    bridge = resnet50(pretrained=args.pretrained)
elif args.bridge == 'Resnet34':
    from .bridges.ResnetBridge import resnet34
    bridge = resnet34(pretrained=args.pretrained)
elif args.bridge == 'Resnet18':
    from .bridges.ResnetBridge import resnet18
    bridge = resnet18(pretrained=args.pretrained)
else:
    assert False, f"Do not exist {args.bridge} bridge!"


if args.decoder == 'ResDecoder':
    from .decoders import ResDecoder as decoder
elif args.decoder == 'SimpleDecoder':
    from .decoders import SimpleDecoder as decoder
else:
    assert False, f"Do not exist {args.decoder} decoder!"




filters_set = [
    [64, 128, 256, 512],
    [64, 128, 256, 512, 512],
    [64, 64, 128, 256, 512],
    [64, 256, 512, 1024, 512],
    ]

fmap_bridge_set = [
    (32, 32),
    (16, 16),
    (44, 44),
    (22, 22)
]



class Unet(nn.Module):
    def __init__(self, in_channels, filters=[64, 128, 256, 512], fmap_size=(args.train_size, args.train_size), num_block=args.n_blocks):
        super(Unet, self).__init__()

        # Process the filters and fmap of backbone and bridge
        if args.backbone == 'ResEncoder' and args.n_filters == 4:
            filters = filters_set[0]
            fmap_size = (int(args.train_size / 8), int(args.train_size / 8))
        elif args.backbone == 'ResEncoder' and args.n_filters == 5:
            filters = filters_set[1]
            fmap_size = (int(args.train_size / 8), int(args.train_size / 8))
        elif args.backbone == 'Resnet34' or args.backbone == 'Resnet18':
            filters = filters_set[2] 
            fmap_size = (int(args.train_size / 16), int(args.train_size / 16))
        elif args.backbone == 'Resnet50':
            filters = filters_set[3] 
            fmap_size = (int(args.train_size / 16), int(args.train_size / 16))
        else:
            assert False, f'Do not exist {args.backbone} backbone!'

        self.enc_filters = filters[:-1]
        self.bridge_in_channels = filters[-2]
        self.bridge_out_channels = filters[-1]
        self.dec_filters = filters
        self.bridge_in_fmap_size = fmap_size

        # Backbone
        if 'Resnet' in args.backbone:
            self.encoder = encoder
        else:
            self.encoder = encoder(in_channels=in_channels, filters=self.enc_filters, fmap_size=fmap_size)

        # Bridge
        if 'Resnet' in args.bridge:
            self.bridge = bridge
        else:
            self.bridge = bridge(
                                in_channels=self.bridge_in_channels, 
                                out_channels=self.bridge_out_channels, 
                                fmap_size=tuple(self.bridge_in_fmap_size),
                                num_block=args.n_blocks
                                )
        
        # Decoder
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
    # x = torch.randn(1,3,256,256).to(device)
    model = Unet(in_channels=3).to(device)
    # outp = model(x)
    
    summary(model, (3,256,256))
    # print(outp.shape)

    
