import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torchsummary import summary

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from model.modules.modules import BasicBlock, Bottleneck, LambdaBlock
from arguments import *
args = get_args_training()

__all__ = ['ResNet', 'lambdaresnet18', 'lambdaresnet34', 'lambdaresnet50']


model_urls = {
    'lambdaresnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'lambdaresnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'lambdaresnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.norm_conv1 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.norm_conv2 = nn.Conv2d(in_channels=self.inplanes, out_channels=256, kernel_size=1)

        self.lambdalayer3 = self._make_layer(LambdaBlock, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], lambdalayer=True)
        self.norm_conv3 = nn.Conv2d(in_channels=self.inplanes, out_channels=512, kernel_size=1)


        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2]) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck, LambdaBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, lambdalayer=False) -> nn.Sequential:
        

        layers = []
        # print(type(block))
        if lambdalayer:
            # print("Check")
            layers = []
            for i in range(blocks):
                is_first = i == 0
                stride = 2 if is_first else 1
                layers.append(
                    block(self.inplanes, int(planes / block.expansion), stride=stride, r=args.r)
                )
                self.inplanes = planes 
            
            
        else:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def load_pretrained_model(self):
        pass

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features = []
        features.append(x)          #       1 64 128 128
        x = self.maxpool(x)
        # print(f"features after conv and pooling {x.shape}")     # 1, 64, 64, 64
        x = self.layer1(x)
        features.append(x)      # 1, 64, 64, 64
        # print(f"features after layer 1 {x.shape}")
        x = self.layer2(x)  
        features.append(x)      # 1, 128, 32, 32
        # print(f"features after layer 2 {x.shape}")
        x = self.lambdalayer3(x)
        features.append(x)      # 1, 256, 16, 16
        # print(f"features after layer 3 {x.shape}")
        # x = self.layer4(x)      # Use BoT at here
        # features.append(x)      # 1, 512, 8, 8
        # print(f"features after layer 4 {x.shape}")

        return features 


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        pretrained_param_names = list(pretrained_state_dict.keys())
        # print("Pretrained params: ")
        # print(pretrained_param_names)

        state_dict = model.state_dict()
        param_names = list(state_dict.keys())
        # print("Model params: ")
        # print(param_names)

        for name in param_names:
            if name not in pretrained_param_names:
                continue
            # if state_dict[name].shape != pretrained_state_dict[name].shape
            state_dict[name] = pretrained_state_dict[name]

        model.load_state_dict(state_dict)
    return model


def lambdaresnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('lambdaresnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def lambdaresnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('lambdaresnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def lambdaresnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('lambdaresnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



if __name__ == '__main__':
    x = torch.randn(1,3,256,256).to('cuda')
    model = resnet50(pretrained=True).to('cuda')
    summary(model, (3,256,256))
    y = model(x)