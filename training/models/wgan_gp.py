# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *

from torch import nn
from lib import persistence
from .layers import ConvBlock
from .base import BaseDiscriminator, BaseGenerator


@persistence.persistent_class
class Critic(BaseDiscriminator):
    """ Critic Class of WGAN """
    def __init__(self, channel_img: int, feature_d: int, in_size: int) -> None:
        super().__init__()

        layers = []
        layers.append(ConvBlock(
            in_channels=channel_img, 
            out_channels=feature_d, 
            kernel_size=4, 
            stride=2, 
            padding=1,
            activation=nn.LeakyReLU(0.2),
            bias=True,
            out_size=in_size//2,
            norm='lnorm',
        )) # down scale

        in_channels = feature_d
        for iter_num in range(6):
            out_channels = in_channels * 2 if iter_num < 4 else in_channels
            layers.append(ConvBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                activation=nn.LeakyReLU(0.2),
                bias=True,
                out_size=in_size//(2**(iter_num+2)),
                norm='lnorm',
            )) # down scale
            in_channels = out_channels

        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))

        self.disc = nn.Sequential(*layers)
        self._initialize_weights(self.disc)

    def forward(self, x: Tensor) -> Tensor:
        return self.disc(x)

    def _initialize_weights(self, model: Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@persistence.persistent_class
class Generator(BaseGenerator):
    """ Generator Class """
    def __init__(self, z_dim: int, channel_img: int, feature_g: int) -> None:
        super().__init__()

        self.z_dim = z_dim

        layers = []
        layers.append(ConvBlock(
            in_channels=z_dim, 
            out_channels=feature_g, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            activation=nn.LeakyReLU(0.2),
            scale_factor=2,
            norm='bnorm',
        )) # up scale

        features_list = [feature_g, feature_g*2, feature_g*4, feature_g*4, feature_g*2, feature_g]
        for i, out_channels in enumerate(features_list):
            layers.append(ConvBlock(
                in_channels=feature_g, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                activation=nn.LeakyReLU(0.2),
                scale_factor=2,
                norm='bnorm',
            )) # up scale
            layers.append(ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                activation=nn.LeakyReLU(0.2),
                norm='bnorm',
            )) # same scale
            feature_g = out_channels

        # To RGB layer
        layers.append(nn.Conv2d(
            in_channels=out_channels,
            out_channels=channel_img,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)
        self._initialize_weights(self.gen)
    
    def forward(self, z: Tensor) -> Tensor:
        return self.gen(z)
    
    def _initialize_weights(self, model: Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
