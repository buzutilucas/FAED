# Copyright (c) 2022, The Images Processing Laboratory Authors. All Rights Reserved.
# Authors:
#

from lib.__typing__ import *

from torch import nn
from torch.nn import functional as F

from lib import persistence, misc

from .base import BaseAE
from .quantizer import VectorQuantizer


# 1x1 convolution
@misc.profiled_function
def conv1x1(in_channels: int, out_channels: int, stride: int=1) -> Tensor:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1,
        stride=stride, padding=0, bias=True
    )

# 3x3 convolution
@misc.profiled_function
def conv3x3(in_channels: int, out_channels: int, stride: int=1) -> Tensor:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=True
    )

# 4x4 convolution
@misc.profiled_function
def conv4x4(in_channels: int, out_channels: int, stride: int=1) -> Tensor:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=4,
        stride=stride, padding=1, bias=True
    )

# 4x4 convolution transpose
@misc.profiled_function
def conv_transpose4x4(in_channels: int, out_channels: int, stride: int=1) -> Tensor:
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=4, 
        stride=stride, padding=1, bias=True
    )


# Residual Block
@persistence.persistent_class
class ResidualBlock(nn.Module):
    """ Residual Block """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int=1, 
        downsample: Sequential=None
    ) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            conv3x3(out_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            conv1x1(out_channels, out_channels),
        )
        self.downsample = downsample
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out


@persistence.persistent_class
class Encoder(nn.Module):
    """ Encoder """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_residual_layers: int, 
        out_channels_residual: int
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_residual_layers = num_residual_layers
        self._out_channels_residual = out_channels_residual

        modules: List[Tensor] = []
        modules.append(
            nn.Sequential(
                conv4x4(self._in_channels, self._out_channels//2, stride=2), # 64x64
                nn.ReLU(inplace=True),
                conv4x4(self._out_channels//2, self._out_channels, stride=2), # 32x32
                nn.ReLU(inplace=True),
                conv3x3(self._out_channels, self._out_channels, stride=1),
                nn.ReLU(inplace=True),
            )
        )
        # Residual Stack
        modules.append(
            self._make_group(
                self._out_channels, self._out_channels_residual, self._num_residual_layers
            )
        )
        self._encode = nn.Sequential(*modules)

    # Residual Stack
    def _make_group(self, in_channels: int, out_channels: int, blocks: int, stride: int=1) -> Sequential:
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                conv1x1(out_channels, out_channels)
            )
        layers: List[Tensor] = []
        layers.append(
            ResidualBlock(
                in_channels, out_channels, stride=stride, downsample=downsample
            )
        )
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._encode(x)


@persistence.persistent_class
class Decoder(nn.Module):
    """ Decoder """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_residual_layers: int, 
        out_channels_residual: int
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_residual_layers = num_residual_layers
        self._out_channels_residual = out_channels_residual

        modules: List[Tensor] = []
        modules.append(
            conv3x3(self._in_channels, self._out_channels)
        )
        # Residual Stack
        modules.append(
            self._make_group(
                self._out_channels, self._out_channels_residual, self._num_residual_layers
            )
        )
        modules.append(
            nn.Sequential(
                conv_transpose4x4(self._out_channels_residual, self._out_channels, stride=2), # 64x64
                nn.ReLU(inplace=True),
                conv_transpose4x4(self._out_channels, 3, stride=2), # 128x128
            )
        )
        
        self._decode = nn.Sequential(*modules)

    # Residual Stack
    def _make_group(self, in_channels: int, out_channels: int, blocks: int, stride: int=1) -> Sequential:
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                conv1x1(out_channels, out_channels)
            )
        layers: List[Tensor] = []
        layers.append(
            ResidualBlock(
                in_channels, out_channels, stride=stride, downsample=downsample
            )
        )
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self._decode(z)


@persistence.persistent_class
class VQVAE(BaseAE):
    """
    Neural Discrete Representation Learning
    paper: https://arxiv.org/pdf/1711.00937.pdf
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        num_residual_layers: int, 
        out_channels_residual: int,
        embedding_dim: int=64,
        num_embeddings: int=512,
        beta: float=0.25,
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_residual_layers = num_residual_layers
        self._out_channels_residual = out_channels_residual
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._beta = beta

        self._encoder = Encoder(
            in_channels=self._in_channels, 
            out_channels=self._out_channels, 
            num_residual_layers=self._num_residual_layers, 
            out_channels_residual=self._out_channels_residual
        )
        self._pre_vq_conv = conv1x1(
            in_channels=self._out_channels_residual, out_channels=self._embedding_dim
        )
        self.vq = VectorQuantizer(
            K=self._num_embeddings, D=self._embedding_dim, beta=self._beta
        )
        self._decoder = Decoder(
            in_channels=self._embedding_dim, 
            out_channels=self._out_channels, 
            num_residual_layers=self._num_residual_layers, 
            out_channels_residual=self._out_channels_residual
        )
    
    def encoder(self, x: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) Latent codes
        """
        return self._pre_vq_conv(self._encoder(x))

    def decoder(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.

        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self._decoder(z)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        z = self.encoder(x)
        vq_output = self.vq(z)
        x_recon = self.decoder(vq_output['quantize'])
        return x_recon, vq_output