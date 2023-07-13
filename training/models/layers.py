# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *

from torch import nn
from lib import persistence
from lib.util import UserError


@persistence.persistent_class
class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int,
        activation: nn.Module,
        bias: bool=False,
        is_transpose: bool=False,
        scale_factor: int=0,
        out_size: int=2,
        norm: str='bnorm',
    ) -> None:
        super().__init__()

        layers = []
        if scale_factor > 0:
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))
        if is_transpose:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

        if norm == 'inorm':
            layers.append(nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))
        elif norm == 'bnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'lnorm':
            layers.append(nn.LayerNorm([out_channels, out_size, out_size]))
        else:
            raise UserError("'norm' must be 'inorm', 'bnorm' or 'lnorm'")
        
        layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)