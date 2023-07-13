# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *
from torch import nn
from abc import abstractmethod


class BaseAE(nn.Module):
    """ Base autoencoder class """
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encoder(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def decoder(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class BaseDiscriminator(nn.Module):
    """ Base discriminator class """
    def __init__(self) -> None:
        super(BaseDiscriminator, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class BaseGenerator(nn.Module):
    """ Base generator class """
    def __init__(self) -> None:
        super(BaseGenerator, self).__init__()

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        pass