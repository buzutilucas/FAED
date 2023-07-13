# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *
from lib import util

class Config(object):
    def __init__(self, model: str) -> None:
        self._model = model

        if self._model == 'vq_vae':
            self._beta = 0.25

            self._in_channels = 3
            self._num_hiddens = 128
            self._num_residual_layers = 2
            self._num_residual_hiddens = 32

            # This value is not that important, usually 64 works.
            # This will not change the capacity in the information-bottleneck.
            self._embedding_dim = 2

            # The higher this value, the higher the capacity in the information bottleneck.
            self._num_embeddings = 512 # For VQ-VAE num embeddings is 512

        if self._model == 'dcgan':
            self.channel_img = 3 # do not change
            self.z_dim = 128
            self.feature_d = 32
            self.feature_g = 512

        if self._model == 'wgan_gp':
            self.channel_img = 3 # do not change
            self.in_size = 128 # do not change
            self.z_dim = 128
            self.feature_d = 32
            self.feature_g = 128
    
    def __call__(self, isgen: bool=False) -> Dict[str, Any]:
        if self._model == 'vq_vae':
            return util.EasyDict(
                in_channels=self._in_channels,
                out_channels=self._num_hiddens,
                num_residual_layers=self._num_residual_layers,
                out_channels_residual=self._num_residual_hiddens,
                embedding_dim=self._embedding_dim,
                num_embeddings=self._num_embeddings,
                beta=self._beta
            )
        
        if self._model == 'dcgan':
            if isgen:
                # config generator
                return util.EasyDict(
                    z_dim=self.z_dim,
                    channel_img=self.channel_img,
                    feature_g=self.feature_g
                )
            # config discriminator
            return util.EasyDict(
                channel_img=self.channel_img,
                feature_d=self.feature_d
            )

        if self._model == 'wgan_gp':
            if isgen:
                # config generator
                return util.EasyDict(
                    z_dim=self.z_dim,
                    channel_img=self.channel_img,
                    feature_g=self.feature_g
                )
            # config critic
            return util.EasyDict(
                channel_img=self.channel_img,
                feature_d=self.feature_d,
                in_size = self.in_size
            )