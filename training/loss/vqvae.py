# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *

from lib import misc
from lib import training_stats
from .base import Baseloss

import torch
from torch.nn import functional as F


class Loss(Baseloss):
    """ Loss """
    def __init__(self, device: str, VQVAE: Module) -> None:
        super(Loss, self).__init__()

        self.device = device
        self.VQVAE = VQVAE

    def run_VQVAE(self, x, sync):
        with misc.ddp_sync(self.VQVAE, sync):
            img, vq_output = self.VQVAE(x)
        return img, vq_output

    def accumulate_gradients(self, img, sync, phase):
        with torch.autograd.profiler.record_function('VQVAEmain_forward'):
            recon_img, vq_output = self.run_VQVAE(img, sync=sync)
            img_tmp = img.detach().requires_grad_(False)
            recon_error = ((recon_img - img_tmp)**2).mean()
            loss_perplexity = vq_output['perplexity']
            loss_vq = vq_output['loss']
            loss_VQVAEmain = recon_error + loss_vq
            training_stats.report(f'{phase}/Loss/recon', recon_error)
            training_stats.report(f'{phase}/Loss/perplexity', loss_perplexity)
            training_stats.report(f'{phase}/Loss/vq', loss_vq)
            training_stats.report(f'{phase}/Loss/recon+vq', loss_VQVAEmain)
        # Only training phase computed loss
        if phase == 'train':
            with torch.autograd.profiler.record_function('VQVAEmain_backward'):
                loss_VQVAEmain.backward()