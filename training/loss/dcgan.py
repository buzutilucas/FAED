# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *

import torch
from lib import misc
from lib import training_stats
from .base import Baseloss


class Loss(Baseloss):
    """ Loss """
    def __init__(self, device: str, G: Module, D: Module) -> None:
        super(Loss, self).__init__()

        self.device = device
        self.G = G
        self.D = D

        # Establish convention for real and fake labels during training
        self.real_label = 0.9 # smoothing
        self.fake_label = 0.1 # smoothing

    def run_G(self, z, sync):
        with misc.ddp_sync(self.G, sync):
            img = self.G(z)
        return img

    def run_D(self, img, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img)
        return torch.sigmoid(logits)

    def accumulate_gradients(self, phase, real_img, gen_z, sync):
        assert phase in ['Gmain', 'Dmain']
        do_Gmain = (phase == 'Gmain')
        do_Dmain = (phase == 'Dmain')

        # Update G network: maximize log(D(G(z)))
        if do_Gmain:
            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Fake labels are real for generator cost
                b_size = real_img.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                gen_img = self.run_G(gen_z, sync=sync)
                fake_output = self.run_D(gen_img, sync=False).view(-1)
                # -(t*logy+(1-t)*log(1-y))
                loss_Gmain = -(label*torch.log(fake_output) + (1 - label)*torch.log(1-fake_output))
                training_stats.report('Loss/G', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().backward()

        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if do_Dmain:
            b_size = real_img.size(0)
            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # fake labels of fake data for discriminator cost
                label = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device)
                gen_img = self.run_G(gen_z, sync=False)
                fake_output = self.run_D(gen_img, sync=False).view(-1) # Gets synced by loss_Dreal.
                # -(t*logy+(1-t)*log(1-y))
                loss_Dgen = -(label*torch.log(fake_output) + (1 - label)*torch.log(1-fake_output))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                # Real labels for discriminator cost
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_output = self.run_D(real_img_tmp, sync=sync).view(-1)
                # -(t*logy+(1-t)*log(1-y))
                loss_Dreal = -(label*torch.log(real_output) + (1 - label)*torch.log(1-real_output))
            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.mean().backward()
            
            loss_D = loss_Dreal + loss_Dgen
            training_stats.report('Loss/D', loss_D)