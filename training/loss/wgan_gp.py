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
    def __init__(self, device: str, G: Module, C: Module) -> None:
        super(Loss, self).__init__()

        self.device = device
        self.G = G
        self.C = C

        # The gradient penalty coefficien
        self.lambda_GP = 10

    def run_G(self, z, sync):
        with misc.ddp_sync(self.G, sync):
            img = self.G(z)
        return img

    def run_C(self, img, sync):
        with misc.ddp_sync(self.C, sync):
            logits = self.C(img)
        return logits

    def accumulate_gradients(self, phase, real_img, gen_z, sync):
        assert phase in ['Gmain', 'Cmain']
        do_Gmain = (phase == 'Gmain')
        do_Cmain = (phase == 'Cmain')

        # Update G network: minimize E[C(G(z)))]
        if do_Gmain:
            # Gmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, sync=sync)
                fake_logits = self.run_C(gen_img, sync=False).view(-1)
                loss_Gmain = -torch.mean(fake_logits)
                training_stats.report('Loss/G', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        # Update C network: maximize E[C(G(z)))] - E[C(x)] + l*E[(||gard(C(x_))||_2 - 1)^2]
        if do_Cmain:
            with torch.autograd.profiler.record_function('Cmain_forward'):
                with torch.autograd.profiler.record_function('Cgen_forward'):
                    gen_img = self.run_G(gen_z, sync=False)
                    fake_logits = self.run_C(gen_img, sync=False).view(-1) # Gets synced by real_logits.

                with torch.autograd.profiler.record_function('gradient_penalty'):
                    batch_size, C, H, W = real_img.shape
                    epsilon = torch.rand(batch_size,1,1,1).repeat(1,C,H,W).to(self.device)
                    x_hat = real_img*epsilon + gen_img*(1-epsilon)
                    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
                    mixed_logits = self.run_C(x_hat, sync=False).view(-1) # Gets synced by real_logits.
                    gradient = torch.autograd.grad(
                        inputs=x_hat,
                        outputs=mixed_logits,
                        grad_outputs=torch.ones_like(mixed_logits).to(self.device),
                        create_graph=True,
                    )[0]
                    gradient = gradient.view(gradient.shape[0], -1)
                    gradient_norm = torch.sqrt(torch.sum(gradient**2, dim = 1)) #gradient.norm(2, dim=1)
                    gradient_penalty = torch.mean((gradient_norm - 1)**2)
                    training_stats.report('Loss/gradient_penalty', gradient_penalty)

                with torch.autograd.profiler.record_function('Creal_forward'):
                        real_img_tmp = real_img.detach().requires_grad_(False)
                        real_logits = self.run_C(real_img_tmp, sync=sync).view(-1)

                with torch.autograd.profiler.record_function('Wasserstein_1_Distance'):
                        w1d = fake_logits.mean() - real_logits.mean()  # Wasserstein-1 Distance
                        training_stats.report('Loss/wasserstein-1_distance', w1d)
                
                loss_Cmain = w1d + self.lambda_GP*gradient_penalty
                training_stats.report('Loss/C', loss_Cmain)
            with torch.autograd.profiler.record_function('Cmain_backward'):
                loss_Cmain.backward()
