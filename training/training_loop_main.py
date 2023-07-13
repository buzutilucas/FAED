# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import time
from lib.util import EasyDict, format_time

from . import training_loop_vq_vae
from . import training_loop_dcgan
from . import training_loop_wgan_gp


# It structures all models
#----------------------------------------------------------------------------
_models_dict = EasyDict()
register = lambda fn: _models_dict.setdefault(fn.__name__, fn)

def is_valid_model(model):
    return model in _models_dict

def list_models():
    return list(_models_dict.keys())

#----------------------------------------------------------------------------
def run(rank, args):
    assert is_valid_model(args.model)
    
    # Calculate time
    start_time = time.time()
    _models_dict[args.model](rank=rank, args=args)
    total_time = time.time() - start_time

    # Decorate with metadata.
    return EasyDict(
        model           = args.model,
        total_time      = total_time,
        total_time_str  = format_time(total_time),
        num_gpus        = args.num_gpus,
    )

# Models used
#----------------------------------------------------------------------------

@register
def vq_vae(rank, args):
    training_loop_vq_vae.run(rank=rank, **args)

@register
def dcgan(rank, args):
    training_loop_dcgan.run(rank=rank, **args)

@register
def wgan_gp(rank, args):
    training_loop_wgan_gp.run(rank=rank, **args)
