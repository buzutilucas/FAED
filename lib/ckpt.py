# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

from lib.__typing__ import *

import os
import copy
import torch
import pickle
from abc import abstractmethod

from . import misc
from . import util


class Checkpoint(object):
    def __init__(self, 
        rank: int, 
        num_gpus: int, 
        ckpt_dir: str='./ckpt', 
        training_set_kwargs: util.EasyDict=None,
        validation_set_kwargs: util.EasyDict=None
    ) -> None:
        self.rank = rank
        self.num_gpus = num_gpus
        self.ckpt_dir = ckpt_dir

        if rank == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

        self.snapshot_pkl = None
        self.snapshot_data = util.EasyDict(
            training_set_kwargs=training_set_kwargs,
            validation_set_kwargs=validation_set_kwargs
        )

    def save_network_pkl(self, models: List[Tuple[str, Module]], fname: str):
        for name, module in models:
            if module is not None:
                if self.num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            self.snapshot_data[name] = module
            del module # conserve memory
        self.snapshot_pkl = os.path.join(self.ckpt_dir, fname)
        if self.rank == 0:
            with open(self.snapshot_pkl, 'wb') as f:
                pickle.dump(self.snapshot_data, f)

    @abstractmethod
    def load_network_pkl(f: str, models: List[str], force_fp16: bool=False):
        data = _LegacyUnpickler(f).load() #pickle.load(f)

        # Add missing fields.
        if 'training_set_kwargs' not in data:
            data['training_set_kwargs'] = None
            data['validation_set_kwargs'] = None

        # Validate contents.
        for model in models:
            assert isinstance(data[model], torch.nn.Module)
        assert isinstance(data['training_set_kwargs'], (dict, type(None)))
        assert isinstance(data['validation_set_kwargs'], (dict, type(None)))

        # Force FP16.
        if force_fp16:
            for key in models:
                old = data[key]
                kwargs = copy.deepcopy(old.init_kwargs)

                if kwargs != old.init_kwargs:
                    new = type(old)(**kwargs).eval().requires_grad_(False)
                    misc.copy_params_and_buffers(old, new, require_all=True)
                    data[key] = new
        return data


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)