# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#
# source: https://github.com/NVlabs/stylegan2-ada-pytorch

from .__typing__ import *

import os
import time
import copy
import uuid
import numpy
import pickle
import hashlib

import torch
import torchvision.transforms as transforms

from . import URL
from . import misc
from . import util
from .ckpt import Checkpoint
from training import database


# Resize for Wasserstein-2 Autoencoder Distance
# ------------------------------------------------------------------------------------------

resize = transforms.Resize((128,128), interpolation=transforms.InterpolationMode.BICUBIC)


# Class to configure of metrics
# ------------------------------------------------------------------------------------------

class ConfigMetrics(object):
    def __init__(
        self,
        model: str='',
        G: Module=None,
        G_kwargs: dict={},
        dataset_kwargs: dict={},
        synthetic_kwargs: dict=None, 
        num_gpus: int=0,
        device: type_device=None, 
        rank: int=0,
        progress: object=None,
        cache: bool=True
    ) -> None:
        assert 0 <= rank < num_gpus
        self.model                  = model
        self.G                      = G
        self.G_kwargs               = G_kwargs
        self.dataset_kwargs         = dataset_kwargs
        self.synthetic_kwargs       = synthetic_kwargs
        self.num_gpus               = num_gpus
        self.device                 = device if device is not None else torch.device('cpu')
        self.rank                   = rank,
        self.progress               = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache                  = cache

# Load model to dectet the features
# ------------------------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(
    url: str, 
    device: torch.device = torch.device('cpu'), 
    num_gpus: int = 0, 
    rank: int = 0, 
    verbose: bool = False,
    is_vqvae: bool = False
) -> dict:
    
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with URL.open_url(url, verbose=(verbose and is_leader)) as f:
            if not is_vqvae:
                _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
            else:
                from training.models import vqvae, config
                _ = vqvae.VQVAE(**config.Config(model='vq_vae')())
                VQVAE = Checkpoint.load_network_pkl(f, ['VQVAE', 'VQVAE_ema'])['VQVAE_ema'].to(device)
                _feature_detector_cache[key] = VQVAE.encoder
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

# Structure to represent a feature class
# ------------------------------------------------------------------------------------------
class FeatureStats(object):
    """
    Structure to represent a feature class
    """
    def __init__(
        self, 
        all: bool = False, 
        mean_cov: bool = False, 
        max_element: int = None
    ) -> None:
        self.all = all
        self.mean_cov = mean_cov
        self.max_element = max_element
        self.num_elements = 0
        self.num_feature = 0
        self.all_data_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_features(self, num_features: int) -> None:
        """ Create the arrays """
        if self.num_feature != 0:
            assert self.num_feature == num_features, "Number of features must be equal."
        else:
            self.num_feature = num_features
            self.all_data_features = []
            self.raw_mean = numpy.zeros([self.num_feature], dtype=numpy.float64)
            self.raw_cov = numpy.zeros([self.num_feature, self.num_feature], dtype=numpy.float64)

    def is_empty(self) -> bool:
        """ Check the feature list is empty """
        return (self.max_element is None) and (self.num_elements <= self.max_element)
    
    def is_full(self):
        """ Check the feature list is full """
        return (self.max_element is not None) and (self.num_elements >= self.max_element)

    def append_(self, x: numpy.ndarray) -> None:
        """ It append the element (numpy array) into list """
        x = numpy.asarray(x, dtype=numpy.float32)
        assert x.ndim == 2, f"The array must be 2-dimensional. {x.ndim} != 2."
        if (self.max_element is not None) and (self.num_elements + x.shape[0] > self.max_element):
            if self.num_elements >= self.max_element:
                return
            x = x[:self.max_element - self.num_elements]

        self.set_features(x.shape[1])
        self.num_elements += x.shape[0]
        if self.all:
            self.all_data_features.append(x)
        if self.mean_cov:
            x64 = x.astype(numpy.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append(self, x: torch.Tensor, num_gpus: int = 0, rank: int = 0) -> None:
        assert isinstance(x, torch.Tensor) and x.ndim == 2, "x must be a Tensor and 2-dimensional."
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)
        self.append_(x.cpu().numpy())

    def get_all_features(self) -> numpy.ndarray:
        assert self.all
        return numpy.concatenate(self.all_data_features, axis=0)
    
    def get_all_features_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.get_all_features())

    def get_mean_cov(self) -> Tuple[numpy.ndarray,numpy.ndarray]:
        assert self.mean_cov
        mean = self.raw_mean / self.num_elements
        cov = self.raw_cov / self.num_elements
        cov = cov - numpy.outer(mean, mean)
        return (mean, cov)

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = util.EasyDict(pickle.load(f))
        obj = FeatureStats(all=s.all, max_element=s.max_element)
        obj.__dict__.update(s)
        return obj

# ------------------------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, 
    tag=None, 
    num_items=None, 
    flush_interval=1000, 
    verbose=False, 
    progress_fn=None, 
    pfn_lo=0, 
    pfn_hi=1000, 
    pfn_total=1000
):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )


# ------------------------------------------------------------------------------------------

def run_features(
    config: ConfigMetrics, 
    detector_url: str=None, 
    detector_kwargs: dict=None, 
    is_vqvae: bool=False, 
    is_synthetic: bool=False, 
    batch_size: int=64, 
    data_loader_kwargs: dict=None, 
    max_element: int=None,
    **features_kwargs
) -> FeatureStats:
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    data = config.dataset_kwargs
    if is_synthetic:
        data = config.synthetic_kwargs
    dataset = database.ImageFolderDataset(**data)

    num_elements = dataset.__len__()
    if max_element is not None:
        num_elements = min(num_elements, max_element)
    stats = FeatureStats(max_element=num_elements, **features_kwargs)
    if detector_url is not None:
        detector = get_feature_detector(
            url=detector_url, device=config.device, num_gpus=config.num_gpus, rank=config.rank, verbose=False, is_vqvae=is_vqvae
        )

    # Main loop
    item_subset = None
    if config.num_gpus > 0:
        item_subset = [(i * config.num_gpus + config.rank) % num_elements for i in range((num_elements - 1) // config.num_gpus + 1)]
    for images in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if not is_vqvae:
            features = detector(images.to(config.device), **detector_kwargs)
        else:
            with torch.no_grad():
                images = resize(images)
                # This will normalize the image in the range [-0.5,0.5].
                images = images.to(torch.float32) / 255.0 - 0.5
                features = detector(images.to(config.device))
                features = features.flatten(start_dim=1)
        stats.append(features, num_gpus=config.num_gpus, rank=config.rank)
    return stats

# ------------------------------------------------------------------------------------------

def compute_feature_stats_for_dataset(
    config: ConfigMetrics,
    detector_url: str,
    detector_kwargs: dict,
    is_vqvae: bool=False, 
    rel_lo: int=0,
    rel_hi: int=1,
    batch_size: int=64,
    data_loader_kwargs: dict=None,
    max_element: int=None,
    **stats_kwargs
) -> FeatureStats:
    dataset = database.ImageFolderDataset(**config.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if config.cache:
        # Choose cache file name.
        args = dict(
            dataset_kwargs=config.dataset_kwargs, detector_url=detector_url, 
            detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs
        )
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = misc.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if config.rank == 0 else False
        if config.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=config.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_element is not None:
        num_items = min(num_items, max_element)
    stats = FeatureStats(max_element=num_items, **stats_kwargs)
    progress = config.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(
        is_vqvae=is_vqvae, url=detector_url, device=config.device, num_gpus=config.num_gpus, rank=config.rank, verbose=progress.verbose
    )

    # Main loop.
    item_subset = [(i * config.num_gpus + config.rank) % num_items for i in range((num_items - 1) // config.num_gpus + 1)]
    for images in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if not is_vqvae:
            features = detector(images.to(config.device), **detector_kwargs)
        else:
            with torch.no_grad():
                images = resize(images)
                # This will normalize the image in the range [-0.5,0.5].
                images = images.to(torch.float32) / 255.0 - 0.5
                features = detector(images.to(config.device))
                features = features.flatten(start_dim=1)
        stats.append(features, num_gpus=config.num_gpus, rank=config.rank)
        progress.update(stats.num_elements)

    # Save to cache.
    if cache_file is not None and config.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

# ------------------------------------------------------------------------------------------

def compute_feature_stats_for_generator(
    config: ConfigMetrics,
    detector_url: str,
    detector_kwargs: dict,
    is_vqvae: bool=False,
    rel_lo: int=0,
    rel_hi: int=1,
    batch_size: int=64,
    batch_gen: int=None,
    jit: bool=False,
    **stats_kwargs
) -> FeatureStats:
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator.
    G = copy.deepcopy(config.G).eval().requires_grad_(False).to(config.device)

    # Image generation func.
    def run_generator(z):
        img = G(z=z)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim, 1, 1], device=config.device)
        run_generator = torch.jit.trace(run_generator, [z], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_element is not None
    progress = config.progress.sub(tag='generator features', num_items=stats.max_element, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(
        is_vqvae=is_vqvae, url=detector_url, device=config.device, num_gpus=config.num_gpus, rank=config.rank, verbose=progress.verbose
    )

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim, 1, 1], device=config.device)
            img = run_generator(z)
            if is_vqvae:
                img = resize(img)
                # This will normalize the image in the range [-0.5,0.5]
                img = img.to(torch.float32) / 255.0 - 0.5
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if not is_vqvae:
            features = detector(images, **detector_kwargs)
        else:
            with torch.no_grad():
                features = detector(images)
                features = features.flatten(start_dim=1)
        stats.append(features, num_gpus=config.num_gpus, rank=config.rank)
        progress.update(stats.num_elements)
    return stats