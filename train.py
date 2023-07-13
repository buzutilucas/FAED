# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import os
import click
import re
import json
import tempfile
import torch

from lib import training_stats
from lib.util import UserError, EasyDict, CommaSeparatedList

from metrics import compute
from training import database, training_loop_main


def setup_training(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    model      = None, # Model to be trained: <str> default = dcgan
    metrics    = None, # List of metric names: [], ['faed50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0
    
    # Dataset.
    data       = None, # Training dataset (required): <path>
    subset     = None, # Train with only N images: <int>, default = all

    # Base config.
    kimg       = None, # Override training duration default = 25000: <int>
    batch      = None, # Override batch size: <int>

    # Load model
    resume     = None, # Load previous network: 'noresume' (default), <file>

    # Performance options (not included in desc).
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    assert model is None or isinstance(model, str)
    if model is None:
        model = 'dcgan'
    desc = model
    args.model = model

    if not model == 'vq_vae':
        if metrics is None:
            metrics = ['faed50k_full']
        assert isinstance(metrics, list)
        if not all(compute.is_valid_metric(metric) for metric in metrics):
            raise UserError('\n'.join(['--metrics can only contain the following values:'] + compute.list_valid_metrics()))
        args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, subset
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = EasyDict(path=data, max_size=None)
    if model == 'vq_vae':
        args.training_set_kwargs = EasyDict(path=f'{data}/train', max_size=None)
        args.validation_set_kwargs = EasyDict(path=f'{data}/val', max_size=None)
    args.data_loader_kwargs = EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = database.ImageFolderDataset(**args.training_set_kwargs)
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        name_data = training_set.name
        del training_set # conserve memory

        if model == 'vq_vae':
            validation_set = database.ImageFolderDataset(**args.validation_set_kwargs)
            args.validation_set_kwargs.resolution = validation_set.resolution # be explicit about resolution
            args.validation_set_kwargs.max_size = len(validation_set) # be explicit about dataset size
            name_data = data[data.rfind('/')+1:]
            del validation_set # conserve memory
        
        desc += f'-{name_data}'
    except IOError as err:
        raise UserError(f'--data: {err}')

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed
    
    # ------------------------------------
    # Base config: kimg, batch
    # ------------------------------------

    res = args.training_set_kwargs.resolution
    mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
    ema = mb * 10 / 32

    args.total_kimg = 25000
    args.batch_size = mb
    args.batch_gpu = mb // gpus
    args.ema_kimg = ema
    args.ema_rampup = None

    args.total_kimg = 25000
    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ----------------------------------
    # Load model: resume
    # ----------------------------------

    assert resume is None or isinstance(resume, str)
    if resume is None:
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path

    # -------------------------------------------------
    # Performance options: nobench, allow tf32, workers
    # -------------------------------------------------

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
    
    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    # Execute training loop.
    training_loop_main.run(rank=rank, args=args)

#----------------------------------------------------------------------------
@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--model', help='Model selector [default: dcgan]', type=str)
@click.option('--metrics', help='Comma-separated list or "none" [default: faed50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-tb', '--tensorboard', help='Print training stats using tensorboard', is_flag=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')

# Base config.
@click.option('--kimg', help='Override training duration [default: 25000]', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Load model
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')

# Performance options.
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, tensorboard, dry_run, **config_kwargs):
    """
    Train GANs and VQ-VAE described in the paper
    "Fréchet AutoEncoder Distance: A new approach for evaluation of Generative Adversarial Networks"
    
    Examples:

    \b
    # Train DCGAN with CelebA on resolution 128x128 dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/datasets/CelebA --gpus=1 \\
        --metrics=faed50k_full
    
    \b
    # Train WGAN-GP with Flickr dataset on resolution 128x128 using 2 GPUs and 
    # FAED and FID like evaluation metrics.
    python train.py --outdir=~/training-runs --data=~/datasets/Flickr \\
        --gpus=2 --model=wgan_gp --metrics=faed50k_full,fid50k_full

    \b
    # Train VQ-VAE with ImageNet dataset on resolution 128x128 using 2 GPU.
    python train.py --outdir=~/training-runs --data=~/datasets/ImageNet --gpus=2 \\
        --batch=128 --kimg=1300 --snap=10 --model=vq_vae

    """

    # Setup configuration of training options.
    try:
        run_desc, args = setup_training(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    args.tensorboard = tensorboard
    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    if args.model == 'vq_vae':
        print(f'Validation data:    {args.validation_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Training model:     {args.model}')
    print(f'Number of GPUs:     {args.num_gpus}')
    if args.model == 'vq_vae':
        print(f'Number of images:   {args.training_set_kwargs.max_size + args.validation_set_kwargs.max_size}')
    else:
        print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Using tensorboard:  {args.tensorboard}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

if __name__ == "__main__":
    main()