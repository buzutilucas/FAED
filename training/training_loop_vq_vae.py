from lib.__typing__ import *

import os
import json
import time
import copy
import torch
import numpy as np

from lib.logger import Logger
from lib.ckpt import Checkpoint
from lib import misc
from lib import util
from lib import URL
from lib import training_stats

from .models import vqvae
from .models.config import Config
from .loss.vqvae import Loss

from metrics import compute
from . import database


#----------------------------------------------------------------------------

def run(
    run_dir                 = '.',       # Output directory.            
    training_set_kwargs     = {},        # Options for training set.
    validation_set_kwargs   = {},        # Options for validation set.
    data_loader_kwargs      = {},        # Options for torch.utils.data.DataLoader.      
    model                   = 'vq_vae',  # Model to be trained.
    random_seed             = 0,         # Global random seed.
    num_gpus                = 1,         # Number of GPUs participating in the training.
    rank                    = 0,         # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,         # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,         # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,        # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,      # EMA ramp-up coefficient.
    total_kimg              = 25000,     # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,         # Progress snapshot interval.
    image_snapshot_ticks    = 50,        # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,        # How often to save network snapshots? None = disable.
    resume_pkl              = None,      # Network pickle to resume training from.
    cudnn_benchmark         = True,      # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,     # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,      # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,      # Callback function for updating training progress. Called for all ranks.
    tensorboard             = False,     # Print training stats using tensorboard.
):
    # Initialize.
    if rank == 0:
        print('Initialize...')
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions

    # Logger
    logger = Logger(rank, log_dir=f'{run_dir}/logs', plot_dir=f'{run_dir}/plots', img_dir=f'{run_dir}/imgs')
    ckpt = Checkpoint(
        rank, num_gpus, ckpt_dir=f'{run_dir}/ckpt', 
        training_set_kwargs=training_set_kwargs, 
        validation_set_kwargs=validation_set_kwargs
    )

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = database.ImageFolderDataset(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    validation_set = database.ImageFolderDataset(**validation_set_kwargs)
    validation_set_sampler = misc.InfiniteSampler(dataset=validation_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    validation_set_iterator = iter(torch.utils.data.DataLoader(dataset=validation_set, sampler=validation_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print(f'Num images of training set:       {len(training_set)}')
        print(f'Num images of validation set:     {len(validation_set)}')
        print(f'Image shape:                      {training_set.image_shape}')
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    VQVAE = vqvae.VQVAE(**Config(model=model)()).train().requires_grad_(False).to(device)
    VQVAE_ema = copy.deepcopy(VQVAE).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with URL.open_url(resume_pkl) as f:
            resume_data = ckpt.load_network_pkl(f, ['VQVAE', 'VQVAE_ema'])
        for name, module in [('VQVAE', VQVAE), ('VQVAE_ema', VQVAE_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        C, H, W = training_set.image_shape
        img = torch.empty([batch_gpu, C, H, W], device=device)
        misc.print_module_summary(VQVAE, [img])

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('VQVAE', VQVAE), (None, VQVAE_ema)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[device], broadcast_buffers=False
            )
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = Loss(device=device, **ddp_modules)
    opt = torch.optim.Adam(VQVAE.parameters(), lr=2e-4)
    arch = util.EasyDict(name='VQVAEmain', module=VQVAE, opt=opt)
    arch.start_event = None
    arch.end_event = None
    if rank == 0:
        arch.start_event = torch.cuda.Event(enable_timing=True)
        arch.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_images = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images = logger.setup_snapshot_image_grid(training_set=validation_set)
        logger.save_image_grid(images, drange=[0,255], grid_size=grid_size)
        grid_images = (torch.from_numpy(images).to(device).to(torch.float32) / 255.0 - 0.5).split(batch_gpu) # This will normalize the image in the range [-0.5,0.5].
        images = torch.cat([VQVAE_ema(x=img)[0].cpu() for img in grid_images]).numpy()
        logger.save_image_grid(images, fname='recon_init.png', drange=[-0.5,0.5], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    monitor_jsonl = None
    stats_tfevents = None
    if rank == 0:
        monitor_jsonl = open(os.path.join(logger.log_dir, 'training_monitor.jsonl'), 'wt')
        if tensorboard:
            try:
                import torch.utils.tensorboard as tensorboard_
                stats_tfevents = tensorboard_.SummaryWriter(run_dir)
            except ImportError as err:
                print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    it = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training and validation data
        phases = []
        for name, set_iterator in [('train', training_set_iterator), ('val', validation_set_iterator)]:
            with torch.autograd.profiler.record_function(name):
                phase_img = next(set_iterator)
                phase_img = (phase_img.to(device).to(torch.float32) / 255.0 - 0.5).split(batch_gpu) # This will normalize the image in the range [-0.5,0.5].
            phases += [util.EasyDict(name=name, phase_img=phase_img)]

        # Execute training and validation phase.
        for phase in phases:
            # Initialize gradient accumulation.
                if arch.start_event is not None:
                    arch.start_event.record(torch.cuda.current_stream(device))
                if phase.name == 'train':
                    arch.opt.zero_grad(set_to_none=True)
                    arch.module.requires_grad_(True)

                # Accumulate gradients over multiple rounds.
                for round_idx, img in enumerate(phase.phase_img):
                    sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                    loss.accumulate_gradients(img=img, sync=sync, phase=phase.name)

                if phase.name == 'train':
                    # Update weights.
                    arch.module.requires_grad_(False)
                    with torch.autograd.profiler.record_function(arch.name + '_opt'):
                        for param in arch.module.parameters():
                            if param.grad is not None:
                                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                        arch.opt.step()
                if arch.end_event is not None:
                    arch.end_event.record(torch.cuda.current_stream(device))

        # Update VQVAE_ema.
        with torch.autograd.profiler.record_function('VQVAE_ema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(VQVAE_ema.parameters(), VQVAE.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(VQVAE_ema.buffers(), VQVAE.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([VQVAE_ema(x=img)[0].cpu() for img in grid_images]).numpy()
            logger.save_image_grid(images, fname=f'recon{cur_nimg//1000:06d}.png', drange=[-0.5,0.5], grid_size=grid_size)

        # Save network snapshot.
        snapshot_data = ckpt.snapshot_data
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            ckpt.save_network_pkl(
                models=[('VQVAE', VQVAE), ('VQVAE_ema', VQVAE_ema)], 
                fname=f'network-snapshot-{cur_nimg//1000:06d}.pkl'
            )
        del snapshot_data # conserve memory

        # Collect statistics.
        value = None
        if (arch.start_event is not None) and (arch.end_event is not None):
            arch.end_event.synchronize()
            value = arch.start_event.elapsed_time(arch.end_event)
        training_stats.report0('Timing/' + arch.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if monitor_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            monitor_jsonl.write(json.dumps(fields) + '\n')
            monitor_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if not tensorboard:
            global_step = int(cur_nimg / 1e3)
            for name, value in stats_dict.items():
                k = name[name.rfind('/')+1:]
                category = name[:name.find('/')]
                logger.add(category=category, k=k, v=value.mean, it=global_step)
            for c1, c2, k_ in [
                ('train', 'val', 'recon'), ('train', 'val', 'perplexity'), 
                ('train', 'val', 'vq'), ('train', 'val', 'recon+vq')
            ]:
                logger.add_plot(category1=c1, category2=c2, k=k_)
            logger.save_stats('stats.pkl')
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)
        
        monitor_tr = ['[TRAIN] Loss']
        monitor_val = ['[VAL] Loss']
        for name, value in stats_dict.items():
            k = name[name.rfind('/')+1:]
            category = name[:name.find('/')]
            if category == 'train':
                monitor_tr += [f"{k} {value.mean:.2f}"]
            if category == 'val':
                monitor_val += [f"{k} {value.mean:.2f}"]
        if rank == 0:
            print(' '.join(monitor_tr))
            print(' '.join(monitor_val))

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break
        it += 1

    # Done.
    if rank == 0:
        print()
        print('Exiting...')
