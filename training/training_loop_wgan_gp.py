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

from .models import wgan_gp
from .models.config import Config
from .loss.wgan_gp import Loss

from metrics import compute
from . import database



#----------------------------------------------------------------------------

def run(
    run_dir                 = '.',          # Output directory.            
    training_set_kwargs     = {},           # Options for training set.
    data_loader_kwargs      = {},           # Options for torch.utils.data.DataLoader.      
    metrics                 = [],           # Metrics to evaluate during training.
    model                   = 'wgan_gp',    # Model to be trained.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    tensorboard             = False,    # Print training stats using tensorboard.
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
    ckpt = Checkpoint(rank, num_gpus, ckpt_dir=f'{run_dir}/ckpt', training_set_kwargs=training_set_kwargs)

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = database.ImageFolderDataset(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    G = wgan_gp.Generator(**Config(model=model)(isgen=True)).train().requires_grad_(False).to(device)
    C = wgan_gp.Critic(**Config(model=model)(isgen=False)).train().requires_grad_(False).to(device)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim, 1, 1], device=device)
        img = misc.print_module_summary(G, [z])
        misc.print_module_summary(C, [img])

    # Sync Batch Normalization on DDP
    if num_gpus > 1:
        G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with URL.open_url(resume_pkl) as f:
            resume_data = ckpt.load_network_pkl(f, ['G', 'C', 'G_ema'])
        for name, module in [('G', G), ('C', C), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G', G), ('C', C), (None, G_ema)]:
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
    phases = {}
    lr = 5e-5
    for name, module in [('G', G), ('C', C)]:
        opt = torch.optim.RMSprop(module.parameters(), lr=lr)
        phases[name] = util.EasyDict(name=name+'main', module=module, opt=opt)
    for name, phase in phases.items():
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images = logger.setup_snapshot_image_grid(training_set=training_set)
        logger.save_image_grid(images, drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([images.shape[0], G.z_dim, 1, 1], device=device).split(batch_gpu)
        images = torch.cat([G_ema(z=z).cpu() for z in grid_z]).numpy()
        logger.save_image_grid(images, fname='fakes_init.png', drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
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

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim, 1, 1], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

        # Execute training phases.
        for name, phase_gen_z in zip(['C', 'G'], all_gen_z):
            phase = phases[name]
            nc = 5 if name == 'C' else 1
            for _ in range(nc):
                # Initialize gradient accumulation.
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

                # Accumulate gradients over multiple rounds.
                for round_idx, (real_img, gen_z) in enumerate(zip(phase_real_img, phase_gen_z)):
                    sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, gen_z=gen_z, sync=sync)

                # Update weights.
                phase.module.requires_grad_(False)
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    for param in phase.module.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    phase.opt.step()
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('G_ema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
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
            images = torch.cat([G_ema(z=z).cpu() for z in grid_z]).numpy()
            logger.save_image_grid(images, fname=f'fakes{cur_nimg//1000:06d}.png', drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = ckpt.snapshot_pkl
        snapshot_data = ckpt.snapshot_data
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            ckpt.save_network_pkl(
                models=[('G', G), ('C', C), ('G_ema', G_ema)], 
                fname=f'network-snapshot-{cur_nimg//1000:06d}.pkl'
            )
        
        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = compute.run_metric(metric=metric, model=model, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    compute.report_metric(result_dict, run_dir=logger.log_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for name, phase in phases.items():
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
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
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if not tensorboard:
            global_step = int(cur_nimg / 1e3)
            for name, value in stats_dict.items():
                category = name[name.rfind('/')+1:]
                k = name[:name.find('/')]
                logger.add(category=category, k=k, v=value.mean, it=global_step)
            logger.add_plot(category1='C', category2='G', k='Loss')
            logger.add_plot(category1='gradient_penalty', k='Loss')
            logger.add_plot(category1='wasserstein-1_distance', k='Loss')
            for name, value in stats_metrics.items():
                logger.add(category=name, k='Scores', v=value, it=global_step)
                logger.add_plot(category1=name, k='Scores')
            logger.save_stats('stats.pkl')
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)
        
        # learning rate decay
        if int(cur_nimg / 1e3) >= 100000 and cur_tick % round((total_kimg / kimg_per_tick) * 0.1) == 0:
            lr -= (lr / float(100000))
            for name in ['C', 'G']:
                phase = phases[name]
                for param in phase.opt.param_groups:
                    param['lr'] = lr

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
