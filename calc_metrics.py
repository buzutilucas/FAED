# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import os
import json
import click
import torch
import tempfile
from lib.util import UserError, EasyDict, CommaSeparatedList
from metrics.compute import run_metric

#---------------------------------------------------------------------------------------------
def setup(
    data          = None, # Real dataset (required): <path>
    synthetic     = None, # synthetic dataset (required): <path>
    gpus          = None, # Number of GPUs or CPU mode: <int>, default = 0 cpu or [1,2,3 or n] gpus
    metrics       = None, # Metrics
    verbose       = None  # Print information
):
    args = EasyDict()

    if data is None:
        raise UserError("--data must be required.")
    assert isinstance(data, str)
    args.data = data

    args.synthetic = synthetic
    if synthetic is not None:
        assert isinstance(synthetic, str)
        args.synthetic = synthetic

    if gpus is None:
        gpus = 0
    assert isinstance(gpus, int)
    if gpus < 0:
        raise UserError("--gpus must be 0 to access CPU or 1,2,3,... to access GPUs.")
    args.num_gpus = gpus

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    args.metrics = metrics

    if verbose is None:
        verbose = False
    assert isinstance(verbose, bool)
    args.verbose = verbose

    return args

#---------------------------------------------------------------------------------------------
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

    # Init device
    device = torch.device('cpu')
    if args.num_gpus > 0:
        device = torch.device('cuda', rank)
        torch.backends.cudnn.benchmark = True  # find the best algorithm to use for your hardware
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    

    config_kwargs = EasyDict(
        dataset_kwargs=EasyDict(path=args.data), 
        synthetic_kwargs=EasyDict(path=args.synthetic), 
        num_gpus=args.num_gpus, 
        rank=rank, 
        device=device
    )

    # Calculate each metric.
    results = []
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print('-----------------------------------')
            print(f'Calculating {metric}...')
        result = run_metric(metric, **config_kwargs)
        if rank == 0 and args.verbose:
            print(f'Results: {result.results}')
            print(f'Computer Time: {result.total_time_str}')
            print(f'Number of GPUs used: {result.num_gpus}')
        results.append(result)

    if rank == 0:
        print('\nSaving results.json...')
        with open('results.json', 'wt') as f:
            json.dump(results, f, indent=2)

    # Done.
    if rank == 0 and args.verbose:
        print('\nDone.')

#---------------------------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--data', help='Real data (directory)', metavar='PATH', required=True)
@click.option('--synthetic', help='Generate data (directory)', metavar='PATH')
@click.option('--gpus', help='Number of GPUs [1,2,3,...,N] or CPU [0] mode to use [default: 0]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: faed50k_full]', type=CommaSeparatedList())
@click.option('--verbose', help='Print optional information', type=bool, metavar='BOOL')

def calc_metrics(ctx, **setup_kwargs):
    """
    Examples:

        # Calculate metrics of the real and synthetic images from directory them. 
        python calc_metrics.py --data=/Path/to/data --synthetic=/Path/to/synthetic/data --gpus=1

        python calc_metrics.py --data=/Path/to/data --synthetic=/Path/to/synthetic/data --gpus=2 \\
            --metrics=faed50k_full,fid50k_full --verbose=True
    """
    # data, gpus, rank, device

    # Setup configuration of training options.
    try:
        args = setup(**setup_kwargs)
    except UserError as err:
        print(err)
        ctx.fail(err)

    # Launch processes.
    if args.verbose:
        print('\nLaunching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus <= 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args,temp_dir), nprocs=args.num_gpus)

#---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    calc_metrics()
    
