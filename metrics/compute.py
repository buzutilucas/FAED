import os
import time
import json
import torch
from lib.util import EasyDict, format_time
from lib.metrics_util import ConfigMetrics

from . import frechet_inception_distance
from . import frechet_autoencoder_distance


# It structures all metrics
#----------------------------------------------------------------------------
_metric_dict = EasyDict()
register = lambda fn: _metric_dict.setdefault(fn.__name__, fn)

def is_valid_metric(metric):
    return metric in _metric_dict

def list_metrics():
    return list(_metric_dict.keys())

#
#----------------------------------------------------------------------------
def run_metric(metric, **kwargs):
    assert is_valid_metric(metric)
    config = ConfigMetrics(**kwargs)
    
    # Calculate time
    start_time = time.time()
    results = _metric_dict[metric](config)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if config.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=config.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return EasyDict(
        results         = EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = format_time(total_time),
        num_gpus        = config.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

# Metrics used
#----------------------------------------------------------------------------

@register
def fid50k_full(config):
    fid = frechet_inception_distance.compute(config, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register
def faed50k_full(config):
    faed = frechet_autoencoder_distance.compute(config, max_real=None, num_gen=50000)
    return dict(faed50k_full=faed)