"""
Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py
source: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/frechet_inception_distance.py
"""

from lib.__typing__ import *

import numpy
import scipy.linalg
from lib import metrics_util


def compute(config: metrics_util.ConfigMetrics, max_real: int , num_gen: int) -> float:
    if isinstance(config.rank, Tuple):
        config.rank = config.rank[0]

    detector_url = 'https://drive.google.com/u/0/uc?export=download&confirm=S1Nz&id=1ai1C7hWKdNlCjgRi5HQDCnimvK4HXn9C'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    if config.synthetic_kwargs is not None:
        mu_real, sigma_real = metrics_util.run_features(
            config=config, detector_url=detector_url, is_synthetic=False,
            detector_kwargs=detector_kwargs, mean_cov=True, max_element=max_real
        ).get_mean_cov()
        mu_gen, sigma_gen = metrics_util.run_features(
            config=config, detector_url=detector_url, is_synthetic=True,
            detector_kwargs=detector_kwargs, mean_cov=True, max_element=num_gen
        ).get_mean_cov()
    else:
        mu_real, sigma_real = metrics_util.compute_feature_stats_for_dataset(
            config=config, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, mean_cov=True, max_element=max_real
        ).get_mean_cov()

        mu_gen, sigma_gen = metrics_util.compute_feature_stats_for_generator(
            config=config, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, mean_cov=True, max_element=num_gen
        ).get_mean_cov()

    if config.rank != 0:
        return float('nan')

    m = numpy.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(numpy.dot(sigma_gen, sigma_real), disp=False)
    fid = numpy.real(m + numpy.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

