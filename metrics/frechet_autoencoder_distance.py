# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

"""
Frechet AutoEncoder Distance (FEAD) from the paper
"Fréchet AutoEncoder Distance: A new approach for evaluation of Generative Adversarial Networks"
"""

from lib.__typing__ import *

import numpy
import scipy.linalg
from lib import metrics_util
import torchvision.transforms as transforms


transform = transforms.Compose([
    # Convert a PIL Image or numpy.ndarray [0,255] to tensor [0,1].
    transforms.ToTensor(),
    transforms.Resize((128,128), interpolation=transforms.InterpolationMode.BICUBIC),
    # Normalize does the following for each channel:
    # image = (image - mean) / std 
    # The parameters mean, std are passed as .5, 1. in this case. 
    # This will normalize the image in the range [-0.5,0.5].
    transforms.Normalize(mean=[.5, .5, .5], std=[1., 1., 1.]),
])

def compute(config: metrics_util.ConfigMetrics, max_real: int , num_gen: int) -> float:
    if isinstance(config.rank, Tuple):
        config.rank = config.rank[0]

    detector_url = 'https://drive.google.com/uc?id=1J2kh7QHoec1EdNfo3CKkt1QeRN1BsXm6&export=download'

    if config.synthetic_kwargs is not None:
        config.dataset_kwargs.transform = transform
        config.synthetic_kwargs.transform = transform
        
        mu_real, sigma_real = metrics_util.run_features(
            config=config, detector_url=detector_url, is_synthetic=False,
            is_vqvae=True, mean_cov=True, max_element=max_real
        ).get_mean_cov()
        mu_gen, sigma_gen = metrics_util.run_features(
            config=config, detector_url=detector_url, is_synthetic=True,
            is_vqvae=True, mean_cov=True, max_element=num_gen
        ).get_mean_cov()
    else:
        mu_real, sigma_real = metrics_util.compute_feature_stats_for_dataset(
            config=config, detector_url=detector_url, detector_kwargs=None,
            is_vqvae=True, rel_lo=0, rel_hi=0, mean_cov=True, max_element=max_real
        ).get_mean_cov()

        mu_gen, sigma_gen = metrics_util.compute_feature_stats_for_generator(
            config=config, detector_url=detector_url, detector_kwargs=None,
            is_vqvae=True, rel_lo=0, rel_hi=1, mean_cov=True, max_element=num_gen
        ).get_mean_cov()

    if config.rank != 0:
        return float('nan')

    m = numpy.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(numpy.dot(sigma_gen, sigma_real), disp=False)
    faed = numpy.real(m + numpy.trace(sigma_gen + sigma_real - s * 2))
    return float(faed)

