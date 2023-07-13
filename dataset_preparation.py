# Copyright (c) 2022, The Images Processing Laboratory Authors. All Rights Reserved.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import os
import glob
import click
import numpy
import PIL.Image
from tqdm import tqdm

from lib.util import UserError, EasyDict


class TqdmExtraFormat(tqdm):
    """
    Provides a `total_time` format parameter
    """
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d['elapsed'] * (d['total'] or 0) / max(d['n'], 1)
        d.update(total_time=self.format_interval(total_time) + ' in total')
        return d

#------------------------------------------------------------------------------------------

def setup(
    # General options (not included in desc).
    path            = None, # Path of dataset to be prepared (required): <path>
    resize          = None, # Value to resize the data (required): <int>
    outdir          = None, # Where to save the results [defalt: original path of dataset]: <path>
    dataset         = None, # Dataset to be prepared [defalt: celeba]: <str>
):
    args = EasyDict()

    if path is None:
        raise UserError("--data must be required.")
    assert isinstance(path, str)
    args.path = path

    if resize is None:
        raise UserError("--resize must be required.")
    assert isinstance(resize, int)
    args.resize = resize

    if dataset == 'imagenet' and outdir is None:
        outdir = './ImageNet'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        if outdir is None:
            outdir = path
    assert isinstance(outdir, str)
    args.outdir = outdir   

    if dataset is None:
        dataset = 'celeba'
    assert isinstance(dataset, str)
    args.dataset = dataset

    return args

#------------------------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--path', help='Path of dataset (directory)', metavar='PATH', required=True)
@click.option('--resize', help='Value to resize the data', type=int, metavar='INT', required=True)
@click.option('--outdir', help='Where the dataset prepared will be saved', metavar='DIR')
@click.option('--dataset', help='Dataset to be prepared [defalt: celeba]', type=str)

def main(ctx, **setup_kwargs):
    """
    Examples:

    \b
    # ImageNet Dataset
    # for training VQ-VAE
    python dataset_preparation.py --path=~/dataset/ImageNet --resize=128 --dataset=imagenet \\
        --outdir=~/where/dataset/will/be/saved

    \b
    # CelebA Dataset
    python dataset_preparation.py --path=~/dataset/CelebA --resize=128 --dataset=celeba

    \b
    # Flickr Faces HQ Dataset
    python dataset_preparation.py --path=~/dataset/Flickr --resize=128 --dataset=ffhq

    \b
    # Animal Faces-HQ Dataset
    python dataset_preparation.py --path=~/dataset/AFHQ --resize=128 --dataset=afhq

    """
    
    # Setup configuration of training options.
    try:
        args = setup(**setup_kwargs)
    except UserError as err:
        print(err)
        ctx.fail(err)

    if args.dataset == 'imagenet':
        """ https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a """

        files_tr = glob.glob(f'{args.path}/train/'+r'**/*.[pjJ][npP][eEg]*')
        files_val = glob.glob(f'{args.path}/val/'+r'**/*.[pjJ][npP][eEg]*')

        outdir_tr = f'{args.outdir}/train'
        outdir_val = f'{args.outdir}/val'
        if not os.path.exists(outdir_tr):
            os.makedirs(outdir_tr)
        if not os.path.exists(outdir_val):
            os.makedirs(outdir_val)

        print('ImageNet training dataset preparation...')
        for file in TqdmExtraFormat(
            files_tr, total=len(files_tr), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]

            img = PIL.Image.open(file)
            if img.mode != 'RGB':
                img = img.convert(mode='RGB')
            img = img.resize((args.resize, args.resize))
            img.save(f'{outdir_tr}/{name_file}')

        print('ImageNet validation dataset preparation...')
        for file in TqdmExtraFormat(
            files_val, total=len(files_val), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]

            img = PIL.Image.open(file)
            if img.mode != 'RGB':
                img = img.convert(mode='RGB')
            img = img.resize((args.resize, args.resize))
            img.save(f'{outdir_val}/{name_file}')

    elif args.dataset == 'celeba':
        """ https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html """

        mask = f'{args.path}/'+r'*.[pj][np][eg]*'
        files = glob.glob(mask)

        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[offset_height:offset_height + crop_size, offset_width:offset_width + crop_size, :]

        print('CelebA dataset preparation...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]

            img = numpy.asarray(PIL.Image.open(file))
            img = crop(img)
            img = PIL.Image.fromarray(img).resize((args.resize, args.resize))
            img.save(f'{args.outdir}/{name_file}')

    elif args.dataset == 'ffhq':
        """ https://github.com/NVlabs/ffhq-dataset """

        mask = f'{args.path}/'+r'*.[pj][np][eg]*'
        files = glob.glob(mask)
        
        print('Flickr Face HQ dataset preparation...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]

            img = PIL.Image.open(file).resize((args.resize, args.resize))
            img.save(f'{args.outdir}/{name_file}')

    elif args.dataset == 'afhq':
        """ 
        Animal Faces-HQ dataset (AFHQ)
        https://github.com/clovaai/stargan-v2 
        """

        mask = f'{args.path}/'+r'**/*.[pj][np][eg]*'
        files = glob.glob(mask, recursive=True)

        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]

            img = PIL.Image.open(file)
            if img.mode != 'RGB':
                img = img.convert(mode='RGB')
            img = img.resize((args.resize, args.resize))
            img.save(f'{args.outdir}/{name_file}')

    else:
        raise UserError("--dataset must be 'imagenet', 'celeba', 'ffhq' or 'afhq'.")


#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()