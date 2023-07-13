# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import os
import glob
import click
import numpy
import random
from tqdm import tqdm

import PIL.Image
from PIL import ImageFilter
from skimage.transform import swirl

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
    path             = None, # Path of dataset to be prepared (required): <path>
    outdir           = None, # Where to save the results [defalt: original path of dataset]: <path>
    disturb          = None, # Disturb to be applied [defalt: gnoise]: <str>
    choices50k       = None  # Choices of 50,000 images on dataset.
):
    args = EasyDict()

    if path is None:
        raise UserError("--data must be required.")
    assert isinstance(path, str)
    args.path = path

    if outdir is None:
        raise UserError("--outdir must be required.")
    assert isinstance(outdir, str)
    args.outdir = outdir

    if disturb is None:
        disturb = 'gnoise'
    assert isinstance(disturb, str)
    args.disturb = disturb

    if choices50k is None:
        choices50k = False
    assert isinstance(choices50k, bool)
    args.choices50k = choices50k

    return args

#------------------------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--path', help='Path of dataset to be prepared (directory)', metavar='PATH', required=True)
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--pathc', help='Path of contaminant dataset', metavar='DIR')
@click.option('--disturb', help='Disturb to be applied [defalt: gnoise]', type=str)
@click.option('--choices50k', help='Choices of 50,000 images on dataset', type=bool, metavar='BOOL')

def main(ctx, pathc, **setup_kwargs):
    """
    Examples:

    \b
    # Choose from 50,000 images in the Flickr Faces HQ Dataset 256 x 256
    python dataset_preparation.py --outdir=~/datasets/with/50k/images --path=~/datasets/Flickr --choices50k=True

    \b
    # Flickr Faces HQ Dataset 256 x 256
    # Disturb types: 'gnoise', 'gblur', 'brectangles', 'swirl', 
    # 'spnoise', 'shelter' or 'exchange'
    python dataset_preparation.py --outdir=~/disturbance/dataset/ --path=~/datasets/Flickr --disturb=gnoise

    \b
    # Flickr Faces HQ Dataset 256 x 256 and ImageNet
    # Disturb types: 'imagenet-contamination'
    python dataset_preparation.py --outdir=~/disturbance/dataset/ --path=~/datasets/Flickr \\
        --pathc=~/dataset/ImageNet --disturb=imagenet-contamination
    """
    
    # Setup configuration of training options.
    try:
        args = setup(**setup_kwargs)
    except UserError as err:
        print(err)
        ctx.fail(err)

    mask = f'{args.path}/'+r'*.[pj][np][eg]*'
    files = glob.glob(mask)

    if args.choices50k:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        print('Choosing 50,000 images...')
        files = files[:50000]
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            PIL.Image.open(file).save(f'{args.outdir}/{name_file}')
        
        return
    
    if args.disturb == 'gnoise':
        """ 
        Gaussian noise
        The noisy image is computed as (1 - w)X + wN for w in {0, 0.25, 0.5, 0.75}.
        """

        print('Disturbance preparation: Gaussian noise...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))

            noise =  numpy.random.normal(loc=0, scale=1, size=img.shape)
            noise = (noise*255.0).astype(numpy.uint8)

            for i, w in enumerate([0.25, 0.5, 0.75]):
                outdir_distrub = f'{args.outdir}/{i+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)
                
                img_copy = img.copy()
                noisy = (1 - w)*img_copy + w*noise
                PIL.Image.fromarray(noisy.astype(numpy.uint8)).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'gblur':
        """ 
        Gaussian blur
        Gaussian kernel with standard deviation w in {0, 1, 2, 4}.
        """

        print('Disturbance preparation: Gaussian blur...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))
            
            for i, w in enumerate([1, 2, 4]):
                outdir_distrub = f'{args.outdir}/{i+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                img_copy = PIL.Image.fromarray(img.copy())
                blur = img_copy.filter(ImageFilter.GaussianBlur(radius=w))
                blur.save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'brectangles':
        """ 
        Black rectangles
        The size of the rectangles is w image size with w in {0, 0.25, 0.5, 0.75}.
        """

        print('Disturbance preparation: Black rectangles...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))
            H, W, C = img.shape
            
            for i, w in enumerate([0.25, 0.5, 0.75]):
                outdir_distrub = f'{args.outdir}/{i+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                img_copy = img.copy()
                for _ in range(5):
                    rhi = numpy.int32(H*w)
                    rwi = numpy.int32(W*w)
                    xpos = random.randint(0, H - rhi)
                    ypos = random.randint(0, W - rwi)
                    xdim = xpos + rhi
                    ydim = ypos + rwi

                    img_copy[xpos:xdim,ypos:ydim,:] = numpy.ones((rhi, rwi, C))*img_copy.min()
                PIL.Image.fromarray(img_copy).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'swirl':
        """ 
        Swirl (whirlpool effect)
        The disturbance level is given by the amount of swirl w in {0, 1, 2, 4}.
        """

        print('Disturbance preparation: Swirl...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))
            H, W, C = img.shape
            
            for i, w in enumerate([1, 2, 4]):
                outdir_distrub = f'{args.outdir}/{i+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                img_copy = img.copy()
                sign = numpy.sign(numpy.random.rand(1) - 0.5)[0] # directions random
                # positioning center
                xpos = H // 2
                ypos = W // 2
                center = (xpos, ypos)
                img_s = swirl(img_copy, rotation=0, strength=sign*w, radius=120, center=center)
                PIL.Image.fromarray((img_s*255).astype(numpy.uint8)).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'spnoise':
        """ 
        Salt and pepper noise
        The ratio of pixel flipped to white or black is given by the noise level w in {0, 0.1, 0.2, 0.3}.
        """

        print('Disturbance preparation: Salt and pepper noise...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))
            H, W, C = img.shape
            
            for i, w in enumerate([0.1, 0.2, 0.3]):
                outdir_distrub = f'{args.outdir}/{i+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                img_copy = img.copy()
                ns, d0, d1, d2 = img_copy.reshape(-1,H,W,C).shape
                coords = numpy.random.rand(ns,d0,d1) < w
                n_co = coords.sum()
                if n_co > 0:
                    vals = (numpy.random.rand(n_co) < 0.5).astype(numpy.float32)
                    vals[vals < 0.5] = -1
                    vals[vals > 0.5] = 1
                    for i in range(C):
                        img_copy.reshape(-1,H,W,C)[coords,i] = vals
                PIL.Image.fromarray(img_copy).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'shelter':
        """ 
        Shelter
        The image is divided into 10x10=100 regions and w in {0, 7, 14, 21} 
        of them are sheltered by a pixel sampled from the image.
        """

        print('Disturbance preparation: Shelter...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))

            for ii, shelter_num in enumerate([7, 14, 21]):
                outdir_distrub = f'{args.outdir}/{ii+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                images = [img.copy()]
                split_size = 10
                for i in range(len(images)):
                    mean_value = images[i][32,64,:]
                    choice_list_candidate = [m for m in range(split_size**2)]
                    choice_list = random.sample(choice_list_candidate, int(shelter_num))
                    for j in range(int(shelter_num)):
                        
                        index2 = numpy.mod(choice_list[j],split_size)
                        index1 = choice_list[j]//split_size
                        for k in range(3):
                            images[i][index1*round(256/split_size):(index1+1)*round(256/split_size), index2*round(256/split_size):(index2+1)*round(256/split_size), k] = mean_value[k]
                    PIL.Image.fromarray(numpy.uint8(images[i])).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'exchange':
        """ 
        Exchange
        The image is divided into 8x8=64 regions and random exchanges are performed w in {0, 4, 8, 16} times.
        """

        print('Disturbance preparation: Exchange...')
        for file in TqdmExtraFormat(
            files, total=len(files), ascii=" 0123456789#",
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

            name_file = file[file.rfind('/')+1:]
            img = numpy.asarray(PIL.Image.open(file))

            for k, change_num in enumerate([4, 8, 16]):
                outdir_distrub = f'{args.outdir}/{k+1}'
                if not os.path.exists(outdir_distrub):
                    os.makedirs(outdir_distrub)

                images = [img.copy()]
                split_size = 8
                for i in range(len(images)):
                    for j in range(change_num):
                        choice_list_candidate = [m for m in range(split_size**2)]
                        choice_list = random.sample(choice_list_candidate, 2)
                        index0_2 = numpy.mod(choice_list[0],split_size)
                        index0_1 = choice_list[0]//split_size
                        index1_2 = numpy.mod(choice_list[1],split_size)
                        index1_1 = choice_list[1]//split_size
                        temp0 = images[i][index0_1*round(256/split_size):(index0_1+1)*round(256/split_size), index0_2*round(256/split_size):(index0_2+1)*round(256/split_size), :].copy()
                        temp1 = images[i][index1_1*round(256/split_size):(index1_1+1)*round(256/split_size), index1_2*round(256/split_size):(index1_2+1)*round(256/split_size), :].copy()
                        images[i][index1_1*round(256/split_size):(index1_1+1)*round(256/split_size), index1_2*round(256/split_size):(index1_2+1)*round(256/split_size), :] = temp0
                        images[i][index0_1*round(256/split_size):(index0_1+1)*round(256/split_size), index0_2*round(256/split_size):(index0_2+1)*round(256/split_size), :] = temp1
                    PIL.Image.fromarray(numpy.uint8(images[i])).save(f'{outdir_distrub}/{name_file}')

    elif args.disturb == 'imagenet-contamination':
        """
        ImageNet contamination
        A percentage of w in {0, 0.25, 0.5, 0.75} of the Flickr Faces HQ images has been replaced by 
        ImageNet images. w = 0 means all images are from Flickr Faces HQ, w = 0.25 means that 75% of the 
        images are from Flickr Faces HQ and 25% from ImageNet etc.
        """
        if pathc is None:
            raise UserError("--pathc must be required.")

        print('Disturbance preparation: ImageNet contamination...')
        folders = glob.glob(f'{pathc}/**')
        filesc = [glob.glob(folder+r'**/*.[pjJ][npP][eEg]*') for folder in folders]
        datac = []
        size = files.__len__()
        while datac.__len__() < size:
            # Of the 1,000 ImageNet classes, 1,000 image of 50 classes 
            # are randomly chosen,which gives 50,000 ImageNet images.
            imgsc = random.choice(filesc)
            while imgsc.__len__() < 1000:
                imgsc = random.choice(filesc)
            datac = datac + imgsc[:1000]

        for i, w in enumerate([0.25, 0.5, 0.75]):
            outdir_distrub = f'{args.outdir}/{i+1}'
            if not os.path.exists(outdir_distrub):
                os.makedirs(outdir_distrub)
            
            imagenet = datac[:int(size*w)]
            flickr = files[int(size*w):]
            contaminated = flickr + imagenet

            print(f'{(1-w)*100:.1f}% of the images are from Flickr Faces HQ and {w*100:.1f}% from ImageNet.')
            print(f'{flickr.__len__()} images are from Flickr Faces HQ.')
            print(f'{imagenet.__len__()} images are from ImageNet.')
            print(f'{contaminated.__len__()} images are from dataset contaminated.')

            for file in TqdmExtraFormat(
                contaminated, total=len(contaminated), ascii=" 0123456789#",
                bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):

                name_file = file[file.rfind('/')+1:]
                img = PIL.Image.open(file)
                if img.mode != 'RGB':
                    img = img.convert(mode='RGB')
                img = img.resize((256, 256))
                img.save(f'{outdir_distrub}/{name_file}')

    else:
        raise UserError("--disturb must be 'gnoise', 'gblur', 'brectangles', 'swirl', 'spnoise', 'shelter', 'exchange' or 'imagenet-contamination'.")


#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()