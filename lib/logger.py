# FEI University Center, São Bernardo do Campo, São Paulo, Brazil
# Images Processing Laboratory.
# Authors: Lucas F. Buzuti and Carlos E. Thomaz
#

import os
import numpy
import pickle
import PIL.Image
import plotly.graph_objects as go
from . import util


class Logger(object):
    def __init__(self, rank=0, log_dir='./logs', plot_dir='./plot', img_dir='./imgs'):
        self.stats = util.EasyDict()
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.img_dir = img_dir

        if rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
   
    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append([it, v])

    def add_plot(self, category1, k, category2=None):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=numpy.array(self.stats[category1][k])[:,0],
                y=numpy.array(self.stats[category1][k])[:,1],
                mode='lines',
                name=f'{category1}'
            )
        )
        if category2 is not None:
            fig.add_trace(
                go.Scatter(
                    x=numpy.array(self.stats[category2][k])[:,0],
                    y=numpy.array(self.stats[category2][k])[:,1],
                    mode='lines',
                    name=f'{category2}'
                )
            )

        fig.update_xaxes(title_text='Iterations', title_standoff=25)
        fig.update_yaxes(title_text=f'{k}', title_standoff=25)
        fig.write_html(os.path.join(self.plot_dir, f'plot_{k}_{category1}_{category2}.html'))

    def setup_snapshot_image_grid(self, training_set, random_seed=0):
        rnd = numpy.random.RandomState(random_seed)
        gw = numpy.clip(7680 // training_set.image_shape[2], 7, 32)
        gh = numpy.clip(4320 // training_set.image_shape[1], 4, 32)

        # Show random subset of training samples.
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

        # Load data.
        images = [training_set[i] for i in grid_indices]
        return (gw, gh), numpy.stack(images)

    def save_image_grid(self, img, drange, grid_size, fname='real.png'):
        outfile = os.path.join(self.img_dir, fname)

        lo, hi = drange
        img = numpy.asarray(img, dtype=numpy.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = numpy.rint(img).clip(0, 255).astype(numpy.uint8)

        gw, gh = grid_size
        _N, C, H, W = img.shape
        img = img.reshape(gh, gw, C, H, W)
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape(gh * H, gw * W, C)

        assert C in [1, 3]
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(outfile)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(outfile)

    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            print('Warning: file "%s" does not exist!' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
        except EOFError:
            print('Warning: log file corrupted!')
