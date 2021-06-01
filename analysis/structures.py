import h5py
import os
from matplotlib import image
from matplotlib import pyplot as plt
import re
import numpy as np
from ipywidgets import interact, widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import morphology as ndmorph
import torch

dnets = [ "pconvnet_double", "lmorphnet_double", "smorphnet_double" ]
snets = [  "pconvnet", "lmorphnet", "smorphnet" ]
nets = { 2: dnets,
         1: snets }
selems = [ "cross3", "cross7", "diskaa2", "diskaa3", "diamondaa3", "complex" ]
dop = { "closing": 1,
        "opening": 0 }
sop = { "dilation": 1,
        "erosion": 0 }
ops = { 1: sop,
        2: dop }
var_names = { "pconvnet": "p",
              "lmorphnet": "p",
              "smorphnet": "alpha" }

layer_names = { "pconvnet": "pconv",
                "lmorphnet": "lm",
                "smorphnet": "sm" }

plot_params = { "xticks": [],
                "yticks": [] }

max_n_filter = 2 # Maximum number of filters per net
n_padding_column = 2 # Empty plots for clean labeling
global_column_size = max_n_filter + 1 # Number of columns per selem with padding

# Creating list of columns ratios, decreased size for label
# padding columns and spacing columns
width_ratios = np.ones(len(selems) * global_column_size + n_padding_column)
width_ratios[0] /= 2
width_ratios[1] /= 100
width_ratios[n_padding_column + max_n_filter::global_column_size] /= 5

meta_details = [ 'batch_size',
                 'dataset',
                 'loss',
                 'max_epochs',
                 'model',
                 'patience',
                 'sel_name',
                 'sel_size',
                 'vis_freq' ]