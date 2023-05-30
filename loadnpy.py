import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os.path import isfile, join, splitext
from scipy.ndimage import gaussian_filter, morphology
from tqdm import tqdm
import math
import glob


path_to_files = "/data/wesley/"  
dir_list = os.listdir(path_to_files)
print(dir_list)

npz = np.load('TX202302140854_TX202302140945.npz')
print(npz.files)
print(npz['TC_image'].shape)

plt.imshow(npz['TC_image'])


image = plt.imshow(npz['masked_TX202301181108_TX202301181231_Co_image'],  vmin=116, vmax = 123)
print(npz['masked_TX202301181108_TX202301181231_Co_image'])
print(np.isnan(npz['masked_TX202301181108_TX202301181231_Co_image']))


mean_value = np.nanmean(npz['masked_TX202301181108_TX202301181231_Co_image'])

# Print the mean
print("Mean:", mean_value)

