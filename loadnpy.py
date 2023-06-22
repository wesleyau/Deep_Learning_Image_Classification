import pandas as pd
import numpy as np
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
print(os.getcwd())

npz = np.load('../TX202201041204_TX202201041303.npz')
print(npz.files)
print(npz['Co_image'].shape)
print(npz['Co_image'])
plt.imshow(npz['Co_image'])
plt.axis('off')
print(npz['masked_TX202201040625_TX202201040729_Elin_image'].shape)

print(type(npz['TX202206101252_TX202206101337_Elin_image']))
print(npz['masked_TX202201040625_TX202201040729_Elin_image'].shape)
print(type(npz['TX202206101252_TX202206101337_Elin_image'][0, 0]))

print(npz['Elin_image'].shape)
print(npz['masked_TX202201040625_TX202201040729_Am_image'].isna())
print(npz['masked_TX202201040933_TX202201041024_Co_image'])
print(npz['masked_TX202201040625_TX202201040729_Eres_image'].shape)
print(npz['masked_TX202201040625_TX202201040729_Co_image'].shape)

plt.imshow(npz['TC_image'])


image = plt.imshow(npz['masked_TX202201040625_TX202201040729_Am_image'])
#image = plt.imshow(npz['Am_image'])
# Turn off axis and white space
plt.axis('off')
plt.autoscale(tight=True)

print(npz['masked_TX202201040933_TX202201041024_Am_image'])
print(np.isnan(npz['masked_TX202301181108_TX202301181231_Co_image']))


mean_value = np.nanmean(npz['masked_TX202301181108_TX202301181231_Co_image'])

# Print the mean
print("Mean:", mean_value)

from PIL import Image
import os
print(os.getcwd())
im = Image.open('/data/wesley/stats_vmin_vmax/npz_folder/TX/Co_image/masked_png_Co/masked_TX202201040625_TX202201040729_Co_image.png')
width, height = im.size
print("Width: ", width)
print("Height: ", height)