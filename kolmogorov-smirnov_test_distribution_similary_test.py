import os
import numpy as np
from scipy.stats import ks_2samp

def load_and_flatten_data(folder_path):
    file_list = os.listdir(folder_path)
    data_list = []

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.npy'):
            data = np.load(file_path)
            flattened_data = data.flatten()  # Flatten the 2D data into a 1D array

            # Exclude 0, 1, and NaN values
            flattened_data = flattened_data[(flattened_data != 0) & (flattened_data != 1) & (~np.isnan(flattened_data))]

            if flattened_data.size > 0:
                data_list.extend(flattened_data)

    return np.array(data_list)

# Provide the paths to your folders
tx_directory = '/data/wesley/3_data_7_31_23/npz_folder/TX/TC_image'  # TX machine directory
ty_directory = '/data/wesley/3_data_7_31_23/npz_folder/TY/TC_image'  # TY machine directory

# Load and flatten the datasets
tx_data = load_and_flatten_data(tx_directory)
ty_data = load_and_flatten_data(ty_directory)

# Perform the K-S test
ks_statistic, p_value = ks_2samp(tx_data, ty_data)

# Transform the p-value
log_p_value = -np.log10(p_value + np.finfo(float).eps)

print('KS statistic:', ks_statistic)
print('P-value:', p_value)
print('Transformed p-value: ~10^' + str(-log_p_value))
#if you see that the transformed p-value is not changing across different image classes, this is not an indication that something is wrong. 
# The p-value is so small that it's actually below the machine precision of your computer (around 10^-16 for typical double precision). 
# In other words, the p-value is so close to zero that it's essentially zero.
