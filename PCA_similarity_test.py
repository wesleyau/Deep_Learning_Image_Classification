import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
                data_list.append(flattened_data)

    return data_list

# Provide the paths to your folders
tx_directory = '/data/wesley/NPZ_Folder/TX_Machine/Am_image'  # Compare any image class from both the TX and TY machines, raw or masked or already
ty_directory = '/data/wesley/NPZ_Folder/TY_Machine/Am_image'  # generally they should be the same image classes though

# Extract the image type from the filenames in the directory
tx_file_list = os.listdir(tx_directory)
ty_file_list = os.listdir(ty_directory)

image_type = None

for file_name in tx_file_list:
    if any(tag in file_name for tag in ["Am", "Co", "Eres", "Elin", "TC"]):
        image_type = next((tag for tag in ["Am", "Co", "Eres", "Elin", "TC"] if tag in file_name), None)
        break

if image_type is None:
    for file_name in ty_file_list:
        if any(tag in file_name for tag in ["Am", "Co", "Eres", "Elin", "TC"]):
            image_type = next((tag for tag in ["Am", "Co", "Eres", "Elin", "TC"] if tag in file_name), None)
            break

# Load and flatten the datasets
tx_data = load_and_flatten_data(tx_directory)
ty_data = load_and_flatten_data(ty_directory)

# Determine the minimum length among the arrays in the combined data
min_length = min(min(len(data) for data in tx_data), min(len(data) for data in ty_data))

# Trim the arrays in the combined data to have the same length
tx_data = [data[:min_length] for data in tx_data]
ty_data = [data[:min_length] for data in ty_data]

# Combine the datasets into one list
combined_data = np.concatenate([tx_data, ty_data], axis=0)

# Set the style to "seaborn"
plt.style.use('seaborn')

# Fit the PCA model
pca = PCA(n_components=2)
principal_components = pca.fit_transform(combined_data)

# Separate the PCA results for the two types of images
tx_pca = principal_components[:len(tx_data)]
ty_pca = principal_components[len(tx_data):]

# Visualize the PCA results
plt.scatter(tx_pca[:, 0], tx_pca[:, 1], color='blue', alpha=0.5, label='TX')
plt.scatter(ty_pca[:, 0], ty_pca[:, 1], color='red', alpha=0.5, label='TY')
plt.legend()
plt.title(f'{image_type} TX and TY PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
