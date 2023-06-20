import numpy as np
import os
from PIL import Image

image_directory = '/data/wesley/stats_vmin_vmax/npz_folder/TX'

# Variables to store cumulative sums
pixel_sum = np.zeros(3, dtype=np.float64)
pixel_squared_sum = np.zeros(3, dtype=np.float64)
total_images = 0
total_pixels = 0

# Iterate over the PNG images in the directory and its subdirectories
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)

            # Accumulate pixel sums
            pixel_sum += np.sum(image_array, axis=(0, 1))
            pixel_squared_sum += np.sum(np.square(image_array), axis=(0, 1))
            total_images += 1
            total_pixels += image_array.shape[0] * image_array.shape[1]

# Calculate mean and standard deviation per channel
if total_images > 0 and total_pixels > 0:
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_squared_sum / total_pixels - mean ** 2)
else:
    mean = np.zeros(3)
    std = np.zeros(3)

print("Mean per channel:", mean)
print("Standard deviation per channel:", std)
print("Total number of images:", total_images)
