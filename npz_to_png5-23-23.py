import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the folder path containing the .npy files
folder_path = '/data/wesley/crop_test_NPZ_Folder/TX_Machine/TC_image'

# Get a list of all .npy files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Iterate over the .npy files to crop them
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    image_array = np.load(file_path)

    # Find the non-zero indices in the array
    non_zero_indices = np.nonzero(np.any(image_array != 0, axis=-1))
    
    # Check if non-zero pixels are found
    if len(non_zero_indices) > 0 and all(len(indices) > 0 for indices in non_zero_indices):
        # Determine the cropping range based on the dimensions of the non-zero indices tuple
        cropping_range = tuple(slice(np.min(indices), np.max(indices) + 1) for indices in non_zero_indices)

        # Crop the image array
        cropped_image_array = image_array[cropping_range]

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Display the image without axis and white space
        ax.imshow(cropped_image_array)

        # Turn off axis and white space
        ax.axis('off')
        ax.autoscale(tight=True)

        # Save the cropped image as PNG
        cropped_file_path = os.path.join(folder_path, f"cropped_{file_name[:-4]}.png")
        plt.savefig(cropped_file_path, dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        print(f"No non-zero pixels found in {file_name}. Skipping...")
