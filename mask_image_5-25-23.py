import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

# Specify the folder path containing the .npy files
folder_path = '/data/wesley/NPZ_Folder/TX_Machine/Co_image'

# Get a list of all .npy files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Iterate over the .npy files to crop them
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    image_array = np.load(file_path)
    
    # Brandon's code
    mask = np.isnan(image_array) | (image_array == 1) | (image_array == 0)
    (n_rows,n_cols)=mask.shape
    
    border_size=5
    mask[:border_size,:]=1
    mask[(n_rows-border_size):,:]=1
    mask[:,:border_size]=1
    mask[:,(n_cols-border_size):]=1
    
    
    #print(mask.shape)
    #plt.imshow(mask)
    #plt.show()
    #print(aaa)
    masked_image = np.where(mask, np.nan, image_array)  # Apply the mask to the image
    
    # Determine the number of iterations for binary dilation based on the image class
    iterations = 3  # Default number of iterations
    if 'Elin_image' in file_name:
        iterations = 5  # Set a different number of iterations for Elin_image
    elif 'Eres_image' in file_name:
        iterations = 5  # Set a different number of iterations for Eres_image
        
    dilated_mask = binary_dilation(mask, iterations=iterations)  # Dilate the mask
    #dilated_mask2 = binary_dilation(dilated_mask, iterations = 14)
    #eroded_mask2 = binary_erosion(dilated_mask2, iterations=1)
    
    
    inverted_mask = np.logical_not(dilated_mask)  # Invert the mask
    #inverted_mask = np.logical_not(eroded_mask2)  # Invert the mask
    masked_image = np.where(inverted_mask, masked_image, np.nan)  # Apply the inverted mask to the image
    # End Brandon's code
    
    # Find the non-zero indices in the array
    non_zero_indices = np.nonzero(np.any(masked_image != 0, axis=-1))
    
    # Check if non-zero pixels are found
    if len(non_zero_indices) > 0 and all(len(indices) > 0 for indices in non_zero_indices):
        # Determine the cropping range based on the dimensions of the non-zero indices tuple
        cropping_range = tuple(slice(np.min(indices), np.max(indices) + 1) for indices in non_zero_indices)

        # Crop the image array
        cropped_image_array = masked_image[cropping_range]

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Determine vmin and vmax based on filename
        if 'Co_image' in file_name:
            vmin, vmax = 116, 123
        elif 'Am_image' in file_name:
            vmin, vmax = 59, 64
        elif 'Elin_image' in file_name:
            vmin, vmax = 1.9, 2
        elif 'Eres_image' in file_name:
            vmin, vmax = 0.09, 0.11
        elif 'TC_image' in file_name:
            vmin, vmax = 0.98, 1.02
        else:
            print(file_name + " does not contain any substring in the filename")  # Default values if no specific word is found
        
        # Display the image without axis and white space
        ax.imshow(cropped_image_array, vmin=vmin, vmax=vmax)

        # Turn off axis and white space
        ax.axis('off')
        ax.autoscale(tight=True)

        # Save the cropped image as PNG
        cropped_file_path = os.path.join(folder_path, f"cropped_{file_name[:-4]}.png")
        plt.savefig(cropped_file_path, dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        print(f"No non-zero pixels found in {file_name}. Skipping...")
