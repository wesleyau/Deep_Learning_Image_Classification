import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy.ndimage import measurements
from skimage.segmentation import find_boundaries
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import fourier_gaussian
from scipy.fftpack import fftshift, ifftshift, fftn, ifftn

# Specify the main directory path containing the subdirectories with .npy files
main_directory = '/data/wesley/3_data_7_31_23/npz_folder/TY'

# Padding parameters
padding_size = 1  # Size of the border extension
padding_value = np.nan  # Value to fill the border with

# I chose a elliptical structuring element for binary dilation because it makes the corners mask better instead of doing a square structure
structuring_element = generate_binary_structure(2, 2)  # Increase the second argument to adjust the size and shape

# Recursively iterate over all subdirectories and files within the main directory
for dirpath, dirnames, filenames in os.walk(main_directory):
    # Iterate over the filenames in the current directory
    for file_name in filenames:
        # Check if the file is a .npy file
        if file_name.endswith('.npy'):
            file_path = os.path.join(dirpath, file_name)
            image_array = np.load(file_path)
            
            # Printing names of each file just in case some of the files are corrupt and need to ask for rerun
            print(file_name)
            
            # Check if the image is Eres_image or Elin_image
            if 'Eres_image' in file_name or 'Elin_image' in file_name:
                # Padding the image array
                padded_image_array = np.pad(image_array, pad_width=padding_size, mode='constant', constant_values=padding_value)
            else:
                padded_image_array = image_array
            
            # Setting mask to nans, 1, and 0
            mask = np.isnan(padded_image_array) | (padded_image_array == 1) | (padded_image_array == 0)
            masked_image = np.where(mask, np.nan, padded_image_array)  # Apply the mask to the image
            
            # Determine the number of iterations for binary dilation based on the image class
            iterations = 1  # Default number of iterations
            if 'Co_image' in file_name:
                iterations = 1  # Set a different number of iterations for Elin_image
            elif 'Am_image' in file_name:
                iterations = 1  # Set a different number of iterations for Am_image
            elif 'Elin_image' in file_name:
                iterations = 7  # Set a different number of iterations for Elin_image
            elif 'Eres_image' in file_name:
                iterations = 6  # Set a different number of iterations for Eres_image
            elif 'TC_image' in file_name:
                iterations = 3  # Set a different number of iterations for TC_image
                
            dilated_mask = binary_dilation(mask, iterations=iterations, structure=structuring_element)  # Dilate the mask with the specified number of iterations and structuring element
            inverted_mask = np.logical_not(dilated_mask)  # Invert the mask
            masked_image = np.where(inverted_mask, masked_image, np.nan)  # Apply the inverted mask to the image
            
            # Exclude values under a minimum threshold of 0.01
            masked_image[masked_image < 0.01] = np.nan
            
            # Find the non-zero indices in the array
            non_zero_indices = np.nonzero(np.any(masked_image != 0, axis=-1))
            
            # Check if non-zero pixels are found
            if len(non_zero_indices) > 0 and all(len(indices) > 0 for indices in non_zero_indices):
                # Determine the cropping range based on the dimensions of the non-zero indices tuple
                cropping_range = tuple(slice(np.min(indices), np.max(indices) + 1) for indices in non_zero_indices)
    
                # Crop the image array
                cropped_image_array = masked_image[cropping_range]
                
                # If the image is Eres_image or Elin_image, remove padding
                if 'Eres_image' in file_name or 'Elin_image' in file_name:
                    cropped_image_array = cropped_image_array[padding_size:-padding_size, padding_size:-padding_size]
                    
                # Get the relative path of the file within the main directory
                relative_path = os.path.relpath(file_path, main_directory)
                
                # Get the directory path within the main directory
                save_directory = os.path.dirname(os.path.join(main_directory, relative_path))
    
                # Extract the image class from the fil#bbox_inches='tight', name
                image_class = ''
                if 'Co_image' in file_name:
                    image_class = 'Co'
                elif 'Am_image' in file_name:
                    image_class = 'Am'
                elif 'Elin_image' in file_name:
                    image_class = 'Elin'
                elif 'Eres_image' in file_name:
                    image_class = 'Eres'
                elif 'TC_image' in file_name:
                    image_class = 'TC'
    
                # Save directories if they don't exist
                modified_npy_directory = os.path.join(save_directory, 'masked_npy_' + image_class)
                os.makedirs(modified_npy_directory, exist_ok=True)
                cropped_png_directory = os.path.join(save_directory, 'masked_png_' + image_class)
                os.makedirs(cropped_png_directory, exist_ok=True)
    
                # Determine vmin and vmax based on filename
                vmin, vmax = None, None
                if 'Co_image' in file_name:
                    vmin, vmax = 116.36538997789857, 123.5873231832517
                elif 'Am_image' in file_name:
                    vmin, vmax = 59.99483151311375, 65.26089135691268
                elif 'Elin_image' in file_name:
                    vmin, vmax = 1.840720640553683, 1.980556323891534  # This may need to be fiddled with, this image is most variable between the tx and ty machines
                elif 'Eres_image' in file_name:
                    vmin, vmax = 0.07705189479517403, 0.12010585273252103
                elif 'TC_image' in file_name:
                    vmin, vmax = 0.9625918840019665, 1.0476702829489846
                else:
                    print(file_name + " does not contain any substring in the filename")  # Default values if no specific word is found
                
                # Save the modified .npy file in the respective directory with a modified filename
                modified_npy_file_name = f"masked_{file_name}"
                modified_npy_file_path = os.path.join(modified_npy_directory, modified_npy_file_name)
                
                # Save the pixel values within the specified vmin and vmax ranges
                np.save(modified_npy_file_path, np.clip(cropped_image_array, vmin, vmax))
    
                # Create a Matplotlib figure and axis
                fig, ax = plt.subplots(figsize=(256/80, 256/80), dpi=80)
    
                # Display the image without axis and white space
                ax.imshow(cropped_image_array, vmin=vmin, vmax=vmax)

                # Turn off axis and white space
                ax.axis('off')
                ax.autoscale(tight=True)
    
                # Set NaN values to black
                ax.set_facecolor('black')
    
                # Save the cropped image in the respective directory with a modified filename
                cropped_png_file_name = f"masked_{file_name[:-4]}.png"
                cropped_png_file_path = os.path.join(cropped_png_directory, cropped_png_file_name)
                #bbox_inches='tight', 
                plt.savefig(cropped_png_file_path, dpi='figure', bbox_inches='tight', pad_inches=0, facecolor='black')
                plt.close(fig)
                
                # Apply Gaussian high pass filter
                # Convert to Fourier space
                fourier_image = fftshift(fftn(cropped_image_array))
                
                # Apply a Gaussian low-pass filter in the Fourier domain
                # The higher this value, the more low frequency components are removed
                sigma = 20  # Modify this according to your requirements
                low_pass = fourier_gaussian(fourier_image, sigma=sigma)
                
                # Convert back to the spatial domain
                low_pass = np.real(ifftn(ifftshift(low_pass)))
                
                # Subtract the low-pass filtered image from the original image to get a high-pass filtered image
                high_pass = cropped_image_array - low_pass
                
                # Save the high-pass filtered image
                high_pass_png_directory = os.path.join(save_directory, 'high_pass_png_' + image_class)
                os.makedirs(high_pass_png_directory, exist_ok=True)
                
                high_pass_png_file_name = f"high_pass_{file_name[:-4]}.png"
                high_pass_png_file_path = os.path.join(high_pass_png_directory, high_pass_png_file_name)
                
                fig, ax = plt.subplots(figsize=(256/80, 256/80), dpi=80)
                ax.imshow(high_pass, cmap='gray')  # Set color map to grayscale
                ax.axis('off')
                ax.autoscale(tight=True)
                ax.set_facecolor('black')
                plt.savefig(high_pass_png_file_path, dpi='figure', bbox_inches='tight', pad_inches=0, facecolor='black')
                plt.close(fig)
            else:
                print(f"No non-zero pixels found in {file_name}. Skipping...")
