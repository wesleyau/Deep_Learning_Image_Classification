import os
import shutil
import numpy as np

# This code iterates over npz folders and moves them in folders of their own class
# first you need to move all npz folders into their own directory 
# then move all the TX and TY data into seperate folders
# I used unix to do all of the above
# run this on each of the TX and TY folders
# I usually use unix to move the npz folders into their own directory outside of the tx and ty folders

# Define the source directory containing the .npz folders
source_directory = "/data/wesley/stats_vmin_vmax/npz_folder/TX"

# Define the destination directory where the .npy files will be moved
destination_directory = "/data/wesley/stats_vmin_vmax/npz_folder/TX"

# Iterate over the .npz folders in the source directory
for npz_folder in os.listdir(source_directory):
    npz_folder_path = os.path.join(source_directory, npz_folder)

    # Check if the item in the source directory is a file with .npz extension
    if os.path.isfile(npz_folder_path) and npz_folder.endswith(".npz"):
        # Load the .npz file
        npz_data = np.load(npz_folder_path)
        
        # Iterate over the keys (filenames) in the .npz file
        for npy_file in npz_data.files:
            npy_file_path = npz_data[npy_file]
            
            # Check if the item in the .npz file is an ndarray
            if isinstance(npy_file_path, np.ndarray):
                
                # Extract the filename without the extension
                npy_filename = os.path.splitext(npy_file)[0]
                
                # Add the initial identifier to each npy filename just incase you need to move them back into folders
                crystal_npy_filename = os.path.splitext(npz_folder)[0] + "_" + npy_filename
                
                # Create the destination folder path based on the filename
                destination_folder_path = os.path.join(destination_directory, npy_filename)
                
                # Create the destination folder if it doesn't exist
                if not os.path.exists(destination_folder_path):
                    os.makedirs(destination_folder_path)
                
                # Save the .npy file to the destination folder
                npy_file_save_path = os.path.join(destination_folder_path, crystal_npy_filename + ".npy")
                np.save(npy_file_save_path, npy_file_path)
