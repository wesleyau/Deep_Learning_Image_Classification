import os
import shutil
import pandas as pd

# Define the directories
source_dir = '/data/wesley/stats_vmin_vmax/npz_folder/TY' # Replace with the directory containing your png images
target_dir = '/data/wesley/stats_vmin_vmax/labels/TY' # Replace with the directory where you want to move the images
txt_file_path = '/data/wesley/stats_vmin_vmax/labels.txt' # Replace with your txt file path

# Load the txt file into a pandas dataframe
df = pd.read_csv(txt_file_path, sep='\t', header=None, names=['filename', 'id', 'classifier'])

# Function to find all png files in directory and subdirectories
def find_png_files(directory, filename):
    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') and filename in file:
                yield os.path.join(dirpath, file)

# Go through each row in the dataframe
for index, row in df.iterrows():
    # Find all png files containing the filename
    png_files = list(find_png_files(source_dir, row['filename']))

    # If any png files were found, move them to a new folder
    if png_files:
        # Create the new folder
        new_folder_path = os.path.join(target_dir, f"{row['id']}_{row['filename']}_{row['classifier']}")
        os.makedirs(new_folder_path, exist_ok=True)

        # Move the png files to the new folder
        for png_file in png_files:
            shutil.move(png_file, new_folder_path)
