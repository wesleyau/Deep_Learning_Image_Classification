import os
import shutil

# Function to move files from source to destination folder
def move_files(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    if os.path.isfile(source):
        shutil.move(source, destination) #change this to shutil.copy if you don't want to delete the files from the original source

# Path to the directory containing the folders and files
directory_path = "/data/wesley/data2/mask_ty_png_no_replacement"

# Path to the text file containing the words
txt_file_path = "/data/wesley/2_data/npz_folder/2channel_labels.txt"

# Output directory to move the files
output_directory = "/data/wesley/data2/dataset_ty/labeled/fail"

# Read the words from the text file
with open(txt_file_path, 'r') as file:
    words = file.read().splitlines()

# Traverse through the directory and move files based on words
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        for word in words:
            if word in file_name:
                source_file_path = os.path.join(root, file_name)
                destination_folder = os.path.join(output_directory, word)
                move_files(source_file_path, destination_folder) 
                break
