import os
import shutil

# Function to move files from source to destination folder
def move_files(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    if os.path.isfile(source):
        shutil.move(source, destination)

# Path to the directory containing the folders and files
directory_path = "/data/wesley/data2_6-27-23/npz_folder/TY"

# Path to the text file containing the words
txt_file_path = "/data/wesley/data2_6-27-23/labels.txt"

# Output directory to move the files
output_directory = "/data/wesley/data2_6-27-23/TY_dataset"

# Read the words from the text file
data = {}
with open(txt_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        cols = line.strip().split()
        if len(cols) >= 3:
            key = cols[0]
            value = f"{cols[1]}_{key}_{cols[2]}"
            data[key] = value

# Traverse through the directory and move files based on words
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        _, ext = os.path.splitext(file_name)
        if ext.lower() == '.png':  # Checking for PNG files
            for key, value in data.items():
                if key in file_name:
                    source_file_path = os.path.join(root, file_name)
                    destination_folder = os.path.join(output_directory, value)
                    move_files(source_file_path, destination_folder) 
                    break