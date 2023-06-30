import os
import shutil

text_file_path = "/data/wesley/2_data/npz_folder/2channel_labels.txt"
source_directory = "/data/wesley/2_data/npz_folder/TY"
target_directory = "/data/wesley/2_data/npz_folder/TY_dataset"

# Read the text file
with open(text_file_path, 'r') as file:
    lines = file.readlines()

# Process each line in the text file
for line in lines:
    columns = line.strip().split('\t')
    if len(columns) == 3:
        first_column = columns[0]
        second_column = columns[1]
        third_column = columns[2]

        folder_name = f"{second_column}_{first_column}_{third_column}"
        target_folder_path = os.path.join(target_directory, folder_name)

        # Create the target folder if it doesn't exist
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        # Find matching PNG files and move them to the target folder
        for root, _, files in os.walk(source_directory):
            for file in files:
                if file.endswith('.png') and first_column in file:
                    source_file_path = os.path.join(root, file)
                    shutil.move(source_file_path, target_folder_path)
