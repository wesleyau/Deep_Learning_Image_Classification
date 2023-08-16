import os
import shutil

text_file_path = "/data/wesley/3_data_7_31_23/2021_2023_labels.txt"
source_directory = "/data/wesley/3_data_7_31_23/npz_folder_reflect/TY"
target_directory = "/data/wesley/3_data_7_31_23/TY_reflect_dataset"

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
        else:
            # Delete existing files in the target folder
            existing_files = os.listdir(target_folder_path)
            for file_name in existing_files:
                file_path = os.path.join(target_folder_path, file_name)
                os.remove(file_path)

        # Find matching PNG files and move them to the target folder
        for root, _, files in os.walk(source_directory):
            for file in files:
                if file.endswith('.png') and first_column in file:
                    source_file_path = os.path.join(root, file)
                    shutil.move(source_file_path, target_folder_path)

        # Delete the target folder if it's empty
        if not os.listdir(target_folder_path):
            os.rmdir(target_folder_path)
