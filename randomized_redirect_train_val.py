import os
import random
import shutil
import math

# Set the source directory containing the crystals
source_directory = '/data/wesley/data2/dataset_tx_test'

# Set the destination directory for train, val, and unlabeled crystals
destination_directory = '/data/wesley/data2/test_dataset_tx'

# Set the desired percentages for train and val
train_percentage = 0.8
val_percentage = 0.2

# Create the train, val, and unlabeled directories
os.makedirs(os.path.join(destination_directory, 'train', 'pass'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'train', 'fail'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'train', 'unlabeled'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'val', 'pass'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'val', 'fail'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'val', 'unlabeled'), exist_ok=True)

# Iterate over the source directory and move the crystals to train, val, or unlabeled folders
for class_name in ['pass', 'fail', 'unlabeled']:
    class_directory = os.path.join(source_directory, class_name)
    crystals = os.listdir(class_directory)
    random.shuffle(crystals)
    num_crystals = len(crystals)
    num_train = math.ceil(num_crystals * train_percentage)
    num_val = math.floor(num_crystals * val_percentage)

    # Move crystals to the train folder
    for i, crystal in enumerate(crystals[:num_train]):
        source_path = os.path.join(class_directory, crystal)
        crystal_name = crystal.split('.')[0]
        dest_path = os.path.join(destination_directory, 'train', class_name, f'{crystal_name}')
        shutil.move(source_path, dest_path)

    # Move crystals to the val folder
    for i, crystal in enumerate(crystals[num_train:num_train + num_val]):
        source_path = os.path.join(class_directory, crystal)
        crystal_name = crystal.split('.')[0]
        dest_path = os.path.join(destination_directory, 'val', class_name, f'{crystal_name}')
        shutil.move(source_path, dest_path)

    # Move remaining crystals to the train folder if there are any
    for i, crystal in enumerate(crystals[num_train + num_val:]):
        source_path = os.path.join(class_directory, crystal)
        crystal_name = crystal.split('.')[0]
        dest_path = os.path.join(destination_directory, 'train', class_name, f'{crystal_name}')
        shutil.move(source_path, dest_path)
