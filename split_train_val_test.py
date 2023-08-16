import os
import shutil
import random

# Setting the random seed for reproducibility
random.seed(42)

# Define the source directory and target directories
source_dir = '/data/wesley/3_data_7_31_23/TY_reflect_dataset/Fail'
train_dir = os.path.join('/data/wesley/3_data_7_31_23/train_test_reflection/TY', 'train')
val_dir = os.path.join('/data/wesley/3_data_7_31_23/train_test_reflection/TY', 'val')
test_dir = os.path.join('/data/wesley/3_data_7_31_23/train_test_reflection/TY', 'test')

# Create train, val, and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all folders in the current directory
all_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) 
               and d not in ['train', 'val', 'test']]

# Shuffle the folders to ensure random distribution
random.shuffle(all_folders)

# Determine the number of folders for each set
num_train = int(0.7 * len(all_folders))
num_val = int(0.15 * len(all_folders))
num_test = int(0.15 * len(all_folders))

# Split folders for each dataset
train_folders = all_folders[:num_train]
val_folders = all_folders[num_train:num_train+num_val]
test_folders = all_folders[num_train+num_val:num_train+num_val+num_test]

# Function to move folders
def move_folders(source, target_folders, target_dir):
    for folder in target_folders:
        shutil.move(os.path.join(source, folder), target_dir)

# Move folders to respective directories
move_folders(source_dir, train_folders, train_dir)
move_folders(source_dir, val_folders, val_dir)
move_folders(source_dir, test_folders, test_dir)
