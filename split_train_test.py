import os
import shutil
import math

# Define source directory and destination directory
source_directory = '/data/wesley/data2_6-27-23/TY_dataset'
destination_directory = '/data/wesley/data2_6-27-23/train_test/TY'

# Create train and test directories
os.makedirs(os.path.join(destination_directory, 'train'), exist_ok=True)
os.makedirs(os.path.join(destination_directory, 'test'), exist_ok=True)

# Define categories
categories = ['Pass', 'Fail']

# For each category, calculate number of samples and move them to corresponding directories
for category in categories:
    category_dir = os.path.join(source_directory, category)
    files = os.listdir(category_dir)
    
    total_samples = len(files)
    train_samples = math.floor(total_samples * 0.8)
    test_samples = total_samples - train_samples

    # Move train samples
    for i in range(train_samples):
        file = files[i]
        source = os.path.join(category_dir, file)
        destination = os.path.join(destination_directory, 'train', category, file)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
    
    # Move test samples
    for i in range(train_samples, total_samples):
        file = files[i]
        source = os.path.join(category_dir, file)
        destination = os.path.join(destination_directory, 'test', category, file)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)

print("Finished splitting dataset into train and test.")
