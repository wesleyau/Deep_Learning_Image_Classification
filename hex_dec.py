import os
import shutil
from collections import defaultdict

# specify your path here
path = "/data/wesley/stats_vmin_vmax/labels_/TX"

# create a dictionary where keys are first 5 characters of the folders, values are lists of corresponding folders
folders = defaultdict(list)
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        folders[folder[:5]].append(folder)

# loop through the dictionary
for key in folders.keys():
    # create a new directory
    os.makedirs(os.path.join(path, key), exist_ok=True)
    # move all corresponding folders to the new directory
    for folder in folders[key]:
        shutil.move(os.path.join(path, folder), os.path.join(path, key, folder))
