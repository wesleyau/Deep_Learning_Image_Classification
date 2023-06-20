import os
import numpy as np

# Specify the directory you want to start from
rootDir = '/data/wesley/stats_vmin_vmax/npz_folder/TX_npy'

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith('.npy'):
            file_path = os.path.join(dirName, fname)
            data = np.load(file_path)
            print(f"Shape of data in file {fname} is {data.shape}")
