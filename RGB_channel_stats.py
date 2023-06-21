import os
import numpy as np
import statistics

# Initialize accumulators
count = 0
mean_acc = 0.0
sd_acc = 0.0
var_acc = 0.0

# Specify the directory you want to start from
rootDir = '/data/wesley/stats_vmin_vmax/npz_folder/TX_npy'

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith('.npy'):
            print(fname)
            file_path = os.path.join(dirName, fname)
            data = np.load(file_path)

            # Ignore the files containing Inf values
            if np.any(np.isinf(data)):
                print(f"File {file_path} contains Inf values. Ignoring this file in calculations.")
                continue

            # Exclude NaN values from calculations
            valid_data = data[~np.isnan(data)]

            if valid_data.size > 0:
                mean_acc += statistics.mean(valid_data)
                sd_acc += statistics.stdev(valid_data)
                var_acc += statistics.variance(valid_data)
                count += 1

# Compute averages across all files
mean_acc /= count

print(f"Processed {count} .npy files")
print(f"Mean: {mean_acc}")
print(f"SD: {sd_acc}")
print(f"Variance: {var_acc}")
