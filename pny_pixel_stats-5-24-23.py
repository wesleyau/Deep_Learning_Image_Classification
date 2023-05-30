import numpy as np
import os

def collect_statistics(folder_path):
    file_list = os.listdir(folder_path)
    folder_stats = []

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isdir(file_path):
            subfolder_stats = collect_statistics(file_path)
            folder_stats.extend(subfolder_stats)
        elif file_name.endswith('.npy'):
            data = np.load(file_path)

            # Skip NaN, 0, and 1 values
            data = data[(data != 0) & (data != 1) & (~np.isnan(data))]

            if data.size > 0:
                # Calculate summary statistics and quantile ranges for pixel intensities
                stats = {
                    'File': file_name,
                    'Min': np.min(data),
                    'Max': np.max(data),
                    'Mean': np.mean(data),
                    'Median': np.median(data),
                    'Std': np.std(data),
                    'Quantile_1': np.percentile(data, 1),
                    'Quantile_5': np.percentile(data, 5),
                    'Quantile_10': np.percentile(data, 10),
                    'Quantile_25': np.percentile(data, 25),
                    'Quantile_50': np.percentile(data, 50),
                    'Quantile_75': np.percentile(data, 75),
                    'Quantile_90': np.percentile(data, 90),
                    'Quantile_95': np.percentile(data, 95),
                    'Quantile_99': np.percentile(data, 99)
                }
                folder_stats.append(stats)

    return folder_stats

def save_statistics_to_file(folder_path, folder_stats):
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(folder_path, folder_name + '_stats.txt')
    with open(output_file, 'w') as file:
        file.write('Folder Statistics: ' + folder_name + '\n')
        file.write('\n')

        # Write statistics for each file
        for stats in folder_stats:
            file.write('File: ' + stats['File'] + '\n')
            file.write('Min: ' + str(stats['Min']) + '\n')
            file.write('Max: ' + str(stats['Max']) + '\n')
            file.write('Mean: ' + str(stats['Mean']) + '\n')
            file.write('Median: ' + str(stats['Median']) + '\n')
            file.write('Std: ' + str(stats['Std']) + '\n')
            file.write('Quantile 1: ' + str(stats['Quantile_1']) + '\n')
            file.write('Quantile 5: ' + str(stats['Quantile_5']) + '\n')
            file.write('Quantile 10: ' + str(stats['Quantile_10']) + '\n')
            file.write('Quantile 25: ' + str(stats['Quantile_25']) + '\n')
            file.write('Quantile 50: ' + str(stats['Quantile_50']) + '\n')
            file.write('Quantile 75: ' + str(stats['Quantile_75']) + '\n')
            file.write('Quantile 90: ' + str(stats['Quantile_90']) + '\n')
            file.write('Quantile 95: ' + str(stats['Quantile_95']) + '\n')
            file.write('Quantile 99: ' + str(stats['Quantile_99']) + '\n')
            file.write('\n')

    print('Statistics saved to', output_file)

# Provide the main directory path
main_directory = '/data/wesley/NPZ_Folder'

# Iterate over subfolders and collect statistics
for root, dirs, files in os.walk(main_directory):
    folder_stats = collect_statistics(root)

    if folder_stats:
        # Save statistics to a text file in the same directory
        save_statistics_to_file(root, folder_stats)
