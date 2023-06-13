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

            # Flatten the data array
            data = data.flatten()

            # Skip 0, 1, NaN, and infinite values
            data = data[(data != 0) & (data != 1) & (~np.isnan(data)) & (~np.isinf(data)) & (data != 255)]

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
                    'Quantile_2': np.percentile(data, 2),
                    'Quantile_3': np.percentile(data, 3),
                    'Quantile_4': np.percentile(data, 4),
                    'Quantile_5': np.percentile(data, 5),
                    'Quantile_10': np.percentile(data, 10),
                    'Quantile_25': np.percentile(data, 25),
                    'Quantile_50': np.percentile(data, 50),
                    'Quantile_75': np.percentile(data, 75),
                    'Quantile_90': np.percentile(data, 90),
                    'Quantile_95': np.percentile(data, 95),
                    'Quantile_96': np.percentile(data, 96),
                    'Quantile_97': np.percentile(data, 97),
                    'Quantile_98': np.percentile(data, 98),
                    'Quantile_99': np.percentile(data, 99)
                }
                folder_stats.append(stats)

    return folder_stats

def save_statistics_to_file(folder_path, folder_stats, output_folder, total_files, overall_stats):
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(output_folder, folder_name + '_stats.txt')
    with open(output_file, 'w') as file:
        file.write('Folder Statistics: ' + folder_name + '\n')
        file.write('Total Files: ' + str(total_files) + '\n')
        file.write('\n')

        # Write overall statistics for the folder
        file.write('Overall Statistics\n')
        file.write('Min: ' + str(overall_stats['Min']) + '\n')
        file.write('Max: ' + str(overall_stats['Max']) + '\n')
        file.write('Mean: ' + str(overall_stats['Mean']) + '\n')
        file.write('Median: ' + str(overall_stats['Median']) + '\n')
        file.write('Std: ' + str(overall_stats['Std']) + '\n')
        file.write('Quantile 1: ' + str(overall_stats['Quantile_1']) + '\n')
        file.write('Quantile 2: ' + str(overall_stats['Quantile_2']) + '\n')
        file.write('Quantile 3: ' + str(overall_stats['Quantile_3']) + '\n')
        file.write('Quantile 4: ' + str(overall_stats['Quantile_4']) + '\n')
        file.write('Quantile 5: ' + str(overall_stats['Quantile_5']) + '\n')
        file.write('Quantile 10: ' + str(overall_stats['Quantile_10']) + '\n')
        file.write('Quantile 25: ' + str(overall_stats['Quantile_25']) + '\n')
        file.write('Quantile 50: ' + str(overall_stats['Quantile_50']) + '\n')
        file.write('Quantile 75: ' + str(overall_stats['Quantile_75']) + '\n')
        file.write('Quantile 90: ' + str(overall_stats['Quantile_90']) + '\n')
        file.write('Quantile 95: ' + str(overall_stats['Quantile_95']) + '\n')
        file.write('Quantile 96: ' + str(overall_stats['Quantile_96']) + '\n')
        file.write('Quantile 97: ' + str(overall_stats['Quantile_97']) + '\n')
        file.write('Quantile 98: ' + str(overall_stats['Quantile_98']) + '\n')
        file.write('Quantile 99: ' + str(overall_stats['Quantile_99']) + '\n')
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
            file.write('Quantile 2: ' + str(stats['Quantile_2']) + '\n')
            file.write('Quantile 3: ' + str(stats['Quantile_3']) + '\n')
            file.write('Quantile 4: ' + str(stats['Quantile_4']) + '\n')
            file.write('Quantile 5: ' + str(stats['Quantile_5']) + '\n')
            file.write('Quantile 10: ' + str(stats['Quantile_10']) + '\n')
            file.write('Quantile 25: ' + str(stats['Quantile_25']) + '\n')
            file.write('Quantile 50: ' + str(stats['Quantile_50']) + '\n')
            file.write('Quantile 75: ' + str(stats['Quantile_75']) + '\n')
            file.write('Quantile 90: ' + str(stats['Quantile_90']) + '\n')
            file.write('Quantile 95: ' + str(stats['Quantile_95']) + '\n')
            file.write('Quantile 96: ' + str(stats['Quantile_96']) + '\n')
            file.write('Quantile 97: ' + str(stats['Quantile_97']) + '\n')
            file.write('Quantile 98: ' + str(stats['Quantile_98']) + '\n')
            file.write('Quantile 99: ' + str(stats['Quantile_99']) + '\n')
            file.write('\n')

    print('Statistics saved to', output_file)

# Provide the main directory paths
tx_directory = '/data/wesley/stats_vmin_vmax/npz_folder/TX'  # TX machine directory
ty_directory = '/data/wesley/stats_vmin_vmax/npz_folder/TY'  # TY machine directory
output_folder = 'statistics'  # Folder to save the statistics files

# Create the output folder within the TX directory
tx_output_folder = os.path.join(tx_directory, output_folder)
os.makedirs(tx_output_folder, exist_ok=True)

# Create the output folder within the TY directory
ty_output_folder = os.path.join(ty_directory, output_folder)
os.makedirs(ty_output_folder, exist_ok=True)

# Traverse the TX machine directory and collect statistics
for root, dirs, files in os.walk(tx_directory):
    folder_stats = collect_statistics(root)
    total_files = len(files)

    if folder_stats:
        # Calculate overall statistics for the folder
        overall_stats = {
            'Min': np.min([stats['Min'] for stats in folder_stats]),
            'Max': np.max([stats['Max'] for stats in folder_stats]),
            'Mean': np.mean([stats['Mean'] for stats in folder_stats]),
            'Median': np.median([stats['Median'] for stats in folder_stats]),
            'Std': np.mean([stats['Std'] for stats in folder_stats]),
            'Quantile_1': np.percentile([stats['Quantile_1'] for stats in folder_stats], 1),
            'Quantile_2': np.percentile([stats['Quantile_2'] for stats in folder_stats], 2),
            'Quantile_3': np.percentile([stats['Quantile_3'] for stats in folder_stats], 3),
            'Quantile_4': np.percentile([stats['Quantile_4'] for stats in folder_stats], 4),
            'Quantile_5': np.percentile([stats['Quantile_5'] for stats in folder_stats], 5),
            'Quantile_10': np.percentile([stats['Quantile_10'] for stats in folder_stats], 10),
            'Quantile_25': np.percentile([stats['Quantile_25'] for stats in folder_stats], 25),
            'Quantile_50': np.percentile([stats['Quantile_50'] for stats in folder_stats], 50),
            'Quantile_75': np.percentile([stats['Quantile_75'] for stats in folder_stats], 75),
            'Quantile_90': np.percentile([stats['Quantile_90'] for stats in folder_stats], 90),
            'Quantile_95': np.percentile([stats['Quantile_95'] for stats in folder_stats], 95),
            'Quantile_96': np.percentile([stats['Quantile_96'] for stats in folder_stats], 96),
            'Quantile_97': np.percentile([stats['Quantile_97'] for stats in folder_stats], 97),
            'Quantile_98': np.percentile([stats['Quantile_98'] for stats in folder_stats], 98),
            'Quantile_99': np.percentile([stats['Quantile_99'] for stats in folder_stats], 99)
        }

        # Save statistics to a text file in the TX output folder
        save_statistics_to_file(root, folder_stats, tx_output_folder, total_files, overall_stats)

# Traverse the TY machine directory and collect statistics
for root, dirs, files in os.walk(ty_directory):
    folder_stats = collect_statistics(root)
    total_files = len(files)

    if folder_stats:
        # Calculate overall statistics for the folder
        overall_stats = {
            'Min': np.min([stats['Min'] for stats in folder_stats]),
            'Max': np.max([stats['Max'] for stats in folder_stats]),
            'Mean': np.mean([stats['Mean'] for stats in folder_stats]),
            'Median': np.median([stats['Median'] for stats in folder_stats]),
            'Std': np.mean([stats['Std'] for stats in folder_stats]),
            'Quantile_1': np.percentile([stats['Quantile_1'] for stats in folder_stats], 1),
            'Quantile_2': np.percentile([stats['Quantile_2'] for stats in folder_stats], 2),
            'Quantile_3': np.percentile([stats['Quantile_3'] for stats in folder_stats], 3),
            'Quantile_4': np.percentile([stats['Quantile_4'] for stats in folder_stats], 4),
            'Quantile_5': np.percentile([stats['Quantile_5'] for stats in folder_stats], 5),
            'Quantile_10': np.percentile([stats['Quantile_10'] for stats in folder_stats], 10),
            'Quantile_25': np.percentile([stats['Quantile_25'] for stats in folder_stats], 25),
            'Quantile_50': np.percentile([stats['Quantile_50'] for stats in folder_stats], 50),
            'Quantile_75': np.percentile([stats['Quantile_75'] for stats in folder_stats], 75),
            'Quantile_90': np.percentile([stats['Quantile_90'] for stats in folder_stats], 90),
            'Quantile_95': np.percentile([stats['Quantile_95'] for stats in folder_stats], 95),
            'Quantile_96': np.percentile([stats['Quantile_96'] for stats in folder_stats], 96),
            'Quantile_97': np.percentile([stats['Quantile_97'] for stats in folder_stats], 97),
            'Quantile_98': np.percentile([stats['Quantile_98'] for stats in folder_stats], 98),
            'Quantile_99': np.percentile([stats['Quantile_99'] for stats in folder_stats], 99)
        }
        # Save statistics to a text file in the TY output folder
        save_statistics_to_file(root, folder_stats, ty_output_folder, total_files, overall_stats)
