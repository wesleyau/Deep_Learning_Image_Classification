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
                    'Quantile_6': np.percentile(data, 6),
                    'Quantile_7': np.percentile(data, 7),
                    'Quantile_8': np.percentile(data, 8),
                    'Quantile_9': np.percentile(data, 9),
                    'Quantile_10': np.percentile(data, 10),
                    'Quantile_11': np.percentile(data, 11),
                    'Quantile_12': np.percentile(data, 12),
                    'Quantile_13': np.percentile(data, 13),
                    'Quantile_14': np.percentile(data, 14),
                    'Quantile_15': np.percentile(data, 15),
                    'Quantile_16': np.percentile(data, 16),
                    'Quantile_17': np.percentile(data, 17),
                    'Quantile_18': np.percentile(data, 18),
                    'Quantile_19': np.percentile(data, 19),
                    'Quantile_20': np.percentile(data, 20),
                    'Quantile_21': np.percentile(data, 21),
                    'Quantile_22': np.percentile(data, 22),
                    'Quantile_23': np.percentile(data, 23),
                    'Quantile_24': np.percentile(data, 24),
                    'Quantile_25': np.percentile(data, 25),
                    'Quantile_50': np.percentile(data, 50),
                    'Quantile_75': np.percentile(data, 75),
                    'Quantile_76': np.percentile(data, 76),
                    'Quantile_77': np.percentile(data, 77),
                    'Quantile_78': np.percentile(data, 78),
                    'Quantile_79': np.percentile(data, 79),
                    'Quantile_80': np.percentile(data, 80),
                    'Quantile_81': np.percentile(data, 81),
                    'Quantile_82': np.percentile(data, 82),
                    'Quantile_83': np.percentile(data, 83),
                    'Quantile_84': np.percentile(data, 84),
                    'Quantile_85': np.percentile(data, 85),
                    'Quantile_86': np.percentile(data, 86),
                    'Quantile_87': np.percentile(data, 87),
                    'Quantile_88': np.percentile(data, 88),
                    'Quantile_89': np.percentile(data, 89),
                    'Quantile_90': np.percentile(data, 90),
                    'Quantile_91': np.percentile(data, 91),
                    'Quantile_92': np.percentile(data, 92),
                    'Quantile_93': np.percentile(data, 93),
                    'Quantile_94': np.percentile(data, 94),
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
        file.write('Quantile 6: ' + str(overall_stats['Quantile_6']) + '\n')
        file.write('Quantile 7: ' + str(overall_stats['Quantile_7']) + '\n')
        file.write('Quantile 8: ' + str(overall_stats['Quantile_8']) + '\n')
        file.write('Quantile 9: ' + str(overall_stats['Quantile_9']) + '\n')
        file.write('Quantile 10: ' + str(overall_stats['Quantile_10']) + '\n')
        file.write('Quantile 11: ' + str(overall_stats['Quantile_11']) + '\n')
        file.write('Quantile 12: ' + str(overall_stats['Quantile_12']) + '\n')
        file.write('Quantile 13: ' + str(overall_stats['Quantile_13']) + '\n')
        file.write('Quantile 14: ' + str(overall_stats['Quantile_14']) + '\n')
        file.write('Quantile 15: ' + str(overall_stats['Quantile_15']) + '\n')
        file.write('Quantile 16: ' + str(overall_stats['Quantile_16']) + '\n')
        file.write('Quantile 17: ' + str(overall_stats['Quantile_17']) + '\n')
        file.write('Quantile 18: ' + str(overall_stats['Quantile_18']) + '\n')
        file.write('Quantile 19: ' + str(overall_stats['Quantile_19']) + '\n')
        file.write('Quantile 20: ' + str(overall_stats['Quantile_20']) + '\n')
        file.write('Quantile 21: ' + str(overall_stats['Quantile_21']) + '\n')
        file.write('Quantile 22: ' + str(overall_stats['Quantile_22']) + '\n')
        file.write('Quantile 23: ' + str(overall_stats['Quantile_23']) + '\n')
        file.write('Quantile 24: ' + str(overall_stats['Quantile_24']) + '\n')
        file.write('Quantile 25: ' + str(overall_stats['Quantile_25']) + '\n')
        file.write('Quantile 50: ' + str(overall_stats['Quantile_50']) + '\n')
        file.write('Quantile 75: ' + str(overall_stats['Quantile_75']) + '\n')
        file.write('Quantile 76: ' + str(overall_stats['Quantile_76']) + '\n')
        file.write('Quantile 77: ' + str(overall_stats['Quantile_77']) + '\n')
        file.write('Quantile 78: ' + str(overall_stats['Quantile_78']) + '\n')
        file.write('Quantile 79: ' + str(overall_stats['Quantile_79']) + '\n')
        file.write('Quantile 80: ' + str(overall_stats['Quantile_80']) + '\n')
        file.write('Quantile 81: ' + str(overall_stats['Quantile_81']) + '\n')
        file.write('Quantile 82: ' + str(overall_stats['Quantile_82']) + '\n')
        file.write('Quantile 83: ' + str(overall_stats['Quantile_83']) + '\n')
        file.write('Quantile 84: ' + str(overall_stats['Quantile_84']) + '\n')
        file.write('Quantile 85: ' + str(overall_stats['Quantile_85']) + '\n')
        file.write('Quantile 86: ' + str(overall_stats['Quantile_86']) + '\n')
        file.write('Quantile 87: ' + str(overall_stats['Quantile_87']) + '\n')
        file.write('Quantile 88: ' + str(overall_stats['Quantile_88']) + '\n')
        file.write('Quantile 89: ' + str(overall_stats['Quantile_89']) + '\n')
        file.write('Quantile 90: ' + str(overall_stats['Quantile_90']) + '\n')
        file.write('Quantile 91: ' + str(overall_stats['Quantile_91']) + '\n')
        file.write('Quantile 92: ' + str(overall_stats['Quantile_92']) + '\n')
        file.write('Quantile 93: ' + str(overall_stats['Quantile_93']) + '\n')
        file.write('Quantile 94: ' + str(overall_stats['Quantile_94']) + '\n')
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
            file.write('Quantile 6: ' + str(stats['Quantile_6']) + '\n')
            file.write('Quantile 7: ' + str(stats['Quantile_7']) + '\n')
            file.write('Quantile 8: ' + str(stats['Quantile_8']) + '\n')
            file.write('Quantile 9: ' + str(stats['Quantile_9']) + '\n')
            file.write('Quantile 10: ' + str(stats['Quantile_10']) + '\n')
            file.write('Quantile 11: ' + str(stats['Quantile_11']) + '\n')
            file.write('Quantile 12: ' + str(stats['Quantile_12']) + '\n')
            file.write('Quantile 13: ' + str(stats['Quantile_13']) + '\n')
            file.write('Quantile 14: ' + str(stats['Quantile_14']) + '\n')
            file.write('Quantile 15: ' + str(stats['Quantile_15']) + '\n')
            file.write('Quantile 16: ' + str(stats['Quantile_16']) + '\n')
            file.write('Quantile 17: ' + str(stats['Quantile_17']) + '\n')
            file.write('Quantile 18: ' + str(stats['Quantile_18']) + '\n')
            file.write('Quantile 19: ' + str(stats['Quantile_19']) + '\n')
            file.write('Quantile 20: ' + str(stats['Quantile_20']) + '\n')
            file.write('Quantile 21: ' + str(stats['Quantile_21']) + '\n')
            file.write('Quantile 22: ' + str(stats['Quantile_22']) + '\n')
            file.write('Quantile 23: ' + str(stats['Quantile_23']) + '\n')
            file.write('Quantile 24: ' + str(stats['Quantile_24']) + '\n')
            file.write('Quantile 25: ' + str(stats['Quantile_25']) + '\n')
            file.write('Quantile 50: ' + str(stats['Quantile_50']) + '\n')
            file.write('Quantile 75: ' + str(stats['Quantile_75']) + '\n')
            file.write('Quantile 76: ' + str(stats['Quantile_76']) + '\n')
            file.write('Quantile 77: ' + str(stats['Quantile_77']) + '\n')
            file.write('Quantile 78: ' + str(stats['Quantile_78']) + '\n')
            file.write('Quantile 79: ' + str(stats['Quantile_79']) + '\n')
            file.write('Quantile 80: ' + str(stats['Quantile_80']) + '\n')
            file.write('Quantile 81: ' + str(stats['Quantile_81']) + '\n')
            file.write('Quantile 82: ' + str(stats['Quantile_82']) + '\n')
            file.write('Quantile 83: ' + str(stats['Quantile_83']) + '\n')
            file.write('Quantile 84: ' + str(stats['Quantile_84']) + '\n')
            file.write('Quantile 85: ' + str(stats['Quantile_85']) + '\n')
            file.write('Quantile 86: ' + str(stats['Quantile_86']) + '\n')
            file.write('Quantile 87: ' + str(stats['Quantile_87']) + '\n')
            file.write('Quantile 88: ' + str(stats['Quantile_88']) + '\n')
            file.write('Quantile 89: ' + str(stats['Quantile_89']) + '\n')
            file.write('Quantile 90: ' + str(stats['Quantile_90']) + '\n')
            file.write('Quantile 91: ' + str(stats['Quantile_91']) + '\n')
            file.write('Quantile 92: ' + str(stats['Quantile_92']) + '\n')
            file.write('Quantile 93: ' + str(stats['Quantile_93']) + '\n')
            file.write('Quantile 94: ' + str(stats['Quantile_94']) + '\n')
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
            'Quantile_6': np.percentile([stats['Quantile_6'] for stats in folder_stats], 6),
            'Quantile_7': np.percentile([stats['Quantile_7'] for stats in folder_stats], 7),
            'Quantile_8': np.percentile([stats['Quantile_8'] for stats in folder_stats], 8),
            'Quantile_9': np.percentile([stats['Quantile_9'] for stats in folder_stats], 9),
            'Quantile_10': np.percentile([stats['Quantile_10'] for stats in folder_stats], 10),
            'Quantile_11': np.percentile([stats['Quantile_11'] for stats in folder_stats], 11),
            'Quantile_12': np.percentile([stats['Quantile_12'] for stats in folder_stats], 12),
            'Quantile_13': np.percentile([stats['Quantile_13'] for stats in folder_stats], 13),
            'Quantile_14': np.percentile([stats['Quantile_14'] for stats in folder_stats], 14),
            'Quantile_15': np.percentile([stats['Quantile_15'] for stats in folder_stats], 15),
            'Quantile_16': np.percentile([stats['Quantile_16'] for stats in folder_stats], 16),
            'Quantile_17': np.percentile([stats['Quantile_17'] for stats in folder_stats], 17),
            'Quantile_18': np.percentile([stats['Quantile_18'] for stats in folder_stats], 18),
            'Quantile_19': np.percentile([stats['Quantile_19'] for stats in folder_stats], 19),
            'Quantile_20': np.percentile([stats['Quantile_20'] for stats in folder_stats], 20),
            'Quantile_21': np.percentile([stats['Quantile_21'] for stats in folder_stats], 21),
            'Quantile_22': np.percentile([stats['Quantile_22'] for stats in folder_stats], 22),
            'Quantile_23': np.percentile([stats['Quantile_23'] for stats in folder_stats], 23),
            'Quantile_24': np.percentile([stats['Quantile_24'] for stats in folder_stats], 24),
            'Quantile_25': np.percentile([stats['Quantile_25'] for stats in folder_stats], 25),
            'Quantile_50': np.percentile([stats['Quantile_50'] for stats in folder_stats], 50),
            'Quantile_75': np.percentile([stats['Quantile_75'] for stats in folder_stats], 75),
            'Quantile_76': np.percentile([stats['Quantile_76'] for stats in folder_stats], 76),
            'Quantile_77': np.percentile([stats['Quantile_77'] for stats in folder_stats], 77),
            'Quantile_78': np.percentile([stats['Quantile_78'] for stats in folder_stats], 78),
            'Quantile_79': np.percentile([stats['Quantile_79'] for stats in folder_stats], 79),
            'Quantile_80': np.percentile([stats['Quantile_80'] for stats in folder_stats], 80),
            'Quantile_81': np.percentile([stats['Quantile_81'] for stats in folder_stats], 81),
            'Quantile_82': np.percentile([stats['Quantile_82'] for stats in folder_stats], 82),
            'Quantile_83': np.percentile([stats['Quantile_83'] for stats in folder_stats], 83),
            'Quantile_84': np.percentile([stats['Quantile_84'] for stats in folder_stats], 84),
            'Quantile_85': np.percentile([stats['Quantile_85'] for stats in folder_stats], 85),
            'Quantile_86': np.percentile([stats['Quantile_86'] for stats in folder_stats], 86),
            'Quantile_87': np.percentile([stats['Quantile_87'] for stats in folder_stats], 87),
            'Quantile_88': np.percentile([stats['Quantile_88'] for stats in folder_stats], 88),
            'Quantile_89': np.percentile([stats['Quantile_89'] for stats in folder_stats], 89),
            'Quantile_90': np.percentile([stats['Quantile_90'] for stats in folder_stats], 90),
            'Quantile_91': np.percentile([stats['Quantile_91'] for stats in folder_stats], 91),
            'Quantile_92': np.percentile([stats['Quantile_92'] for stats in folder_stats], 92),
            'Quantile_93': np.percentile([stats['Quantile_93'] for stats in folder_stats], 93),
            'Quantile_94': np.percentile([stats['Quantile_94'] for stats in folder_stats], 94),
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
            'Quantile_6': np.percentile([stats['Quantile_6'] for stats in folder_stats], 6),
            'Quantile_7': np.percentile([stats['Quantile_7'] for stats in folder_stats], 7),
            'Quantile_8': np.percentile([stats['Quantile_8'] for stats in folder_stats], 8),
            'Quantile_9': np.percentile([stats['Quantile_9'] for stats in folder_stats], 9),
            'Quantile_10': np.percentile([stats['Quantile_10'] for stats in folder_stats], 10),
            'Quantile_11': np.percentile([stats['Quantile_11'] for stats in folder_stats], 11),
            'Quantile_12': np.percentile([stats['Quantile_12'] for stats in folder_stats], 12),
            'Quantile_13': np.percentile([stats['Quantile_13'] for stats in folder_stats], 13),
            'Quantile_14': np.percentile([stats['Quantile_14'] for stats in folder_stats], 14),
            'Quantile_15': np.percentile([stats['Quantile_15'] for stats in folder_stats], 15),
            'Quantile_16': np.percentile([stats['Quantile_16'] for stats in folder_stats], 16),
            'Quantile_17': np.percentile([stats['Quantile_17'] for stats in folder_stats], 17),
            'Quantile_18': np.percentile([stats['Quantile_18'] for stats in folder_stats], 18),
            'Quantile_19': np.percentile([stats['Quantile_19'] for stats in folder_stats], 19),
            'Quantile_20': np.percentile([stats['Quantile_20'] for stats in folder_stats], 20),
            'Quantile_21': np.percentile([stats['Quantile_21'] for stats in folder_stats], 21),
            'Quantile_22': np.percentile([stats['Quantile_22'] for stats in folder_stats], 22),
            'Quantile_23': np.percentile([stats['Quantile_23'] for stats in folder_stats], 23),
            'Quantile_24': np.percentile([stats['Quantile_24'] for stats in folder_stats], 24),
            'Quantile_25': np.percentile([stats['Quantile_25'] for stats in folder_stats], 25),
            'Quantile_50': np.percentile([stats['Quantile_50'] for stats in folder_stats], 50),
            'Quantile_75': np.percentile([stats['Quantile_75'] for stats in folder_stats], 75),
            'Quantile_76': np.percentile([stats['Quantile_76'] for stats in folder_stats], 76),
            'Quantile_77': np.percentile([stats['Quantile_77'] for stats in folder_stats], 77),
            'Quantile_78': np.percentile([stats['Quantile_78'] for stats in folder_stats], 78),
            'Quantile_79': np.percentile([stats['Quantile_79'] for stats in folder_stats], 79),
            'Quantile_80': np.percentile([stats['Quantile_80'] for stats in folder_stats], 80),
            'Quantile_81': np.percentile([stats['Quantile_81'] for stats in folder_stats], 81),
            'Quantile_82': np.percentile([stats['Quantile_82'] for stats in folder_stats], 82),
            'Quantile_83': np.percentile([stats['Quantile_83'] for stats in folder_stats], 83),
            'Quantile_84': np.percentile([stats['Quantile_84'] for stats in folder_stats], 84),
            'Quantile_85': np.percentile([stats['Quantile_85'] for stats in folder_stats], 85),
            'Quantile_86': np.percentile([stats['Quantile_86'] for stats in folder_stats], 86),
            'Quantile_87': np.percentile([stats['Quantile_87'] for stats in folder_stats], 87),
            'Quantile_88': np.percentile([stats['Quantile_88'] for stats in folder_stats], 88),
            'Quantile_89': np.percentile([stats['Quantile_89'] for stats in folder_stats], 89),
            'Quantile_90': np.percentile([stats['Quantile_90'] for stats in folder_stats], 90),
            'Quantile_91': np.percentile([stats['Quantile_91'] for stats in folder_stats], 91),
            'Quantile_92': np.percentile([stats['Quantile_92'] for stats in folder_stats], 92),
            'Quantile_93': np.percentile([stats['Quantile_93'] for stats in folder_stats], 93),
            'Quantile_94': np.percentile([stats['Quantile_94'] for stats in folder_stats], 94),
            'Quantile_95': np.percentile([stats['Quantile_95'] for stats in folder_stats], 95),
            'Quantile_96': np.percentile([stats['Quantile_96'] for stats in folder_stats], 96),
            'Quantile_97': np.percentile([stats['Quantile_97'] for stats in folder_stats], 97),
            'Quantile_98': np.percentile([stats['Quantile_98'] for stats in folder_stats], 98),
            'Quantile_99': np.percentile([stats['Quantile_99'] for stats in folder_stats], 99)
        }
        # Save statistics to a text file in the TY output folder
        save_statistics_to_file(root, folder_stats, ty_output_folder, total_files, overall_stats)
