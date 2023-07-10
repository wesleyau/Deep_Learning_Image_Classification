import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets

plt.style.use('ggplot')

# Set your directories here
base_dir = '/data/wesley/2_data/train_test/TX/'

def count_furnaces_in_folders(base_dir):
    data = []
    for subset in ['train', 'val']:
        subset_path = os.path.join(base_dir, subset)
        for result in ['Pass', 'Fail']:
            result_path = os.path.join(subset_path, result)
            for crystal in os.listdir(result_path):
                if os.path.isdir(os.path.join(result_path, crystal)):
                    furnace = crystal[0]  # Extract the first character
                    data.append({'Furnace': furnace, 'Subset': subset})
    return pd.DataFrame(data)

def plot_bar(df, subset, save_path=None):
    df = df[df['Subset'] == subset] if subset != "All" else df
    max_frequency = df['Furnace'].value_counts().max()  # get max frequency for y-ticks

    title = f'{subset} Number of Instances per Furnace'
    plt.yticks(range(0, max_frequency + 1, 10))  # Set y-ticks every 10 for 'All' option

    df_counts = df['Furnace'].value_counts().reset_index()
    df_counts.columns = ['Furnace', 'Frequency']
    plt.bar(df_counts['Furnace'], df_counts['Frequency'], alpha=0.5, color='#ec6602', edgecolor='black')

    plt.title(title)
    plt.xlabel('Furnace')
    plt.ylabel('Frequency')
    
    if save_path and subset == "All":
        plt.savefig(os.path.join(save_path, "TX_furnaces_histogram.png"))
    
    plt.show()

# Specify your save directory here
save_dir = "/data/wesley/2_data/model_outputs"

df = count_furnaces_in_folders(base_dir)
subset_widget = widgets.Dropdown(options=["All", "train", "val"])
widgets.interact(plot_bar, df=widgets.fixed(df), subset=subset_widget, save_path=widgets.fixed(save_dir))
