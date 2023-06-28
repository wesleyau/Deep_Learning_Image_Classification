import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets

plt.style.use('ggplot')

# Set your directory here
base_dir = '/data/wesley/stats_vmin_vmax/labels_/TX'

def count_furnaces_in_folders(base_dir):
    data = []
    for folder_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, folder_name)):
            furnace = folder_name[0]  # Extract the first character
            data.append({'Furnace': furnace})
    return pd.DataFrame(data)

def plot_bar(df, furnace):
    max_frequency = df['Furnace'].value_counts().max()  # get max frequency for y-ticks
    if furnace == "All":
        data_to_plot = df
        title = 'TX Number of Instances per Furnace'
        plt.yticks(range(0, max_frequency + 1, 10))  # Set y-ticks every 10 for 'All' option
    else:
        data_to_plot = df[df['Furnace'] == furnace]
        title = 'TX Number of Instances for Furnace ' + furnace

    df_counts = data_to_plot['Furnace'].value_counts().reset_index()
    df_counts.columns = ['Furnace', 'Frequency']
    plt.bar(df_counts['Furnace'], df_counts['Frequency'], alpha=0.5, color='#ec6602', edgecolor='black')

    plt.title(title)
    plt.xlabel('Furnace')
    plt.ylabel('Frequency')
    plt.show()

df = count_furnaces_in_folders(base_dir)
furnace_widget = widgets.Dropdown(options=["All", "1", "2", "3", "4"])
widgets.interact(plot_bar, df=widgets.fixed(df), furnace=furnace_widget)
