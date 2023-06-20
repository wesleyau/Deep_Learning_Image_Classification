import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets

plt.style.use('ggplot')

# Set your directory here
base_dir = '/data/wesley/stats_vmin_vmax/labels_/TX'

def count_items_in_folders(base_dir):
    data = []
    for folder_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, folder_name)):
            count = len(os.listdir(os.path.join(base_dir, folder_name)))
            year = folder_name[-2:]  # Extract the last two characters
            data.append({'Year': year, 'Count': count})
    return pd.DataFrame(data)

def plot_histogram(df, year):
    max_slices = df['Count'].max()
    max_frequency = df['Count'].value_counts().max()  # get max frequency for y-ticks
    if year == "All":
        data_to_plot = df
        title = 'TX Number of Slices per Ingot'
        plt.yticks(range(0, max_frequency + 1, 10))  # Set y-ticks every 10 for 'All' option
    else:
        data_to_plot = df[df['Year'] == year]
        title = 'TX Number of Slices per Ingot for year ' + year
    plt.hist(data_to_plot['Count'], bins=max_slices, alpha=0.5, color='#ec6602', edgecolor='black')
    plt.title(title)
    plt.xlabel('Number of slices')
    plt.ylabel('Frequency')
    plt.xticks(range(0, max_slices + 1, 1))  # Set x-ticks for every number
    plt.show()

df = count_items_in_folders(base_dir)
year_widget = widgets.Dropdown(options=["All"] + sorted(df['Year'].unique().tolist()))
widgets.interact(plot_histogram, df=widgets.fixed(df), year=year_widget)
