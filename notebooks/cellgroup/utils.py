import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np
import pandas as pd

def plot_points(points, ordered_list=None, clusters=None, title=''):
    """
    Plot points and clusters in 2D space.

    Args:
    points (np.array): Array of points to plot.
    ordered_list (np.array, optional): Ordered list of points (used for OPTICS).
    clusters (list, optional): List of cluster indices.
    title (str): Title of the plot.
    """
    # Plot all points
    plt.scatter(points[:,0], points[:,1], c='k', linewidth=0.2, edgecolor='w', facecolor=None)

    # Plot clusters, if any
    if clusters:
        # Generate a list of colors for each cluster and randomize their order
        colors = cm.inferno(np.linspace(0.3, 1, len(clusters)))
        color_order = random.sample(range(len(colors)), len(colors))

        # Plot each cluster with a different color
        for color, cluster in zip(colors[color_order], clusters):
            if ordered_list is not None:
                cluster_points = ordered_list[cluster][:, 3:5]
            else:
                cluster_points = points[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, linewidth=0.2, edgecolor='w')

    plt.title(title)
    plt.gca().grid(color='0.5')
    plt.gca().set_facecolor('black')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_csv(csv_file):
    """
    Plot data from a CSV file.
    
    Args:
    csv_file (str): Path to the CSV file.
    
    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    """
    results = pd.read_csv(csv_file)
    fig, ax = plt.subplots()
    for label, group in results.groupby('Labels'):
        ax.scatter(group['X'], group['Y'], label=label)
    plt.show()
    return fig

def plotPointsFromCSV(data, title=''):
    """
    Plot points from a DataFrame, coloring by cluster labels.
    
    Args:
    data (pd.DataFrame): DataFrame containing 'X', 'Y', and 'Labels' columns.
    title (str): Title of the plot.
    """
    points = data[['X', 'Y']].to_numpy()
    labels = data['Labels']
    unique_labels = labels.unique()
    clusters = [points[labels == label] for label in unique_labels]

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c='k', linewidth=0.2, edgecolor='w', facecolor=None)

    colors = cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for color, cluster in zip(colors, clusters):
        ax.scatter(cluster[:, 0], cluster[:, 1], c=color, linewidth=0.2, edgecolor='w')

    ax.set_title(title)
    ax.set_xlim(0, 6500)
    ax.set_ylim(0, 6500)
    ax.grid(color='0.5')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def save_plot_and_show(plot_function, filename, *args, **kwargs):
    """
    Create a plot, save it to a file, and display it.
    
    Args:
    plot_function (function): Function to create the plot.
    filename (str): Name of the file to save the plot.
    *args, **kwargs: Additional arguments for the plot function.
    """
    fig = plot_function(*args, **kwargs)
    fig.savefig(filename)

def saveResults(df, ordered_list, time, clusters=[], filename='results.csv'):
    # Create an empty dataframe to store the results
    results = pd.DataFrame(df)

    # Plot clusters, if any
    if clusters:
        # Generate a list of colors for each cluster and randomize their order
        colors = cm.inferno(np.linspace(0.3, 1, len(clusters)))
        color_order = random.sample(range(len(colors)), len(colors))

        # Create a tuple to map each unique color to a unique integer ID
        tuple_colors = [tuple(color) for color in colors]
        unique_colors = list(set(tuple_colors))
        color_to_id = {color: i for i, color in enumerate(unique_colors)}

        results['Labels'] = np.nan
        results['Time'] = np.nan

        # Store the data in the results dataframe
        for color, cluster in zip(colors[color_order], clusters):
            if len(ordered_list) > 0:
                x = ordered_list[cluster][:,3]
                y = ordered_list[cluster][:,4]
            else:
                x = df.iloc[cluster, 0]  # Assuming X is the first column
                y = df.iloc[cluster, 1]  # Assuming Y is the second column

            # Replace column X and Y with x and y
            results.loc[cluster, 'X'] = x
            results.loc[cluster, 'Y'] = y
            # Create the Labels column and store labels in the dataframe df
            results.loc[cluster, 'Labels'] = color_to_id[tuple(color)]
            results.loc[cluster, 'Time'] = time

    # Save the results to a CSV file
    results = results.dropna(subset=['Labels'])
    results.to_csv(filename, index=False)
    print(f"Successfully saved the results to {filename}")
    return results