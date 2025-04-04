import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.neighbors import NearestNeighbors
from typing import Iterator, Tuple, List, Dict, Optional
import seaborn as sns

def iter_coordinates_by_time(df: pd.DataFrame, reverse: bool = False) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Create an iterator that yields timestamp and coordinate matrices for each timestamp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with X, Y, and Time columns
    reverse : bool, default=False
        If True, yields timesteps from latest to earliest
    
    Yields:
    -------
    tuple: (time, coordinates) where coordinates is a matrix of X,Y points
    """
    required_cols = ['X', 'Y', 'Time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns")
    
    # Sort the unique timestamps
    sorted_times = sorted(df['Time'].unique(), reverse=reverse)
    
    # Group by time and yield matrices in sorted order
    for time in sorted_times:
        group = df[df['Time'] == time]
        coordinates = group[['X', 'Y']].values
        yield time, coordinates

def build_knn_igraph(coordinates: np.ndarray, k: int = 5, include_self: bool = False, 
                    symmetric: bool = True) -> ig.Graph:
    """
    Build a k-nearest neighbors graph from coordinates using igraph
    
    Parameters:
    -----------
    coordinates : numpy.ndarray
        Matrix of shape (n_points, 2) containing X,Y coordinates
    k : int, default=5
        Number of nearest neighbors for each point
    include_self : bool, default=False
        Whether to include self-connections
    symmetric : bool, default=True
        Whether to ensure symmetric connections
        
    Returns:
    --------
    igraph.Graph
        An undirected graph connecting k-nearest neighbors
    """
    n_points = coordinates.shape[0]
    if n_points <= 1:
        return ig.Graph()
        
    # Adjust k based on parameters
    actual_k = min(k + (0 if include_self else 1), n_points)
    
    # Use efficient ball_tree algorithm for nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm='ball_tree', n_jobs=-1).fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    # Create graph with pre-allocated vertices
    G = ig.Graph(n=n_points)
    
    # Add vertex coordinates in bulk
    G.vs["x"] = coordinates[:, 0]
    G.vs["y"] = coordinates[:, 1]
    
    # Use numpy for efficient edge list creation
    start_idx = 1 if not include_self else 0
    rows, cols = np.nonzero(np.ones((n_points, actual_k - start_idx)))
    
    sources = rows
    targets = indices[rows, cols + start_idx]
    weights = distances[rows, cols + start_idx]
    
    # Pre-allocate edge list array for better performance
    edge_list = np.column_stack((sources, targets))
    
    # Add all edges at once
    G.add_edges(edge_list)
    G.es["weight"] = weights
    
    if symmetric:
        # Make graph undirected (ensures symmetry in kNN relationships)
        G.simplify(combine_edges=dict(weight="min"))
    
    return G

def find_communities(graph: ig.Graph, resolution: float = 1.0) -> np.ndarray:
    """
    Find communities in a graph using the Leiden algorithm.
    
    Parameters:
    -----------
    graph : igraph.Graph
        The graph to find communities in
    resolution : float, default=1.0
        Resolution parameter for Leiden algorithm
        
    Returns:
    --------
    numpy.ndarray
        Community membership for each vertex
    """
    return np.array(graph.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution_parameter=resolution,
        n_iterations=10,
    ).membership)

def map_communities(current_coords: np.ndarray, current_membership: np.ndarray,
                   next_coords: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Map community memberships from current timestep to next timestep using KNN.
    
    Parameters:
    -----------
    current_coords : numpy.ndarray
        Coordinates from current timestep
    current_membership : numpy.ndarray
        Community memberships for current timestep
    next_coords : numpy.ndarray
        Coordinates from next timestep
    k : int, default=5
        Number of neighbors for voting
        
    Returns:
    --------
    numpy.ndarray
        Predicted community memberships for next timestep
    """
    # Find k nearest neighbors from current timestep for each point in next timestep
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(current_coords)
    _, indices = nbrs.kneighbors(next_coords)
    
    # For each point in next_coords, get the community labels of its k nearest neighbors
    neighbor_memberships = current_membership[indices]
    
    # Assign community based on majority vote
    next_membership = np.array([
        np.bincount(neighbor_memberships[i]).argmax()
        for i in range(len(next_coords))
    ])
    
    return next_membership

def iter_trajectory_analysis(df: pd.DataFrame, k_graph: int = 10, k_mapping: int = 5, 
                          resolution: float = 1.0) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Process cell trajectories in reverse time order, detecting communities and mapping between timesteps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with X, Y, Time columns
    k_graph : int, default=10
        Number of nearest neighbors for graph construction
    k_mapping : int, default=5
        Number of nearest neighbors for community mapping
    resolution : float, default=1.0
        Resolution parameter for Leiden algorithm
        
    Yields:
    -------
    tuple: (time, coordinates, membership) for each timestep
    """
    # Create iterator for coordinates in reverse order (latest to earliest)
    coord_iterator = iter_coordinates_by_time(df, reverse=True)
    
    # Process first (latest) timestep
    try:
        time, coords = next(coord_iterator)
    except StopIteration:
        return
    
    # Build KNN graph and find communities for latest timestep
    graph = build_knn_igraph(coords, k=k_graph)
    membership = find_communities(graph, resolution=resolution)
    
    # Yield results for latest timestep
    yield time, coords, membership
    
    # Process remaining timesteps by mapping communities
    prev_coords = coords
    prev_membership = membership
    
    for time, coords in coord_iterator:
        # Map communities from previous (later in time) to current timestep
        membership = map_communities(prev_coords, prev_membership, coords, k=k_mapping)
        
        # Build KNN graph and refine communities for current timestep
        graph = build_knn_igraph(coords, k=k_graph)
        membership = find_communities(graph, resolution=resolution)
        
        # Update for next iteration
        prev_coords = coords
        prev_membership = membership
        
        # Yield results for this timestep
        yield time, coords, membership

def plot_trajectory_frame(time: float, coords: np.ndarray, membership: np.ndarray, 
                        ax=None, title: Optional[str] = None):
    """
    Plot a single frame of the trajectory with points colored by membership.
    
    Parameters:
    -----------
    time : float
        Timestamp for this frame
    coords : numpy.ndarray
        Coordinate matrix (n_points, 2)
    membership : numpy.ndarray
        Community membership for each point
    ax : matplotlib.axes, default=None
        Axes to plot on (creates new figure if None)
    title : str, optional
        Custom title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique communities and assign colors
    unique_communities = np.unique(membership)
    n_communities = len(unique_communities)
    
    # Use a qualitative colormap for distinct communities
    if n_communities <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_communities))
    
    # Plot each community with a different color
    for i, community in enumerate(unique_communities):
        mask = membership == community
        ax.scatter(coords[mask, 0], coords[mask, 1], color=colors[i % len(colors)], 
                   s=3, alpha=0.5, label=f"Community {community}")
    
    if title is None:
        title = f"Cell Communities at Time {time}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend if not too many communities
    if n_communities <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

def plot_all_trajectories(df: pd.DataFrame, k_graph: int = 10, k_mapping: int = 5, 
                         resolution: float = 1.0, max_plots: int = 9):
    """
    Plot the full trajectory analysis with cells colored by community.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with X, Y, Time columns
    k_graph : int, default=10
        Number of nearest neighbors for graph construction
    k_mapping : int, default=5
        Number of nearest neighbors for community mapping
    resolution : float, default=1.0
        Resolution parameter for Leiden algorithm
    max_plots : int, default=9
        Maximum number of timesteps to plot
    """
    # Get trajectory analysis iterator
    trajectory_iter = iter_trajectory_analysis(df, k_graph, k_mapping, resolution)
    
    # Collect all timesteps for plotting
    all_frames = list(trajectory_iter)
    n_frames = min(len(all_frames), max_plots)
    
    # Create grid of plots
    rows = int(np.ceil(np.sqrt(n_frames)))
    cols = int(np.ceil(n_frames / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 5*rows))
    if n_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each timestep
    for i in range(n_frames):
        time, coords, membership = all_frames[i]
        plot_trajectory_frame(time, coords, membership, axes[i])
    
    # Hide unused axes
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main(data_path: str):
    """
    Main function to run the full analysis.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with X, Y, Time columns
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} points across {df['Time'].nunique()} timesteps")
    
    # Run and plot trajectory analysis
    plot_all_trajectories(df, k_graph=10, k_mapping=5, resolution=1.0)
    
    # Example of accessing the iterator directly
    print("\nProcessing timesteps individually:")
    for i, (time, coords, membership) in enumerate(
            iter_trajectory_analysis(df, k_graph=10, k_mapping=5)):
        n_communities = len(np.unique(membership))
        print(f"Time {time}: {coords.shape[0]} cells in {n_communities} communities")
        
        # Break after a few iterations for demonstration
        if i >= 2:
            print("...")
            break

if __name__ == "__main__":
    # Replace with your actual data path
    main("df_all_new.csv")