import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.neighbors import NearestNeighbors
from typing import Iterator, Tuple, List, Dict, Optional, Union
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

def build_knn_igraph(coordinates: np.ndarray, k: Union[int, str] = 5, include_self: bool = False, 
                    symmetric: bool = True, weight_method: str = "inverse") -> ig.Graph:
    """
    Build a k-nearest neighbors graph from coordinates using igraph
    
    Parameters:
    -----------
    coordinates : numpy.ndarray
        Matrix of shape (n_points, 2) containing X,Y coordinates
    k : int or str, default=5
        Number of nearest neighbors for each point.
        If 'sqrt', k is computed as sqrt(n_points).
    include_self : bool, default=False
        Whether to include self-connections
    symmetric : bool, default=True
        Whether to ensure symmetric connections
    weight_method : str, default="inverse"
        How to weight edges based on distance:
        - "raw": use raw distances (larger = further apart)
        - "inverse": use 1/distance (larger = closer together)
        - "gaussian": use exp(-distance²/mean_distance²) (larger = closer together)
        - "exponential": use exp(-distance/mean_distance) (larger = closer together)
        - "constant": use constant weight of 1.0 for all edges
        
    Returns:
    --------
    igraph.Graph
        An undirected graph connecting k-nearest neighbors
    """
    n_points = coordinates.shape[0]
    if n_points <= 1:
        return ig.Graph()
    
    # Handle 'sqrt' special case for dynamic k calculation
    if k == 'sqrt':
        k = int(np.sqrt(n_points))
        
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
    
    # Transform distances to weights based on selected method
    if weight_method == "inverse":
        # Avoid division by zero (add small epsilon)
        weights = 1.0 / (weights + 1e-10)
    elif weight_method == "gaussian":
        # Gaussian kernel with adaptive sigma
        mean_dist = np.mean(weights)
        weights = np.exp(-(weights**2) / (mean_dist**2))
    elif weight_method == "exponential":
        # Exponential decay with adaptive scale
        mean_dist = np.mean(weights)
        weights = np.exp(-weights / mean_dist)
    elif weight_method == "constant":
        # Constant weights of 1.0
        weights = np.ones_like(weights)
    # "raw" doesn't need transformation
    
    # Pre-allocate edge list array for better performance
    edge_list = np.column_stack((sources, targets))
    
    # Add all edges at once
    G.add_edges(edge_list)
    G.es["weight"] = weights
    
    if symmetric:
        # Make graph undirected (ensures symmetry in kNN relationships)
        G.simplify(combine_edges=dict(weight="min" if weight_method == "raw" else "max"))
    
    return G

def find_communities(graph: ig.Graph, resolution: float = 1.0, 
                    initial_membership: Optional[np.ndarray] = None,
                    **leiden_params) -> np.ndarray:
    """
    Find communities in a graph using the Leiden algorithm.
    
    Parameters:
    -----------
    graph : igraph.Graph
        The graph to find communities in
    resolution : float, default=1.0
        Resolution parameter for Leiden algorithm
    initial_membership : numpy.ndarray, optional
        Initial community assignment to start from
    **leiden_params : dict
        Additional parameters for Leiden algorithm
        
    Returns:
    --------
    numpy.ndarray
        Community membership for each vertex
    """
    # Default parameters
    params = {
        'weights': 'weight',
        'objective_function': 'modularity',
        'resolution': resolution,
        'n_iterations': 10,
        'initial_membership': initial_membership,
    }
    # Update with any user-provided parameters
    params.update(leiden_params)
    
    return np.array(graph.community_leiden(**params).membership)

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

def iter_trajectory_analysis(df: pd.DataFrame, k_mapping: int = 5, 
                          knn_params: Dict = {}, 
                          leiden_params: Dict = {}) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Process cell trajectories in reverse time order, detecting communities and mapping between timesteps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with X, Y, Time columns
    k_mapping : int, default=5
        Number of nearest neighbors for community mapping
    knn_params : dict, default={}
        Dictionary of parameters for KNN graph construction (including 'k' parameter)
    leiden_params : dict, default={}
        Dictionary of parameters for Leiden community detection
        (includes 'resolution' parameter, defaults to 1.0)
        
    Yields:
    -------
    tuple: (time, coordinates, membership) for each timestep
    """
    # Create copies to avoid mutable default argument issues
    knn_params = knn_params.copy()
    leiden_params = leiden_params.copy()
    
    # Create iterator for coordinates in reverse order (latest to earliest)
    coord_iterator = iter_coordinates_by_time(df, reverse=True)
    
    # Default parameters for KNN
    default_knn_params = {
        'k': 10,  # Default k value
        'include_self': False,
        'symmetric': True,
        'weight_method': 'inverse'
    }
    default_knn_params.update(knn_params)
    
    # Default parameters for Leiden community detection
    default_leiden_params = {
        'objective_function': 'modularity',
        'n_iterations': 10,
        'resolution': 1.0  # Default resolution now set here
    }
    default_leiden_params.update(leiden_params)
    
    # Process first (latest) timestep
    try:
        time, coords = next(coord_iterator)
    except StopIteration:
        return
    
    # Build KNN graph and find communities for latest timestep
    graph = build_knn_igraph(coords, **default_knn_params)
    membership = find_communities(graph, **default_leiden_params)
    
    # Yield results for latest timestep
    yield time, coords, membership
    
    # Process remaining timesteps by mapping communities
    prev_coords = coords
    prev_membership = membership
    
    for time, coords in coord_iterator:
        # Map communities from previous (later in time) to current timestep
        initial_membership = map_communities(prev_coords, prev_membership, coords, k=k_mapping)
        
        # Build KNN graph and refine communities for current timestep
        graph = build_knn_igraph(coords, **default_knn_params)
        current_leiden_params = default_leiden_params.copy()
        current_leiden_params['initial_membership'] = initial_membership
        membership = find_communities(graph, **current_leiden_params)
        
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
                   s=0.5, alpha=0.5, label=f"Community {community}")
    
    if title is None:
        title = f"Cell Communities at Time {time}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend if not too many communities
    if n_communities <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

def plot_all_trajectories(df: pd.DataFrame, k_mapping: int = 5, max_plots: Optional[int] = None,
                          knn_params: Dict = {}, leiden_params: Dict = {}, 
                          save: bool = False, output_dir: str = './'):
    """
    Plot the full trajectory analysis with cells colored by community.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with X, Y, Time columns
    k_mapping : int, default=5
        Number of nearest neighbors for community mapping
    max_plots : int, default=None
        Maximum number of timesteps to plot. If None, all timesteps are plotted.
    knn_params : dict, default={}
        Dictionary of parameters for KNN graph construction (including 'k' parameter)
    leiden_params : dict, default={}
        Dictionary of parameters for Leiden community detection
    save : bool, default=False
        If True, save the plot to disk instead of displaying
    output_dir : str, default='./'
        Directory to save the plot when save=True
    """
    # Create copies to avoid mutable default argument issues
    knn_params = knn_params.copy()
    leiden_params = leiden_params.copy()
        
    # Get trajectory analysis iterator
    trajectory_iter = iter_trajectory_analysis(df, k_mapping=k_mapping,
                                             knn_params=knn_params, leiden_params=leiden_params)
    
    # Collect all timesteps for plotting
    all_frames = list(trajectory_iter)
    n_frames = len(all_frames) if max_plots is None else min(len(all_frames), max_plots)
    
    # Determine global min/max for X and Y coordinates across all frames
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for _, coords, _ in all_frames:
        x_min = min(x_min, np.min(coords[:, 0]))
        x_max = max(x_max, np.max(coords[:, 0]))
        y_min = min(y_min, np.min(coords[:, 1]))
        y_max = max(y_max, np.max(coords[:, 1]))
    
    # Add a small margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
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
        # Set consistent axis limits
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
    
    # Hide unused axes
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    # Prepare hyperparameter strings for global title and filename
    # Get default parameters to display full parameter set
    default_knn_params = {
        'k': 10, 
        'include_self': False,
        'symmetric': True,
        'weight_method': 'inverse'
    }
    default_knn_params.update(knn_params)
    
    default_leiden_params = {
        'objective_function': 'modularity',
        'n_iterations': 10,
        'resolution': 1.0
    }
    default_leiden_params.update(leiden_params)
    
    # Format KNN parameters
    knn_str = f"KNN(k={default_knn_params['k']}, weight={default_knn_params['weight_method']}"
    if not default_knn_params['symmetric']:
        knn_str += ", asymmetric"
    if default_knn_params['include_self']:
        knn_str += ", self-loops"
    knn_str += ")"
    
    # Format Leiden parameters
    leiden_str = f"Leiden(res={default_leiden_params['resolution']}, " \
                 f"obj={default_leiden_params['objective_function']}, " \
                 f"iter={default_leiden_params['n_iterations']})"
    
    # Add mapping parameter
    mapping_str = f"Mapping(k={k_mapping})"
    
    # Set global title with hyperparameters
    fig.suptitle(f"Cell Communities Analysis\n{knn_str} - {leiden_str} - {mapping_str}", 
                fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save:
        # Create filename with parameters
        filename_params = [
            f"knn_k{default_knn_params['k']}",
            f"w{default_knn_params['weight_method']}",
            f"leiden_res{default_leiden_params['resolution']}",
            f"obj{default_leiden_params['objective_function'][:3]}",
            f"map_k{k_mapping}"
        ]
        if not default_knn_params['symmetric']:
            filename_params.append("asym")
        if default_knn_params['include_self']:
            filename_params.append("self")
        
        filename = f"cell_communities_{'_'.join(filename_params)}.png"
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {filepath}")
        plt.close(fig)
    else:
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
    
    # Run and plot trajectory analysis with custom parameters
    knn_params = {'k': 10, 'weight_method': 'inverse'}
    leiden_params = {'resolution': 1.0}
    plot_all_trajectories(df, k_mapping=5, knn_params=knn_params, leiden_params=leiden_params)
    
    # Example of accessing the iterator directly
    print("\nProcessing timesteps individually:")
    leiden_params = {'resolution': 1.0}
    for i, (time, coords, membership) in enumerate(
            iter_trajectory_analysis(df, k_mapping=5, knn_params={'k': 10}, leiden_params=leiden_params)):
        n_communities = len(np.unique(membership))
        print(f"Time {time}: {coords.shape[0]} cells in {n_communities} communities")
        
        # Break after a few iterations for demonstration
        if i >= 2:
            print("...")
            break

if __name__ == "__main__":
    # Replace with your actual data path
    main("df_all_new.csv")