import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm
from utils import plot_points, saveResults
from cyoptics_clustering.GradientClustering import gradientClustering, plotClusteringReachability, filterLargeClusters, mergeSimilarClusters
from typing import Optional

def run_OPTICS(input_list: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    import pyximport
    pyximport.install(setup_args={'include_dirs': [np.get_include()]})
    from cyoptics_clustering.cyOPTICS import runCyOPTICS
    return runCyOPTICS(input_list, eps, min_pts)

def run_optics_clustering(input_data, min_points: int, epsilon: float, w: float, 
                          max_points_ratio: float, cluster_similarity_threshold: float, 
                          t: int = 150, verbose: int = 1):
    input_data = input_data.values if hasattr(input_data, 'values') else input_data
    if verbose >= 2:
        plot_points(input_data, title='Input data')

    t1 = time.perf_counter()
    ordered_list = run_OPTICS(input_data, epsilon, min_points)
    
    if verbose >= 1:
        print(f'Total time for processing: {time.perf_counter() - t1:.2f} s')
    
    if verbose >= 3:
        plotClusteringReachability(ordered_list[:,1])

    clusters = gradientClustering(ordered_list[:,1], min_points, t, w)
    filtered_clusters = filterLargeClusters(clusters, len(ordered_list), max_points_ratio)
    
    if verbose >= 3:
        plotClusteringReachability(ordered_list[:,1], filtered_clusters)

    filtered_clusters = mergeSimilarClusters(filtered_clusters, cluster_similarity_threshold)
    
    if verbose >= 3:
        plotClusteringReachability(ordered_list[:,1], filtered_clusters)
    
    if verbose >= 2:
        plot_points(input_data, ordered_list, filtered_clusters, title='OPTICS Clustering Results')

    results = saveResults(pd.DataFrame(input_data, columns=['X', 'Y']), ordered_list, time.perf_counter() - t1, filtered_clusters)
    return filtered_clusters, ordered_list, results

def plot_cdf(matrix):
    values = np.sort(matrix.flatten())
    p = 1. * np.arange(len(values)) / (len(values) - 1)
    plt.figure(figsize=(10, 6))
    plt.plot(values, p, marker='.', linestyle='none')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF) of Matrix Values')
    plt.grid(True)
    plt.show()

def plot_approximate_cdf(matrix, n_bins=1000):
    values = matrix.flatten()
    counts, bin_edges = np.histogram(values, bins=n_bins)
    cdf = np.cumsum(counts) / cdf[-1]
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[1:], cdf, '-')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Approximate Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.show()

def Dspectral_clustering(df, n_clusters=None, n_neighbors=20, max_eigenvectors=40):
    """
    Perform spectral clustering on a large DataFrame with X and Y coordinates.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'X' and 'Y' columns
    n_clusters (int, optional): Number of clusters to form. If None, auto-determined.
    n_neighbors (int): Number of nearest neighbors for sparse similarity matrix
    max_eigenvectors (int): Maximum number of eigenvectors to compute
    
    Returns:
    numpy.ndarray: Cluster labels for each point
    """
    coords = df[['X', 'Y']].values
    
    # Normalize the input data
    scaler = StandardScaler()
    coords_normalized = scaler.fit_transform(coords)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1).fit(coords_normalized)
    distances, indices = nbrs.kneighbors(coords_normalized)

    # Efficiently compute the sparse similarity matrix
    sigma = np.mean(distances, axis=1)
    rows = np.repeat(np.arange(coords_normalized.shape[0]), n_neighbors)
    cols = indices.ravel()
    similarities = np.exp(-distances.ravel()**2 / (2 * np.repeat(sigma, n_neighbors)**2))
    
    similarity_matrix = csr_matrix((similarities, (rows, cols)), shape=(coords_normalized.shape[0], coords_normalized.shape[0]))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    # A diagonal matrix where each diagonal entry represents the sum of similarities (weights) of the edges connected to a node (point). It is effectively a measure of how connected each point is in the graph.
    degree_matrix = np.array(similarity_matrix.sum(axis=1)).flatten()
    # The Laplacian matrix is computed as 𝐷 − 𝑊, where 𝐷 is the degree matrix and is the similarity matrix. The Laplacian matrix encodes the graph structure and is crucial for spectral clustering.
    laplacian = csr_matrix(np.diag(degree_matrix)) - similarity_matrix
    # The normalized Laplacian is computed to scale the Laplacian by the inverse of the degree matrix, which can make the eigenvalue problem more stable and interpretable.
    norm_laplacian = laplacian.multiply(csr_matrix(1 / degree_matrix[:, None]))
    
    # Compute eigenvectors of the normalized Laplacian
    eigenvalues, eigenvectors = eigsh(norm_laplacian, k=max_eigenvectors, which='SM', return_eigenvectors=True)
    
    # Determine the number of clusters using auto_spectral_gap
    if n_clusters is None:
        n_clusters = auto_spectral_gap(eigenvalues)
    else:
        n_clusters = min(n_clusters, len(eigenvalues))
    
    # Select the appropriate number of eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]
    
    U_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    
    labels = kmeans(U_normalized, n_clusters)
    
    return labels, n_clusters


def compute_scaled_fixed_distance(cell_size, scaler):
    """
    Compute the scaled fixed distance in the normalized space given the distance in original space.

    Parameters:
    cell_size (float): Distance between two points in original space.
    scaler (StandardScaler): The scaler used to normalize the data.

    Returns:
    float: Distance in the scaled (normalized) space.
    """
    # Get the standard deviations used in scaling
    std_dev = scaler.scale_

    # Calculate the distance between point1 and point2 in original space
    distance_original = cell_size

    # Calculate the scaling factor
    scaling_factor = np.sqrt(np.sum(std_dev ** 2))

    # Scale the fixed distance
    distance_scaled = distance_original / scaling_factor

    return distance_scaled

def spectral_clustering(df, n_clusters=None, n_neighbors=20, max_eigenvectors=40, plot=False, min_cluster=0, tau_percentile=0, cell_size=0):

    """
    Perform spectral clustering on a large DataFrame with X and Y coordinates, removing links with a distance lower than tau.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'X' and 'Y' columns
    n_clusters (int, optional): Number of clusters to form. If None, auto-determined.
    n_neighbors (int): Number of nearest neighbors for sparse similarity matrix
    max_eigenvectors (int): Maximum number of eigenvectors to compute
    tau_percentile (float): Percentile of distances to use as threshold for removing links

    Returns:
    numpy.ndarray: Cluster labels for each point
    """

    coords = df[['X', 'Y']].values

    # Normalize the input data
    scaler = StandardScaler()
    coords_normalized = scaler.fit_transform(coords)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1).fit(coords_normalized)
    distances, indices = nbrs.kneighbors(coords_normalized)
    if cell_size > 0:
        distance_scaled = compute_scaled_fixed_distance(cell_size, scaler)
        # distances are infinite when the distance is higher scaled distance
        distances[distances > distance_scaled] = np.inf


    # Determine the distance threshold tau based on a percentile
    tau = np.percentile(distances, tau_percentile)

    # Efficiently compute the sparse similarity matrix
    sigma = np.mean(distances, axis=1)
    rows = np.repeat(np.arange(coords_normalized.shape[0]), n_neighbors)
    cols = indices.ravel()

    # Gaussian kernel similarity
    #similarities = np.exp(-distances.ravel()**2 / (2 * sigma[rows]**2))
    # Laplacian kernel similarity
    similarities = np.exp(-np.abs(distances.ravel()) / sigma[rows])
    # Apply the threshold tau to remove links with distance below tau
    mask = distances.ravel() >= tau  # Create a boolean mask for distances above tau
    similarities[~mask] = 0  # Set similarities to 0 where distance is below tau

    similarity_matrix = csr_matrix((similarities, (rows, cols)), shape=(coords_normalized.shape[0], coords_normalized.shape[0]))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    # A diagonal matrix where each diagonal entry represents the sum of similarities (weights) of the edges connected to a node (point). It is effectively a measure of how connected each point is in the graph.
    degree_matrix = np.array(similarity_matrix.sum(axis=1)).flatten()
    # The Laplacian matrix is computed as 𝐷 − 𝑊, where 𝐷 is the degree matrix and is the similarity matrix. The Laplacian matrix encodes the graph structure and is crucial for spectral clustering.
    laplacian = csr_matrix(np.diag(degree_matrix)) - similarity_matrix
    # The normalized Laplacian is computed to scale the Laplacian by the inverse of the degree matrix, which can make the eigenvalue problem more stable and interpretable.
    norm_laplacian = laplacian.multiply(csr_matrix(1 / degree_matrix[:, None]))

    # Compute eigenvectors of the normalized Laplacian
    eigenvalues, eigenvectors = eigsh(norm_laplacian, k=max_eigenvectors, which='SM', return_eigenvectors=True)

    # Determine the number of clusters using auto_spectral_gap
    if n_clusters is None:
        n_clusters = auto_spectral_gap(eigenvalues, min_cluster)
    else:
        n_clusters = min(n_clusters, len(eigenvalues))

    if plot is True:
        plot_cdf(eigenvalues)
        plot_approximate_cdf(eigenvalues)
        plot_gradients(eigenvalues)

    # Select the appropriate number of eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]

    U_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    labels = kmeans(U_normalized, n_clusters)

    return labels, n_clusters

def Gspectral_clustering(df, n_clusters=None, n_neighbors=20, max_eigenvectors=40, plot=False, min_cluster=0):
    """
    Perform spectral clustering on a large DataFrame with X and Y coordinates.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'X' and 'Y' columns
    n_clusters (int, optional): Number of clusters to form. If None, auto-determined.
    n_neighbors (int): Number of nearest neighbors for sparse similarity matrix
    max_eigenvectors (int): Maximum number of eigenvectors to compute

    Returns:
    numpy.ndarray: Cluster labels for each point
    """
    coords = df[['X', 'Y']].values

    # Normalize the input data
    coords_normalized = StandardScaler().fit_transform(coords)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1).fit(coords_normalized)
    distances, indices = nbrs.kneighbors(coords_normalized)

    # Efficiently compute the sparse similarity matrix
    sigma = np.mean(distances, axis=1)
    rows = np.repeat(np.arange(coords_normalized.shape[0]), n_neighbors)
    cols = indices.ravel()
    similarities = np.exp(-distances.ravel()**2 / (2 * sigma[rows]**2))
    similarity_matrix = csr_matrix((similarities, (rows, cols)), shape=(coords_normalized.shape[0], coords_normalized.shape[0]))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    # A diagonal matrix where each diagonal entry represents the sum of similarities (weights) of the edges connected to a node (point). It is effectively a measure of how connected each point is in the graph.
    degree_matrix = np.array(similarity_matrix.sum(axis=1)).flatten()
    # The Laplacian matrix is computed as 𝐷 − 𝑊, where 𝐷 is the degree matrix and is the similarity matrix. The Laplacian matrix encodes the graph structure and is crucial for spectral clustering.
    laplacian = csr_matrix(np.diag(degree_matrix)) - similarity_matrix
    # The normalized Laplacian is computed to scale the Laplacian by the inverse of the degree matrix, which can make the eigenvalue problem more stable and interpretable.
    norm_laplacian = laplacian.multiply(csr_matrix(1 / degree_matrix[:, None]))

    # Compute eigenvectors of the normalized Laplacian
    eigenvalues, eigenvectors = eigsh(norm_laplacian, k=max_eigenvectors, which='SM', return_eigenvectors=True)

    # Determine the number of clusters using auto_spectral_gap
    if n_clusters is None:
        n_clusters = auto_spectral_gap(eigenvalues, min_cluster)
    else:
        n_clusters = min(n_clusters, len(eigenvalues))

    if plot is True:
        plot_cdf(eigenvalues)
        plot_approximate_cdf(eigenvalues)
        plot_gradients(eigenvalues)

    # Select the appropriate number of eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]

    U_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    labels = kmeans(U_normalized, n_clusters)

    return labels, n_clusters


def auto_spectral_gap(evals, min_cluster=0):
    """
    Determines the optimal number of clusters by analyzing the spectral gap of eigenvalues.

    Parameters:
    evals (numpy.ndarray): Array of eigenvalues sorted in ascending order.
    min_cluster (int): Minimum index for the cluster count (default is 0).

    Returns:
    int: Index indicating the optimal number of clusters.
    """
    # Compute the second derivative of the eigenvalues to identify curvature changes
    second_derivative = np.gradient(np.gradient(evals, edge_order=2), edge_order=2)

    # Avoid numerical instability by adding a small constant
    second_derivative += 1e-8

    # Find the first positive curvature point
    positive_indices = np.where(second_derivative > 0)[0]
    if positive_indices.size == 0:
        # If no positive curvature, return min_cluster or the last index
        return max(min_cluster, len(evals) - 1)

    pos_point = positive_indices[0]

    # Find the indices where the second derivative is negative
    negative_indices = np.where(second_derivative < 0)[0]

    # Search for the next significant change in curvature after the first positive point
    i = 0
    while True:
        if i >= len(negative_indices):
            # If there are no more negative indices, use the last index
            cut_point = len(evals) - 1
            break

        # Compute cut_point
        cut_point = negative_indices[i]

        # If cut_point is valid
        if cut_point > min_cluster and cut_point > pos_point:
            cut_point = cut_point
            break

        # Otherwise, move to the next negative index
        i += 1

    print("pos_point = " + str(pos_point) + " cut_point = " + str(cut_point))
    return cut_point

def plot_cdf(evals):
    """
    Plot the cumulative distribution function (CDF) of eigenvalues.

    Parameters:
    evals (numpy.ndarray): Array of eigenvalues sorted in ascending order.
    """
    values = np.sort(evals)
    p = 1. * np.arange(len(values)) / (len(values) - 1)
    plt.figure(figsize=(10, 6))
    plt.plot(values, p, marker='.', linestyle='none')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF) of Eigenvalues')
    plt.grid(True)
    plt.show()

def plot_approximate_cdf(evals, n_bins=1000):
    """
    Plot an approximate cumulative distribution function (CDF) of eigenvalues.

    Parameters:
    evals (numpy.ndarray): Array of eigenvalues.
    n_bins (int): Number of bins for the histogram.
    """
    values = evals
    counts, bin_edges = np.histogram(values, bins=n_bins)
    cdf = np.cumsum(counts) / counts[-1]
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[1:], cdf, '-')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Approximate Cumulative Distribution Function (CDF) of Eigenvalues')
    plt.grid(True)
    plt.show()

def plot_gradients(evals):
    """
    Plot the gradients of eigenvalues to identify the spectral gap.

    Parameters:
    evals (numpy.ndarray): Array of eigenvalues sorted in ascending order.
    """
    gradient1 = np.gradient(evals)
    gradient2 = np.gradient(gradient1)
    plt.figure(figsize=(10, 6))
    plt.plot(gradient1, label='First Gradient')
    plt.plot(gradient2, label='Second Gradient')
    plt.xlabel('Index')
    plt.ylabel('Gradient')
    plt.title('Gradients of Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.show()

def kmeans(X: np.ndarray, k: int, n_runs: int = 10, max_iters: int = 100) -> np.ndarray:
    """
    Optimized k-means clustering algorithm.

    Parameters:
    X (np.ndarray): Input data points.
    k (int): Number of clusters.
    n_runs (int): Number of different initializations to run.
    max_iters (int): Maximum number of iterations for each run.

    Returns:
    np.ndarray: Best cluster labels determined across all runs.
    """
    best_labels = None  # Store the labels of the best clustering found.
    best_inertia = np.inf  # Store the inertia of the best clustering found.

    n_samples, n_features = X.shape

    # Repeat the process for the number of specified runs
    for _ in range(n_runs):
        # Initialize centroids using k-means++ to improve convergence speed
        centroids = initialize_centroids(X, k)

        for _ in range(max_iters):
            # Compute distances from each point to each centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Assign each point to the nearest centroid
            labels = np.argmin(distances, axis=1)

            # Recalculate centroids as the mean of assigned points
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])

            # Check for convergence: if centroids do not change, break
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        # Compute inertia (sum of squared distances to the closest centroid)
        inertia = np.sum((X - centroids[labels]) ** 2)

        # Keep track of the best run
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels


def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """
    Initialize centroids using the k-means++ algorithm.

    Parameters:
    X (np.ndarray): Input data points.
    k (int): Number of clusters.

    Returns:
    np.ndarray: Initialized centroids.
    """
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    centroids[0] = X[np.random.randint(n_samples)]

    for i in range(1, k):
        distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2), axis=1)
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        centroids[i] = X[np.searchsorted(cumulative_probabilities, r)]

    return centroids


def run_spectral_clustering(input_data, n_clusters: Optional[int] = None, verbose: int = 1, plot: bool = False, min_cluster = 0):
    input_data = input_data.values if hasattr(input_data, 'values') else input_data
    if verbose >= 2:
        plot_points(input_data, title='Input data')


    t1 = time.perf_counter()
    cluster_labels, n_clusters = spectral_clustering(pd.DataFrame(input_data, columns=['X', 'Y']), n_clusters, plot=plot, min_cluster=min_cluster)
    processing_time = time.perf_counter() - t1
    
    if verbose >= 1:
        print(f'Total time for processing: {processing_time:.2f} s')
    
    clusters = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    if verbose >= 2:
        plot_points(input_data, clusters=clusters, title='Spectral Clustering Results')
    
    results = saveResults(pd.DataFrame(input_data, columns=['X', 'Y']), [], processing_time, clusters)
    return clusters, results