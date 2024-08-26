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


def spectral_clustering(df, n_clusters=None, n_neighbors=20, max_eigenvectors=50): #guido
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
        n_clusters = auto_spectral_gap(eigenvalues)
    else:
        n_clusters = min(n_clusters, len(eigenvalues))

    # Select the appropriate number of eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]

    U_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    labels = kmeans(U_normalized, n_clusters)

    return labels, n_clusters

def Dauto_spectral_gap(evals):
    """
    The auto_spectral_gap function is designed to determine the optimal number of clusters (cut_point) by analyzing the spectral gap of the eigenvalues (evals) of a matrix, typically the Laplacian matrix in spectral clustering.
    :param evals: eigenvalues
    :return: index of evals where the spectral gap is significant
    """
    # The gradient provides an approximation of the rate of change of the eigenvalues, which helps identify where the spectral gap (difference between successive eigenvalues) is significant.
    tmp = np.gradient(evals, edge_order=2) # order 2 for better accuracy
    tmp = np.gradient(tmp, edge_order=2)
    tmp = tmp[1:]
    tmp = tmp + 1e-8  # fix numerical instability
    pos_point = np.argmax(tmp > 0) # find the first positive point
    tmp = tmp[pos_point+1:] # Identifying the Subsequent Spectral Gap
    cut_point = np.argmax(tmp < 0) + 1 + pos_point + 1 # adjustment recalculates the absolute index in the original evals array since tmp was sliced
    return cut_point


def auto_spectral_gap(evals): #guido
    """
    Determines the optimal number of clusters by analyzing the spectral gap of eigenvalues.

    Parameters:
    evals (numpy.ndarray): Array of eigenvalues sorted in ascending order.

    Returns:
    int: Index indicating the optimal number of clusters.
    """
    # Compute the second derivative of the eigenvalues to identify curvature changes
    second_derivative = np.gradient(np.gradient(evals, edge_order=2), edge_order=2)

    # Avoid numerical instability by adding a small constant
    second_derivative += 1e-8

    # Find the first point where the curvature becomes positive
    pos_point = np.argmax(second_derivative > 0)

    # Search for the next significant change in curvature after the first positive point
    cut_point = pos_point + 1 + np.argmax(second_derivative[pos_point + 1:] < 0)

    return cut_point


def Dkmeans(X: np.ndarray, k: int, n_runs: int = 10, max_iters: int = 100) -> np.ndarray:
    best_labels = None  # store the labels of the best clustering found.
    best_inertia = np.inf # store the inertia of the best clustering found.

    for _ in range(n_runs): # run the algorithm multiple times to avoid local minima
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        
        for _ in range(max_iters): # run the algorithm for a maximum number of iterations
            distances = ((X - centroids[:, np.newaxis])**2).sum(axis=2)
            labels = np.argmin(distances, axis=0)
            
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # If a cluster is empty, reinitialize it
                    new_centroids[i] = X[np.random.choice(X.shape[0])]
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids # update centroids
        
        inertia = np.sum((X - centroids[labels])**2) # compute the inertia of the clustering
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    
    return best_labels


import numpy as np


def kmeans(X: np.ndarray, k: int, n_runs: int = 10, max_iters: int = 100) -> np.ndarray: #guido
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


def Drun_spectral_clustering(input_data, n_clusters: Optional[int] = None, verbose: int = 1):
    input_data = input_data.values if hasattr(input_data, 'values') else input_data
    if verbose >= 2:
        plot_points(input_data, title='Input data')
    
    t1 = time.perf_counter()
    cluster_labels, n_clusters = spectral_clustering(pd.DataFrame(input_data, columns=['X', 'Y']), n_clusters)
    processing_time = time.perf_counter() - t1
    
    if verbose >= 1:
        print(f'Total time for processing: {processing_time:.2f} s')
    
    clusters = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    if verbose >= 2:
        plot_points(input_data, clusters=clusters, title='Spectral Clustering Results')
    
    results = saveResults(pd.DataFrame(input_data, columns=['X', 'Y']), [], processing_time, clusters)
    return clusters, results


def run_spectral_clustering(input_data, n_clusters: Optional[int] = None, verbose: int = 1): #guido
    """
    Perform spectral clustering on input data and optionally visualize the results.

    Parameters:
    input_data : array-like or pandas.DataFrame
        The input data to cluster.
    n_clusters : int, optional
        The number of clusters to form. If None, the number of clusters is determined automatically.
    verbose : int
        The verbosity level. Higher values result in more detailed output and plotting.

    Returns:
    clusters : list of numpy.ndarray
        Indices of input data points for each cluster.
    results : object
        The result object from the saveResults function, which might contain processed data and statistics.
    """

    # Ensure input data is in DataFrame format and store it once
    input_df = pd.DataFrame(input_data if not hasattr(input_data, 'values') else input_data.values, columns=['X', 'Y'])

    if verbose >= 2:
        plot_points(input_df.values, title='Input data')

    # Start timing and perform spectral clustering
    t1 = time.perf_counter()
    cluster_labels, n_clusters = spectral_clustering(input_df, n_clusters)
    processing_time = time.perf_counter() - t1

    if verbose >= 1:
        print(f'Total time for processing: {processing_time:.2f} s')

    # Group indices by cluster label efficiently
    clusters = [np.flatnonzero(cluster_labels == i) for i in range(n_clusters)]

    if verbose >= 2:
        plot_points(input_df.values, clusters=clusters, title='Spectral Clustering Results')

    # Save results and return
    results = saveResults(input_df, [], processing_time, clusters)
    return clusters, results