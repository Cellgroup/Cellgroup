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

def spectral_clustering(df, n_clusters=None, n_neighbors=20, max_eigenvectors=50):
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
    
    degree_matrix = np.array(similarity_matrix.sum(axis=1)).flatten()
    laplacian = csr_matrix(np.diag(degree_matrix)) - similarity_matrix
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

def auto_spectral_gap(evals):
    tmp = np.gradient(evals, edge_order=2)
    tmp = np.gradient(tmp, edge_order=2)
    tmp = tmp[1:]
    tmp = tmp + 1e-8  # fix numerical instability
    pos_point = np.argmax(tmp > 0)
    tmp = tmp[pos_point+1:]
    cut_point = np.argmax(tmp < 0) + 1 + pos_point + 1
    return cut_point

def kmeans(X: np.ndarray, k: int, n_runs: int = 10, max_iters: int = 100) -> np.ndarray:
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_runs):
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        
        for _ in range(max_iters):
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
            
            centroids = new_centroids
        
        inertia = np.sum((X - centroids[labels])**2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    
    return best_labels


def run_spectral_clustering(input_data, n_clusters: Optional[int] = None, verbose: int = 1):
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