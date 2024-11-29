from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

class TestScenarios:
    """Predefined test scenarios for clustering evaluation"""

    @staticmethod
    def test_split_scenario():
        """Generate test data with cluster splitting"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        # Initial single cluster
        center = np.array([0, 0])

        for t in range(n_timepoints):
            if t < 2:
                # Single cluster
                X, y = make_blobs(n_samples=n_points,
                                centers=[center],
                                cluster_std=0.5)
            else:
                # Split into two clusters
                centers = np.array([center + [2, 0], center - [2, 0]])
                X, y = make_blobs(n_samples=n_points,
                                centers=centers,
                                cluster_std=0.5)

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': centers if t >= 2 else np.array([center]),
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth

    @staticmethod
    def test_merge_scenario():
        """Generate test data with cluster merging"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        # Initial two clusters
        centers = np.array([[2, 0], [-2, 0]])

        for t in range(n_timepoints):
            if t < 2:
                # Two separate clusters
                X, y = make_blobs(n_samples=n_points,
                                centers=centers,
                                cluster_std=0.5)
            else:
                # Merged into one cluster
                X, y = make_blobs(n_samples=n_points,
                                centers=[np.array([0, 0])],
                                cluster_std=0.5)

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': np.array([0, 0]).reshape(1, -1) if t >= 2 else centers,
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth

    @staticmethod
    def test_density_variation():
        """Generate test data with varying densities"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        for t in range(n_timepoints):
            # Generate clusters with different densities
            centers = np.array([[0, 0], [5, 5], [-5, -5]])
            cluster_stds = [0.3, 0.7, 1.0]  # Different spreads

            X = np.vstack([
                np.random.normal(center, std, (n_points // 3, 2))
                for center, std in zip(centers, cluster_stds)
            ])
            y = np.repeat(np.arange(len(centers)), n_points // 3)

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': centers,
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth


class ImprovedTemporalClusterTracker:

    def __init__(self, verbose: bool = False):
        """
        Initialize a simplified temporal cluster tracker.
        Focus on basic spectral clustering with temporal consistency.
        """
        self.verbose = verbose
        self.cluster_history = {}
        self.next_cluster_id = 0
        self.previous_results = {}

    def process_timepoint(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], Dict[int, int]]:
        """
        Process a single timepoint with improved cluster number detection.
        """
        points = df[['X', 'Y']].values
        time = df['Time'].iloc[0]

        # 1. Build affinity matrix
        affinity = self._build_affinity_matrix(points)

        # 2. Compute normalized Laplacian
        degree = np.array(affinity.sum(axis=1)).flatten()
        degree_matrix = csr_matrix(np.diag(degree))
        laplacian = degree_matrix - affinity
        normalized_laplacian = laplacian.multiply(1 / np.sqrt(degree)[:, None])

        # 3. Get eigenvectors - reduced number to avoid noise
        k = min(4, points.shape[0] - 1)  # Reduced from 5 to 4
        eigenvalues, eigenvectors = eigsh(normalized_laplacian, k=k, which='SM')

        # 4. Improved cluster number determination
        n_clusters = self._determine_cluster_number(eigenvalues)

        # 5. Perform k-means on eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(eigenvectors[:, :n_clusters])

        # 6. Convert to clusters
        unique_labels = np.unique(labels)
        clusters = [np.where(labels == label)[0] for label in unique_labels]

        # 7. Match with previous timepoint
        matching = self._match_clusters(clusters, points, time)

        # 8. Update history
        self._update_trajectories(clusters, points, time, matching)

        # Store results
        self.previous_results[time] = {
            'points': points,
            'labels': labels
        }

        return clusters, matching

    def _determine_cluster_number(self, eigenvalues: np.ndarray) -> int:
        """
        Improved cluster number determination using eigenvalue analysis.
        """
        # Sort eigenvalues and compute gaps
        sorted_eigs = np.sort(eigenvalues)
        gaps = np.diff(sorted_eigs)

        # Normalize gaps
        normalized_gaps = gaps / np.mean(gaps)

        # Find significant gaps
        significant_gaps = np.where(normalized_gaps > 1.5)[0]  # Increased threshold

        if len(significant_gaps) == 0:
            # If no significant gaps, assume minimal clustering
            return 2

        # The number of clusters is one more than the index of the largest gap
        n_clusters = significant_gaps[0] + 1  # Changed from +2 to +1

        # Stricter bounds
        return min(max(n_clusters, 1), 3)  # Changed upper bound to 3

    def _build_affinity_matrix(self, points: np.ndarray) -> csr_matrix:
        """
        Build affinity matrix with improved scaling.
        """
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))

        # Use more robust bandwidth estimation
        # Take the 20th percentile of non-zero distances for tighter clusters
        non_zero_distances = distances[distances > 0]
        sigma = np.percentile(non_zero_distances, 20)

        # Compute affinities with sharper cutoff
        affinities = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # Apply threshold to create more distinct clusters
        affinities[affinities < 0.1] = 0  # Remove weak connections

        return csr_matrix(affinities)

    def _match_clusters(self, clusters: List[np.ndarray], points: np.ndarray, time: int) -> Dict[int, int]:
        """
        Match clusters with improved thresholds.
        """
        if not self.cluster_history:
            return {i: i for i in range(len(clusters))}

        # Get previous active clusters
        active_clusters = {
            cluster_id: info
            for cluster_id, info in self.cluster_history.items()
            if info['last_seen'] == time - 1
        }

        if not active_clusters:
            return {i: self.next_cluster_id + i for i in range(len(clusters))}

        # Compute current centroids
        current_centroids = [np.mean(points[cluster], axis=0) for cluster in clusters]

        # Compute previous centroids
        prev_centroids = [
            np.mean(info['points'][-1], axis=0)
            for info in active_clusters.values()
        ]

        # Build distance matrix
        distances = np.array([
            [np.linalg.norm(c1 - c2) for c2 in prev_centroids]
            for c1 in current_centroids
        ])

        # More stringent matching threshold
        threshold = np.median(distances) * 1.5  # Reduced from 2 to 1.5

        # Find best matches
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(distances)

        # Create matching with stricter threshold
        matching = {}
        prev_ids = list(active_clusters.keys())

        for i, j in zip(row_ind, col_ind):
            if distances[i, j] < threshold:
                matching[i] = prev_ids[j]
            else:
                matching[i] = self.next_cluster_id
                self.next_cluster_id += 1

        # Handle unmatched clusters
        for i in range(len(clusters)):
            if i not in matching:
                matching[i] = self.next_cluster_id
                self.next_cluster_id += 1

        return matching


    def _update_trajectories(self, clusters: List[np.ndarray], points: np.ndarray, time: int, matching: Dict[int, int]):
        """
        Update cluster trajectories.
        """
        for cluster_idx, cluster_id in matching.items():
            cluster_points = points[clusters[cluster_idx]]

            if cluster_id in self.cluster_history:
                self.cluster_history[cluster_id]['points'].append(cluster_points)
                self.cluster_history[cluster_id]['last_seen'] = time
            else:
                self.cluster_history[cluster_id] = {
                    'points': [cluster_points],
                    'first_seen': time,
                    'last_seen': time
                }
    def _estimate_parameters(self, points: np.ndarray) -> None:
        """
        Estimate all necessary parameters from the data structure.
        """
        n_points = len(points)

        # Compute distance statistics
        tree = NearestNeighbors(n_neighbors=min(20, n_points - 1))
        tree.fit(points)
        distances, _ = tree.kneighbors()

        # Estimate characteristic scale of the data
        self.characteristic_scale = np.median(distances[:, -1])

        # Estimate number of neighbors from data density
        self.n_neighbors = max(int(np.sqrt(n_points)), 5)

        # Estimate number of clusters from eigenvalue spectrum
        self.max_eigenvectors = self._estimate_n_clusters_from_spectrum(points)

        # Minimum cluster size based on statistical significance
        self.min_cluster_size = max(int(np.sqrt(n_points / 10)), 5)

        # Adaptive similarity threshold from distance distribution
        self.similarity_threshold = np.exp(-1)  # Natural threshold at 1 std

        if len(self.previous_results) > 0:
            # Estimate temporal parameters from previous data
            prev_points = list(self.previous_results.values())[-1]['points']
            self.temporal_weight = self._estimate_temporal_weight(points, prev_points)
        else:
            self.temporal_weight = 0.3  # Initial default

        # Time gap based on temporal coherence
        self.max_time_gap = 1  # Start conservative


    def _estimate_n_clusters_from_spectrum(self, points: np.ndarray) -> int:
        """
        Estimate number of clusters from eigenvalue spectrum of the Laplacian.
        """
        # Build affinity matrix
        tree = NearestNeighbors(n_neighbors=min(20, len(points) - 1))
        tree.fit(points)
        distances, indices = tree.kneighbors()

        # Compute adaptive bandwidth
        bandwidth = np.median(distances[:, -1])

        # Build sparse affinity matrix
        rows = np.repeat(np.arange(len(points)), distances.shape[1])
        cols = indices.reshape(-1)
        vals = np.exp(-distances.reshape(-1) ** 2 / (2 * bandwidth ** 2))
        affinity = csr_matrix((vals, (rows, cols)), shape=(len(points), len(points)))
        affinity = (affinity + affinity.T) / 2

        # Compute normalized Laplacian
        degree = np.array(affinity.sum(axis=1)).flatten()
        laplacian = csr_matrix(np.diag(degree)) - affinity
        norm_laplacian = laplacian.multiply(1 / np.sqrt(degree)[:, None])

        # Get eigenvalues
        try:
            eigenvalues, _ = eigsh(norm_laplacian, k=min(50, len(points) - 1),
                                   which='SM', return_eigenvectors=True)

            # Find significant gap in eigenvalue spectrum
            gaps = np.diff(eigenvalues)
            significant_gaps = np.where(gaps > np.mean(gaps) + np.std(gaps))[0]

            if len(significant_gaps) > 0:
                n_clusters = significant_gaps[0] + 1
            else:
                # Fallback: estimate from eigenvalue decay
                decay = np.abs(np.diff(gaps))
                n_clusters = np.argmax(decay) + 2

            return min(max(n_clusters, 2), 10)  # Reasonable bounds

        except:
            return 2  # Conservative fallback

    def _estimate_temporal_weight(self,
                                  current_points: np.ndarray,
                                  previous_points: np.ndarray) -> float:
        """
        Estimate temporal weight from consecutive point sets.
        """
        # Compute typical displacement
        tree = NearestNeighbors(n_neighbors=1)
        tree.fit(previous_points)
        distances, _ = tree.kneighbors(current_points)

        # Compute characteristic scales
        current_scale = self._compute_characteristic_scale(current_points)
        prev_scale = self._compute_characteristic_scale(previous_points)

        # Adjust weight based on relative displacement
        displacement = np.median(distances)
        rel_displacement = displacement / np.mean([current_scale, prev_scale])

        return np.exp(-rel_displacement)

    def _compute_characteristic_scale(self, points: np.ndarray) -> float:
        """
        Compute characteristic scale of point distribution.
        """
        tree = NearestNeighbors(n_neighbors=min(20, len(points) - 1))
        tree.fit(points)
        distances, _ = tree.kneighbors()
        return np.median(distances[:, -1])


    def _perform_spectral_clustering(self, points: np.ndarray, prev_df: Optional[pd.DataFrame]) -> np.ndarray:
        """
        Perform spectral clustering with adaptive parameters.
        """
        # Build affinity matrix
        affinity = self._build_affinity_matrix(points)

        if prev_df is not None:
            # Add temporal component
            temporal_affinity = self._build_temporal_affinity(points, prev_df[['X', 'Y']].values)
            affinity = (1 - self.temporal_weight) * affinity + self.temporal_weight * temporal_affinity

        # Compute normalized Laplacian
        degree = np.array(affinity.sum(axis=1)).flatten()
        laplacian = csr_matrix(np.diag(degree)) - affinity
        norm_laplacian = laplacian.multiply(1 / np.sqrt(degree)[:, None])

        # Compute eigenvectors
        n_components = min(self.max_eigenvectors, len(points) - 1)
        eigenvalues, eigenvectors = eigsh(norm_laplacian, k=n_components, which='SM')

        # Perform clustering
        from sklearn.cluster import KMeans
        n_clusters = self._estimate_n_clusters_from_spectrum(points)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(eigenvectors)

        return labels


    def _build_temporal_affinity(self, current_points: np.ndarray, previous_points: np.ndarray) -> csr_matrix:
        """
        Build temporal affinity matrix.
        """
        tree = NearestNeighbors(n_neighbors=1)
        tree.fit(previous_points)
        distances, _ = tree.kneighbors(current_points)

        # Compute temporal bandwidth
        bandwidth = np.median(distances)
        similarities = np.exp(-distances ** 2 / (2 * bandwidth ** 2))

        return csr_matrix(np.diag(similarities.ravel()))



    def _get_previous_timepoint(self, current_time: int) -> Optional[pd.DataFrame]:
        """
        Get data from previous timepoint.
        """
        if current_time - 1 in self.previous_results:
            return pd.DataFrame(
                self.previous_results[current_time - 1]['points'],
                columns=['X', 'Y']
            )
        return None


    def _build_similarity_matrix(self, points: np.ndarray) -> csr_matrix:
        """
        Build similarity matrix with enhanced adaptive bandwidth.
        """
        # Compute nearest neighbors with error handling
        n_neighbors = min(self.n_neighbors, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors()

        # Compute adaptive bandwidth with local density awareness
        local_density = self._compute_local_density(points)
        sigma = np.mean(distances, axis=1) * np.exp(-local_density)

        # Build sparse similarity matrix with improved numerical stability
        rows = np.repeat(np.arange(len(points)), n_neighbors)
        cols = indices.ravel()

        # Compute similarities with density-aware scaling
        similarities = np.exp(-distances.ravel() ** 2 / (2 * sigma[rows] ** 2))
        similarities = np.clip(similarities, 1e-10, 1.0)  # Numerical stability

        # Create symmetric similarity matrix
        similarity_matrix = csr_matrix((similarities, (rows, cols)),
                                       shape=(len(points), len(points)))
        return (similarity_matrix + similarity_matrix.T) / 2

    def _build_temporal_similarity(self,
                                   current_points: np.ndarray,
                                   previous_points: np.ndarray) -> csr_matrix:
        """
        Build temporal similarity matrix with improved temporal coherence.
        """
        # Use k-nearest neighbors for more robust temporal matching
        k = min(3, len(previous_points))
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(previous_points)
        distances, indices = nbrs.kneighbors(current_points)

        # Compute adaptive temporal bandwidth
        temporal_sigma = np.median(distances) * 1.4  # Slightly increased for better temporal linking

        # Convert distances to similarities with soft thresholding
        max_distance = np.percentile(distances, 95)  # Ignore outliers
        normalized_distances = np.minimum(distances, max_distance) / max_distance
        similarities = np.exp(-normalized_distances ** 2 / (2 * temporal_sigma ** 2))

        # Average similarities from k-nearest neighbors
        similarities = np.mean(similarities, axis=1)

        return csr_matrix(np.diag(similarities.ravel()))

    def _compute_dynamic_temporal_weight(self,
                                         current_points: np.ndarray,
                                         previous_points: np.ndarray) -> float:
        """
        Compute dynamic temporal weight based on point distribution changes.
        """
        # Compare point distributions
        current_density = self._compute_local_density(current_points)
        previous_density = self._compute_local_density(previous_points)

        # Compute distribution similarity
        density_diff = np.abs(np.mean(current_density) - np.mean(previous_density))
        density_similarity = np.exp(-density_diff)

        # Adjust temporal weight
        base_weight = self.temporal_weight
        return base_weight * density_similarity

    def _enhance_clustering(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Enhanced clustering with improved split detection and density awareness.
        """
        enhanced_labels = labels.copy()
        n_clusters = len(np.unique(labels))

        # Compute local density with adaptive bandwidth
        density = self._compute_local_density(points)

        # Normalize density for better comparison
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)

        # Check each cluster for potential splits
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= 1.5 * self.min_cluster_size:  # Changed threshold
                cluster_points = points[cluster_mask]
                cluster_density = density[cluster_mask]

                # Improved split detection using density-aware criteria
                split_score = self._compute_enhanced_split_score(
                    cluster_points,
                    cluster_density
                )

                if split_score > 0.6:  # Reduced threshold
                    sub_labels = self._density_aware_split(
                        cluster_points,
                        cluster_density
                    )
                    enhanced_labels[cluster_mask] = sub_labels + n_clusters
                    n_clusters += len(np.unique(sub_labels))

        return enhanced_labels

    def _compute_enhanced_split_score(self, points: np.ndarray, density: np.ndarray) -> float:
        """
        Improved split score computation using multiple criteria.
        """
        # Compute spatial distribution
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        # Kernel density estimation with adaptive bandwidth
        bandwidth = np.std(distances) * 0.5
        kde = stats.gaussian_kde(distances, bw_method=bandwidth)
        x_eval = np.linspace(min(distances), max(distances), 200)
        density_eval = kde(x_eval)

        # Find peaks with improved sensitivity
        peaks, properties = find_peaks(density_eval,
                                       distance=len(x_eval) // 10,
                                       prominence=0.05)

        if len(peaks) > 1:
            # Consider both valley depth and peak prominence
            valley_depths = []
            for i in range(len(peaks) - 1):
                valley = np.min(density_eval[peaks[i]:peaks[i + 1]])
                avg_peak_height = (density_eval[peaks[i]] + density_eval[peaks[i + 1]]) / 2
                valley_depths.append(1 - valley / avg_peak_height)

            # Return maximum valley depth weighted by density variation
            density_variation = np.std(density)
            return max(valley_depths) * (1 + density_variation)

        return 0

    def _density_aware_split(self, points: np.ndarray, density: np.ndarray) -> np.ndarray:
        """
        Split clusters using density-aware spectral clustering.
        """
        # Build affinity matrix with density awareness
        n_neighbors = min(self.n_neighbors, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors()

        # Compute adaptive bandwidth using density
        sigma = np.mean(distances, axis=1) * np.exp(-density)[:, None]

        # Build sparse similarity matrix
        rows = np.repeat(np.arange(len(points)), n_neighbors)
        cols = indices.ravel()
        similarities = np.exp(-distances.ravel() ** 2 / (2 * sigma.ravel() ** 2))

        similarity_matrix = csr_matrix((similarities, (rows, cols)),
                                       shape=(len(points), len(points)))
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        # Compute normalized Laplacian
        degree = np.array(similarity_matrix.sum(axis=1)).flatten()
        laplacian = csr_matrix(np.diag(degree)) - similarity_matrix
        norm_laplacian = laplacian.multiply(1 / np.sqrt(degree)[:, None])

        # Compute eigenvectors
        n_clusters = 2  # Binary split
        _, eigenvectors = eigsh(norm_laplacian, k=n_clusters, which='SM')

        # Perform k-means on eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(eigenvectors)

        # Post-process to ensure minimum cluster size
        for i in range(n_clusters):
            if np.sum(labels == i) < self.min_cluster_size:
                # Assign to nearest large cluster
                other_label = 1 - i  # For binary split
                labels[labels == i] = other_label

        return labels


    def _compute_enhanced_features(self, points: np.ndarray) -> Dict:
        """
        Compute enhanced cluster features including shape and distribution metrics.
        """
        if len(points) < 2:
            return {
                'centroid': np.mean(points, axis=0),
                'radius': 0,
                'density': 0,
                'size': len(points),
                'shape': 0,
                'orientation': 0
            }

        # Basic features
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        # Compute covariance and its eigenvalues for shape analysis
        covar = np.cov(points.T)
        if covar.shape == (2, 2):  # Ensure we have enough points
            eigenvals, eigenvecs = np.linalg.eigh(covar)
            shape = eigenvals[0] / (eigenvals[1] + 1e-10)  # Eccentricity
            orientation = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        else:
            shape = 0
            orientation = 0

        return {
            'centroid': centroid,
            'radius': np.max(distances),
            'density': len(points) / (np.pi * np.max(distances) ** 2 + 1e-10),
            'size': len(points),
            'shape': shape,
            'orientation': orientation
        }

    def _compute_enhanced_similarity(self,
                                     features1: Dict,
                                     features2: Dict) -> float:
        """
        Compute enhanced similarity between clusters using multiple metrics.
        """
        # Spatial similarity
        centroid_dist = np.linalg.norm(
            features1['centroid'] - features2['centroid']
        )
        spatial_sim = np.exp(-centroid_dist / (features1['radius'] + 1e-10))

        # Size similarity
        size_ratio = min(features1['size'], features2['size']) / \
                     max(features1['size'], features2['size'])

        # Shape similarity
        shape_diff = abs(features1['shape'] - features2['shape'])
        shape_sim = np.exp(-shape_diff)

        # Orientation similarity
        orient_diff = abs(features1['orientation'] - features2['orientation'])
        orient_sim = np.exp(-orient_diff)

        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # Adjusted weights
        return np.average([spatial_sim, size_ratio, shape_sim, orient_sim],
                          weights=weights)


class ClusteringTestFramework:
    def __init__(self):
        self.test_cases = {}
        self.results = {}

    def plot_cluster_evolution_with_points(self, test_name: str = 'synthetic', time_steps: List[int] = None):
        """
        Plot the evolution of clusters over time, showing actual data points.
        """
        results = self.results[test_name]
        data = self.test_cases[test_name]['data']

        # Get unique time points
        all_times = sorted(data['Time'].unique())
        if time_steps is None:
            time_steps = all_times

        # Calculate grid dimensions
        n_plots = len(time_steps)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

        # Get global data bounds for consistent axes
        x_min, x_max = data['X'].min(), data['X'].max()
        y_min, y_max = data['Y'].min(), data['Y'].max()

        # Plot each time step
        for idx, t in enumerate(time_steps):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Get data for this time step
            time_data = data[data['Time'] == t]

            # Get cluster assignments
            time_results = results['timepoint_results'][t]
            labels = np.zeros(len(time_data), dtype=int)
            clusters = time_results['cluster_sizes']

            # Create scatter plot
            scatter = ax.scatter(time_data['X'],
                                 time_data['Y'],
                                 c=labels,
                                 cmap='tab10',
                                 alpha=0.6)

            # Plot cluster centers if available
            if 'ground_truth' in self.test_cases[test_name]:
                centers = self.test_cases[test_name]['ground_truth'][t]['centers']
                ax.scatter(centers[:, 0], centers[:, 1],
                           c='red', marker='x', s=100, linewidths=3,
                           label='True Centers')

            # Set consistent bounds
            ax.set_xlim([x_min - 1, x_max + 1])
            ax.set_ylim([y_min - 1, y_max + 1])

            # Add title and labels
            ax.set_title(f'Time {t}')
            if col == 0:  # Only add y-label for leftmost plots
                ax.set_ylabel('Y')
            if row == n_rows - 1:  # Only add x-label for bottom plots
                ax.set_xlabel('X')

            # Add legend showing number of clusters
            ax.legend([f'{len(clusters)} clusters'], loc='upper right')

        # Remove empty subplots if any
        for idx in range(len(time_steps), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if idx < len(axes.flat):  # Check if the subplot exists
                fig.delaxes(axes.flat[idx])

        plt.tight_layout()
        plt.show()

    def generate_test_data(self,
                           n_timepoints: int = 10,
                           n_points: int = 1000,
                           n_base_clusters: int = 3,
                           noise_level: float = 0.1,
                           split_probability: float = 0.2,
                           merge_probability: float = 0.1) -> pd.DataFrame:
        """Generate synthetic test data with known ground truth"""
        all_data = []
        ground_truth = {}

        # Initial cluster centers
        centers = np.random.uniform(-10, 10, size=(n_base_clusters, 2))

        for t in range(n_timepoints):
            # Add random movement to centers
            centers += np.random.normal(0, 0.2, size=centers.shape)

            # Handle splits and merges
            if np.random.random() < split_probability and len(centers) < n_base_clusters * 2:
                # Split a random cluster
                split_idx = np.random.randint(len(centers))
                new_center = centers[split_idx] + np.random.normal(0, 1, size=2)
                centers = np.vstack([centers, new_center])

            if np.random.random() < merge_probability and len(centers) > n_base_clusters:
                # Merge closest centers
                distances = cdist(centers, centers)
                np.fill_diagonal(distances, np.inf)
                i, j = np.unravel_index(np.argmin(distances), distances.shape)
                centers = np.delete(centers, j, axis=0)

            # Generate points around centers
            X, y = make_blobs(n_samples=n_points,
                              centers=centers,
                              cluster_std=noise_level)

            # Create DataFrame for this timepoint
            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            # Store ground truth
            ground_truth[t] = {
                'centers': centers.copy(),
                'labels': y
            }

        self.test_cases['synthetic'] = {
            'data': pd.concat(all_data, ignore_index=True),
            'ground_truth': ground_truth
        }

        return self.test_cases['synthetic']['data']

    def run_tests(self, tracker: 'ImprovedTemporalClusterTracker') -> Dict:
        """Run tests and evaluate results"""
        results = {}

        for test_name, test_case in self.test_cases.items():
            print(f"Running test: {test_name}")

            df = test_case['data']
            ground_truth = test_case['ground_truth']

            # Process each timepoint
            timepoint_results = {}
            for time in sorted(df['Time'].unique()):
                time_df = df[df['Time'] == time]
                clusters, matching = tracker.process_timepoint(time_df)

                # Evaluate results
                metrics = self._evaluate_timepoint(
                    time_df, clusters, matching,
                    ground_truth[time]
                )

                timepoint_results[time] = metrics

            # Compute overall metrics
            results[test_name] = {
                'timepoint_results': timepoint_results,
                'overall_metrics': self._compute_overall_metrics(timepoint_results)
            }

        self.results = results
        return results

    def _evaluate_timepoint(self,
                            df: pd.DataFrame,
                            clusters: List[np.ndarray],
                            matching: Dict[int, int],
                            ground_truth: Dict) -> Dict:
        """Evaluate clustering results for a single timepoint"""
        # Convert clusters to labels
        labels = np.zeros(len(df), dtype=int)
        for i, cluster in enumerate(clusters):
            labels[cluster] = matching[i]

        metrics = {
            'adjusted_rand_score': adjusted_rand_score(ground_truth['labels'], labels),
            'number_of_clusters': len(clusters),
            'expected_clusters': len(ground_truth['centers']),
            'cluster_sizes': [len(c) for c in clusters],
            'cluster_centers_error': self._compute_centers_error(
                df[['X', 'Y']].values, labels, ground_truth['centers']
            )
        }

        return metrics

    def _compute_centers_error(self,
                               points: np.ndarray,
                               labels: np.ndarray,
                               true_centers: np.ndarray) -> float:
        """Compute error between predicted and true cluster centers"""
        # Compute predicted centers
        unique_labels = np.unique(labels)
        pred_centers = np.array([
            points[labels == label].mean(axis=0)
            for label in unique_labels
        ])

        # Compute minimum matching distance between predicted and true centers
        distances = cdist(pred_centers, true_centers)
        total_error = 0

        # For each predicted center, find the closest true center
        for dist_row in distances:
            if len(dist_row) > 0:  # Check if there are any true centers to compare to
                total_error += np.min(dist_row)

        return total_error / len(pred_centers) if len(pred_centers) > 0 else float('inf')

    def _compute_overall_metrics(self, timepoint_results: Dict) -> Dict:
        """Compute overall test metrics"""
        metrics = {
            'mean_rand_score': np.mean([r['adjusted_rand_score'] for r in timepoint_results.values()]),
            'cluster_number_accuracy': np.mean([
                r['number_of_clusters'] == r['expected_clusters']
                for r in timepoint_results.values()
            ]),
            'mean_centers_error': np.mean([r['cluster_centers_error'] for r in timepoint_results.values()]),
            'temporal_stability': self._compute_temporal_stability(timepoint_results)
        }

        return metrics

    def _compute_temporal_stability(self, timepoint_results: Dict) -> float:
        """Compute temporal stability metric"""
        # Get list of cluster counts over time
        cluster_counts = [r['number_of_clusters'] for r in timepoint_results.values()]

        # Compute standard deviation of cluster counts
        stability = 1.0 / (1.0 + np.std(cluster_counts))

        return stability

    def visualize_results(self, test_name: str = 'synthetic'):
        """Visualize test results"""
        results = self.results[test_name]

        # Plot metrics over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Adjusted Rand Score
        times = sorted(results['timepoint_results'].keys())
        rand_scores = [results['timepoint_results'][t]['adjusted_rand_score'] for t in times]
        axes[0, 0].plot(times, rand_scores)
        axes[0, 0].set_title('Adjusted Rand Score over Time')
        axes[0, 0].set_ylim([-0.1, 1.1])

        # Number of Clusters
        n_clusters = [results['timepoint_results'][t]['number_of_clusters'] for t in times]
        expected_clusters = [results['timepoint_results'][t]['expected_clusters'] for t in times]
        axes[0, 1].plot(times, n_clusters, label='Detected')
        axes[0, 1].plot(times, expected_clusters, label='Expected')
        axes[0, 1].set_title('Number of Clusters')
        axes[0, 1].legend()

        # Centers Error
        centers_error = [results['timepoint_results'][t]['cluster_centers_error'] for t in times]
        axes[1, 0].plot(times, centers_error)
        axes[1, 0].set_title('Cluster Centers Error')

        # Overall Metrics
        metrics = results['overall_metrics']
        axes[1, 1].bar(range(len(metrics)), list(metrics.values()))
        axes[1, 1].set_xticks(range(len(metrics)))
        axes[1, 1].set_xticklabels(list(metrics.keys()), rotation=45)
        axes[1, 1].set_title('Overall Metrics')

        plt.tight_layout()
        plt.show()


class TestScenarios:
    """Predefined test scenarios for clustering evaluation"""

    @staticmethod
    def test_split_scenario():
        """Generate test data with cluster splitting"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        # Initial single cluster
        center = np.array([0, 0])

        for t in range(n_timepoints):
            if t < 2:
                # Single cluster
                X, y = make_blobs(n_samples=n_points,
                                  centers=[center],
                                  cluster_std=0.5)
                centers = np.array([center])
            else:
                # Split into two clusters
                centers = np.array([center + [2, 0], center - [2, 0]])
                X, y = make_blobs(n_samples=n_points,
                                  centers=centers,
                                  cluster_std=0.5)

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': centers,
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth

    @staticmethod
    def test_merge_scenario():
        """Generate test data with cluster merging"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        # Initial two clusters
        centers = np.array([[2, 0], [-2, 0]])

        for t in range(n_timepoints):
            if t < 2:
                # Two separate clusters
                X, y = make_blobs(n_samples=n_points,
                                  centers=centers,
                                  cluster_std=0.5)
                current_centers = centers
            else:
                # Merged into one cluster
                X, y = make_blobs(n_samples=n_points,
                                  centers=[np.array([0, 0])],
                                  cluster_std=0.5)
                current_centers = np.array([[0, 0]])

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': current_centers,
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth

    @staticmethod
    def test_density_variation():
        """Generate test data with varying densities"""
        n_points = 1000
        n_timepoints = 5

        all_data = []
        ground_truth = {}

        centers = np.array([[0, 0], [5, 5], [-5, -5]])
        cluster_stds = [0.3, 0.7, 1.0]  # Different spreads

        for t in range(n_timepoints):
            # Generate clusters with different densities
            X = np.vstack([
                np.random.normal(center, std, (n_points // 3, 2))
                for center, std in zip(centers, cluster_stds)
            ])
            y = np.repeat(np.arange(len(centers)), n_points // 3)

            df = pd.DataFrame(X, columns=['X', 'Y'])
            df['Time'] = t
            all_data.append(df)

            ground_truth[t] = {
                'centers': centers,
                'labels': y
            }

        return pd.concat(all_data, ignore_index=True), ground_truth




# Add this to your run_comprehensive_tests function:
def run_comprehensive_tests(tracker):
    """Run comprehensive tests on all scenarios"""
    scenarios = {
        'split': TestScenarios.test_split_scenario,
        'merge': TestScenarios.test_merge_scenario,
        'density': TestScenarios.test_density_variation
    }

    test_framework = ClusteringTestFramework()
    all_results = {}

    for scenario_name, scenario_func in scenarios.items():
        print(f"\nTesting {scenario_name} scenario:")

        # Generate test data
        test_data, ground_truth = scenario_func()

        # Add to test framework
        test_framework.test_cases[scenario_name] = {
            'data': test_data,
            'ground_truth': ground_truth
        }

        # Run tests
        results = test_framework.run_tests(tracker)
        all_results[scenario_name] = results[scenario_name]

        # Visualize results
        test_framework.visualize_results(scenario_name)

        # Add the cluster evolution visualization
        print("\nPlotting cluster evolution...")
        # Plot all time steps
        test_framework.plot_cluster_evolution_with_points(scenario_name)
        # Plot specific time steps
        test_framework.plot_cluster_evolution_with_points(scenario_name, time_steps=[0, 2, 4])

        # Print summary
        print(f"\nResults for {scenario_name}:")
        print("Overall metrics:")
        for metric, value in results[scenario_name]['overall_metrics'].items():
            print(f"{metric}: {value:.3f}")

    return all_results


def generate_clustering_report(results, output_filename="clustering_analysis.pdf"):
    """
    Generate a PDF report from clustering analysis results

    Parameters:
    results (dict): Dictionary containing clustering results
    output_filename (str): Name of output PDF file
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Temporal Clustering Analysis Report", title_style))
    story.append(Spacer(1, 12))

    # For each scenario
    for scenario, result in results.items():
        # Scenario Header
        story.append(Paragraph(f"{scenario.upper()} Scenario", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Overall Metrics Table
        metrics_data = [['Metric', 'Value']]
        for metric, value in result['overall_metrics'].items():
            metrics_data.append([metric.replace('_', ' ').title(), f"{value:.3f}"])

        metrics_table = Table(metrics_data, colWidths=[4 * inch, 2 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Generate plots for this scenario
        plot_buffer = generate_scenario_plots(result)
        story.append(Image(plot_buffer, width=7 * inch, height=5 * inch))
        story.append(Spacer(1, 30))

    # Build the PDF
    doc.build(story)


def generate_scenario_plots(result):
    """Generate plots for a scenario and return as bytes buffer"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Get time points and metrics
    times = sorted(result['timepoint_results'].keys())

    # Adjusted Rand Score
    rand_scores = [result['timepoint_results'][t]['adjusted_rand_score'] for t in times]
    axes[0, 0].plot(times, rand_scores)
    axes[0, 0].set_title('Adjusted Rand Score over Time')
    axes[0, 0].set_ylim([-0.1, 1.1])

    # Number of Clusters
    n_clusters = [result['timepoint_results'][t]['number_of_clusters'] for t in times]
    expected_clusters = [result['timepoint_results'][t]['expected_clusters'] for t in times]
    axes[0, 1].plot(times, n_clusters, label='Detected')
    axes[0, 1].plot(times, expected_clusters, label='Expected')
    axes[0, 1].set_title('Number of Clusters')
    axes[0, 1].legend()

    # Centers Error
    centers_error = [result['timepoint_results'][t]['cluster_centers_error'] for t in times]
    axes[1, 0].plot(times, centers_error)
    axes[1, 0].set_title('Cluster Centers Error')

    # Overall Metrics
    metrics = result['overall_metrics']
    axes[1, 1].bar(range(len(metrics)), list(metrics.values()))
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(list(metrics.keys()), rotation=45)
    axes[1, 1].set_title('Overall Metrics')

    plt.tight_layout()

    # Save plot to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()

    return buffer



if __name__ == "__main__":
    # Create an instance of your clustering tracker
    tracker = ImprovedTemporalClusterTracker()

    # Run all tests
    results = run_comprehensive_tests(tracker)
    generate_clustering_report(results, "../clustering_analysis.pdf")
    # Example of accessing results
    print("\nFinal Summary:")
    for scenario, result in results.items():
        print(f"\n{scenario.upper()} scenario:")
        print(f"Mean Rand Score: {result['overall_metrics']['mean_rand_score']:.3f}")
        print(f"Cluster Number Accuracy: {result['overall_metrics']['cluster_number_accuracy']:.3f}")
        print(f"Temporal Stability: {result['overall_metrics']['temporal_stability']:.3f}")