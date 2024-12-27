import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import squareform, pdist
from scipy.stats import gmean, pearsonr, spearmanr, stats
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns

@dataclass
class IntensityMetrics:
    """Stores intensity analysis results"""
    arithmetic_mean: float
    geometric_mean: float
    total_intensity: float
    intensity_std: float
    intensity_per_area: float
    radial_gradient: float
    normalized_radial_gradient: float
    max_distance: float
    intensity_profile: np.ndarray

@dataclass
class MorphologyMetrics:
    """Stores morphological analysis results"""
    area: float  # Total area of cluster
    perimeter: float  # Perimeter length
    circularity: float  # 4π * area / perimeter²
    cell_count: int  # Number of cells
    density: float  # Cells per unit area
    compactness: float  # Area to perimeter ratio
    center_x: float  # X coordinate of centroid
    center_y: float  # Y coordinate of centroid
    radius_of_gyration: float  # Measure of spread from center
    aspect_ratio: float  # Major axis / minor axis
    orientation: float  # Angle of major axis
    convex_hull_area: float  # Area of convex hull
    solidity: float  # Area / convex hull area

class ClusterAnalyzer:
    """
    Unified analyzer for cluster analysis with four main components:
    1. Intensity Analysis: Study light intensity distribution and patterns
    2. Morphological Analysis: Analyze cluster shapes and spatial characteristics
    3. Correlation Analysis: Study relationships between different metrics
    4. Temporal Analysis: Track evolution of cluster properties over time
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with cluster data.

        Args:
            df: DataFrame with columns ['X', 'Y', 'Labels', 'Time', 'IntDen', 'Area']
        """
        self.df = df.copy()
        self._validate_dataframe()

    def _validate_dataframe(self):
        """Validate input DataFrame has required columns."""
        required_columns = ['X', 'Y', 'Labels', 'Time', 'IntDen', 'Area']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _get_cluster_data(self, time_point: int, label: int) -> pd.DataFrame:
        """Get data for specific cluster and time point."""
        return self.df[(self.df['Time'] == time_point) &
                       (self.df['Labels'] == label)]

    # ==================== INTENSITY ANALYSIS ====================

    def analyze_intensity(self, time_point: int, label: int,
                          n_bins: int = 20) -> Optional[IntensityMetrics]:
        """
        Analyze light intensity distribution within a cluster.

        Args:
            time_point: Time point to analyze
            label: Cluster label to analyze
            n_bins: Number of bins for radial intensity profile

        Returns:
            IntensityMetrics object containing intensity analysis results
        """
        cluster_data = self._get_cluster_data(time_point, label)
        if len(cluster_data) < 2:
            return None

        # Basic intensity metrics
        arithmetic_mean = cluster_data['IntDen'].mean()
        geometric_mean = gmean(cluster_data['IntDen'])
        total_intensity = cluster_data['IntDen'].sum()
        intensity_std = cluster_data['IntDen'].std()
        intensity_per_area = total_intensity / cluster_data['Area'].sum()

        # Calculate normalized distances and gradients
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))
        max_distance = distances.max()
        norm_distances = distances / max_distance if max_distance > 0 else distances

        # Calculate both regular and normalized gradients
        radial_gradient = pearsonr(distances, cluster_data['IntDen'])[0]
        norm_radial_gradient = pearsonr(norm_distances, cluster_data['IntDen'])[0]

        # Calculate normalized radial intensity profile
        bins = np.linspace(0, 1, n_bins)
        profile = np.zeros(n_bins - 1)
        for i in range(n_bins - 1):
            mask = (norm_distances >= bins[i]) & (norm_distances < bins[i + 1])
            profile[i] = cluster_data.loc[mask, 'IntDen'].mean() if mask.any() else 0

        return IntensityMetrics(
            arithmetic_mean=arithmetic_mean,
            geometric_mean=geometric_mean,
            total_intensity=total_intensity,
            intensity_std=intensity_std,
            intensity_per_area=intensity_per_area,
            radial_gradient=radial_gradient,
            normalized_radial_gradient=norm_radial_gradient,
            max_distance=max_distance,
            intensity_profile=profile
        )

    def compare_intensities(self, time_point: int, label1: int, label2: int,
                            n_bins: int = 20) -> Dict[str, float]:
        """
        Compare intensity distributions between two clusters.

        Args:
            time_point: Time point to analyze
            label1, label2: Cluster labels to compare
            n_bins: Number of bins for profile comparison

        Returns:
            Dictionary containing comparison metrics
        """
        metrics1 = self.analyze_intensity(time_point, label1, n_bins)
        metrics2 = self.analyze_intensity(time_point, label2, n_bins)

        if metrics1 is None or metrics2 is None:
            return {}

        # Calculate relative metrics
        comparison = {
            'relative_mean': metrics1.arithmetic_mean / metrics2.arithmetic_mean,
            'relative_total': metrics1.total_intensity / metrics2.total_intensity,
            'gradient_difference': metrics1.normalized_radial_gradient - metrics2.normalized_radial_gradient,
            'profile_correlation': pearsonr(metrics1.intensity_profile,
                                            metrics2.intensity_profile)[0]
        }

        # Add distribution overlap
        cluster1 = self._get_cluster_data(time_point, label1)
        cluster2 = self._get_cluster_data(time_point, label2)
        comparison['intensity_overlap'] = self._calculate_distribution_overlap(
            cluster1['IntDen'], cluster2['IntDen'])

        return comparison

    def plot_intensity_analysis(self, time_point: int, label: int,
                                include_comparison: Optional[int] = None):
        """
        Create comprehensive visualization of intensity analysis.

        Args:
            time_point: Time point to analyze
            label: Primary cluster label to analyze
            include_comparison: Optional second cluster label for comparison
        """
        metrics = self.analyze_intensity(time_point, label)
        if metrics is None:
            return None

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Radial intensity profile
        ax = axes[0, 0]
        bins = np.linspace(0, 1, len(metrics.intensity_profile) + 1)[:-1]
        ax.plot(bins, metrics.intensity_profile, label=f'Cluster {label}')
        if include_comparison is not None:
            metrics2 = self.analyze_intensity(time_point, include_comparison)
            if metrics2:
                ax.plot(bins, metrics2.intensity_profile,
                        label=f'Cluster {include_comparison}')
        ax.set_xlabel('Normalized Distance from Center')
        ax.set_ylabel('Mean Intensity')
        ax.set_title('Radial Intensity Profile')
        ax.legend()

        # Plot 2: Intensity distribution
        ax = axes[0, 1]
        cluster_data = self._get_cluster_data(time_point, label)
        ax.hist(cluster_data['IntDen'], bins=30, density=True, alpha=0.6,
                label=f'Cluster {label}')
        if include_comparison is not None:
            cluster2 = self._get_cluster_data(time_point, include_comparison)
            ax.hist(cluster2['IntDen'], bins=30, density=True, alpha=0.6,
                    label=f'Cluster {include_comparison}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title('Intensity Distribution')
        ax.legend()

        # Plot 3: Spatial intensity map
        ax = axes[1, 0]
        scatter = ax.scatter(cluster_data['X'], cluster_data['Y'],
                             c=cluster_data['IntDen'], cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Intensity')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Spatial Intensity Distribution')

        # Plot 4: Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = (
            f'Arithmetic Mean: {metrics.arithmetic_mean:.2f}\n'
            f'Geometric Mean: {metrics.geometric_mean:.2f}\n'
            f'Total Intensity: {metrics.total_intensity:.2f}\n'
            f'Intensity StdDev: {metrics.intensity_std:.2f}\n'
            f'Intensity/Area: {metrics.intensity_per_area:.2f}\n'
            f'Radial Gradient: {metrics.radial_gradient:.2f}\n'
            f'Norm. Radial Gradient: {metrics.normalized_radial_gradient:.2f}'
        )
        ax.text(0.1, 0.5, metrics_text, fontsize=12, va='center')

        plt.tight_layout()
        return fig

    def _calculate_distribution_overlap(self, dist1: np.ndarray,
                                        dist2: np.ndarray, n_bins: int = 50) -> float:
        """Calculate overlap between two distributions using histogram intersection."""
        hist1, edges = np.histogram(dist1, bins=n_bins, density=True)
        hist2, _ = np.histogram(dist2, bins=edges, density=True)
        return np.sum(np.minimum(hist1, hist2)) * (edges[1] - edges[0])

    # ==================== MORPHOLOGICAL ANALYSIS ====================

    def analyze_morphology(self, time_point: int, label: int) -> Optional[MorphologyMetrics]:
        """
        Analyze morphological characteristics of a cluster.

        Args:
            time_point: Time point to analyze
            label: Cluster label to analyze

        Returns:
            MorphologyMetrics object containing shape analysis results
        """
        cluster_data = self._get_cluster_data(time_point, label)
        if len(cluster_data) < 3:
            return None

        # Get points and calculate basic metrics
        points = cluster_data[['X', 'Y']].values
        hull = ConvexHull(points)
        convex_hull_area = hull.area
        perimeter = self._calculate_perimeter(points[hull.vertices])
        area = self._calculate_alpha_shape_area(points)

        # Calculate center and radius of gyration
        center = cluster_data[['X', 'Y']].mean()
        r_gyr = np.sqrt(np.mean(((points - center.values) ** 2).sum(axis=1)))

        # Calculate orientation and aspect ratio using PCA
        pca = self._calculate_pca_metrics(points)

        return MorphologyMetrics(
            area=area,
            perimeter=perimeter,
            circularity=4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
            cell_count=len(cluster_data),
            density=len(cluster_data) / area if area > 0 else 0,
            compactness=area / perimeter if perimeter > 0 else 0,
            center_x=center['X'],
            center_y=center['Y'],
            radius_of_gyration=r_gyr,
            aspect_ratio=pca['aspect_ratio'],
            orientation=pca['orientation'],
            convex_hull_area=convex_hull_area,
            solidity=area / convex_hull_area if convex_hull_area > 0 else 0
        )

    def compare_morphology(self, time_point: int, label1: int, label2: int) -> Dict[str, float]:
        """
        Compare morphological characteristics between two clusters.

        Args:
            time_point: Time point to analyze
            label1, label2: Cluster labels to compare

        Returns:
            Dictionary containing comparison metrics
        """
        metrics1 = self.analyze_morphology(time_point, label1)
        metrics2 = self.analyze_morphology(time_point, label2)

        if metrics1 is None or metrics2 is None:
            return {}

        # Calculate relative metrics and differences
        comparison = {
            'relative_area': metrics1.area / metrics2.area,
            'relative_density': metrics1.density / metrics2.density,
            'circularity_difference': metrics1.circularity - metrics2.circularity,
            'compactness_difference': metrics1.compactness - metrics2.compactness,
            'orientation_difference': self._calculate_angle_difference(
                metrics1.orientation, metrics2.orientation),
            'aspect_ratio_difference': metrics1.aspect_ratio - metrics2.aspect_ratio,
            'solidity_difference': metrics1.solidity - metrics2.solidity,
            'center_distance': np.sqrt(
                (metrics1.center_x - metrics2.center_x) ** 2 +
                (metrics1.center_y - metrics2.center_y) ** 2
            )
        }

        return comparison

    def plot_morphology_analysis(self, time_point: int, label: int,
                                 include_comparison: Optional[int] = None):
        """
        Create comprehensive visualization of morphological analysis.

        Args:
            time_point: Time point to analyze
            label: Primary cluster label to analyze
            include_comparison: Optional second cluster label for comparison
        """
        metrics = self.analyze_morphology(time_point, label)
        if metrics is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Spatial distribution with convex hull
        ax = axes[0, 0]
        cluster_data = self._get_cluster_data(time_point, label)
        points = cluster_data[['X', 'Y']].values
        hull = ConvexHull(points)

        # Plot points and hull
        ax.scatter(points[:, 0], points[:, 1], label=f'Cluster {label}')
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)

        if include_comparison is not None:
            comp_data = self._get_cluster_data(time_point, include_comparison)
            comp_points = comp_data[['X', 'Y']].values
            comp_hull = ConvexHull(comp_points)
            ax.scatter(comp_points[:, 0], comp_points[:, 1],
                       label=f'Cluster {include_comparison}')
            for simplex in comp_hull.simplices:
                ax.plot(comp_points[simplex, 0], comp_points[simplex, 1],
                        'r-', alpha=0.5)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Spatial Distribution and Convex Hull')
        ax.legend()

        # Plot 2: Density heatmap
        ax = axes[0, 1]
        x = points[:, 0]
        y = points[:, 1]
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        scatter = ax.scatter(x, y, c=z, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Density Distribution')

        # Plot 3: Principal axes
        ax = axes[1, 0]
        pca = self._calculate_pca_metrics(points)
        center = points.mean(axis=0)

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], alpha=0.5)

        # Plot principal axes
        for eigenvec, eigenval in zip(pca['eigenvectors'], pca['eigenvalues']):
            ax.arrow(center[0], center[1],
                     eigenvec[0] * np.sqrt(eigenval),
                     eigenvec[1] * np.sqrt(eigenval),
                     head_width=0.1, head_length=0.1, fc='k', ec='k')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Principal Axes')

        # Plot 4: Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = (
            f'Area: {metrics.area:.2f}\n'
            f'Perimeter: {metrics.perimeter:.2f}\n'
            f'Circularity: {metrics.circularity:.2f}\n'
            f'Cell Count: {metrics.cell_count}\n'
            f'Density: {metrics.density:.2f}\n'
            f'Compactness: {metrics.compactness:.2f}\n'
            f'Radius of Gyration: {metrics.radius_of_gyration:.2f}\n'
            f'Aspect Ratio: {metrics.aspect_ratio:.2f}\n'
            f'Orientation: {np.degrees(metrics.orientation):.1f}°\n'
            f'Solidity: {metrics.solidity:.2f}'
        )
        ax.text(0.1, 0.5, metrics_text, fontsize=12, va='center')

        plt.tight_layout()
        return fig

    def _calculate_pca_metrics(self, points: np.ndarray) -> Dict:
        """Calculate PCA-based shape metrics."""
        # Center the points
        centered_points = points - points.mean(axis=0)

        # Calculate covariance matrix and its eigenvectors/values
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate metrics
        aspect_ratio = np.sqrt(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 1
        orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'aspect_ratio': aspect_ratio,
            'orientation': orientation
        }

    def _calculate_alpha_shape_area(self, points: np.ndarray, alpha: float = None) -> float:
        """
        Calculate area using alpha shape (or convex hull if alpha is None).
        Alpha shapes provide a more accurate area for non-convex shapes.
        """
        if alpha is None or len(points) < 4:
            return ConvexHull(points).area

        # Implementation of alpha shape calculation would go here
        # For now, return convex hull area
        return ConvexHull(points).area

    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate the minimum angle between two orientations."""
        diff = abs(angle1 - angle2) % np.pi
        return min(diff, np.pi - diff)

    # ==================== CORRELATION ANALYSIS ====================

    def analyze_correlation(self, time_point: int, label: int) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlations between different metrics within a cluster.

        Args:
            time_point: Time point to analyze
            label: Cluster label to analyze

        Returns:
            Dictionary containing various correlation measures and statistical analyses
        """
        cluster_data = self._get_cluster_data(time_point, label)
        if len(cluster_data) < 3:
            return {}

        # Calculate centers and distances
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        # Calculate local densities
        kde = gaussian_kde(cluster_data[['X', 'Y']].values.T)
        local_densities = kde(cluster_data[['X', 'Y']].values.T)

        # Basic correlations
        spatial_correlations = {
            'intensity_area': {
                'pearson': pearsonr(cluster_data['IntDen'], cluster_data['Area'])[0],
                'spearman': spearmanr(cluster_data['IntDen'], cluster_data['Area'])[0]
            },
            'intensity_distance': {
                'pearson': pearsonr(cluster_data['IntDen'], distances)[0],
                'spearman': spearmanr(cluster_data['IntDen'], distances)[0]
            },
            'intensity_density': {
                'pearson': pearsonr(cluster_data['IntDen'], local_densities)[0],
                'spearman': spearmanr(cluster_data['IntDen'], local_densities)[0]
            },
            'area_density': {
                'pearson': pearsonr(cluster_data['Area'], local_densities)[0],
                'spearman': spearmanr(cluster_data['Area'], local_densities)[0]
            }
        }

        # Calculate spatial autocorrelation
        spatial_autocorr = self._calculate_spatial_autocorrelation(
            cluster_data[['X', 'Y']].values,
            cluster_data['IntDen'].values
        )

        # Calculate distance-based correlations
        distance_correlations = self._calculate_distance_correlations(cluster_data)

        return {
            'spatial_correlations': spatial_correlations,
            'spatial_autocorrelation': spatial_autocorr,
            'distance_correlations': distance_correlations
        }

    def compare_correlations(self, time_point: int, label1: int, label2: int) -> Dict:
        """
        Compare correlation patterns between two clusters.

        Args:
            time_point: Time point to analyze
            label1, label2: Cluster labels to compare
        """
        corr1 = self.analyze_correlation(time_point, label1)
        corr2 = self.analyze_correlation(time_point, label2)

        if not corr1 or not corr2:
            return {}

        # Calculate differences in correlation patterns
        correlation_differences = {}
        for metric, values in corr1['spatial_correlations'].items():
            correlation_differences[metric] = {
                'pearson_diff': abs(values['pearson'] -
                                    corr2['spatial_correlations'][metric]['pearson']),
                'spearman_diff': abs(values['spearman'] -
                                     corr2['spatial_correlations'][metric]['spearman'])
            }

        return {
            'correlation_differences': correlation_differences,
            'autocorr_difference': abs(corr1['spatial_autocorrelation'] -
                                       corr2['spatial_autocorrelation']),
            'distance_correlation_difference': self._compare_distance_correlations(
                corr1['distance_correlations'],
                corr2['distance_correlations']
            )
        }

    def plot_correlation_analysis(self, time_point: int, label: int,
                                  include_comparison: Optional[int] = None):
        """
        Create comprehensive visualization of correlation analysis.

        Args:
            time_point: Time point to analyze
            label: Primary cluster label to analyze
            include_comparison: Optional second cluster label for comparison
        """
        correlations = self.analyze_correlation(time_point, label)
        if not correlations:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Intensity vs Distance scatter
        ax = axes[0, 0]
        cluster_data = self._get_cluster_data(time_point, label)
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        ax.scatter(distances, cluster_data['IntDen'], alpha=0.5,
                   label=f'Cluster {label}')
        if include_comparison is not None:
            comp_data = self._get_cluster_data(time_point, include_comparison)
            comp_center = comp_data[['X', 'Y']].mean()
            comp_distances = np.sqrt(((comp_data[['X', 'Y']] - comp_center) ** 2).sum(axis=1))
            ax.scatter(comp_distances, comp_data['IntDen'], alpha=0.5,
                       label=f'Cluster {include_comparison}')
        ax.set_xlabel('Distance from Center')
        ax.set_ylabel('Intensity')
        ax.set_title('Intensity vs Distance')
        ax.legend()

        # Plot 2: Correlation matrix heatmap
        ax = axes[0, 1]
        cluster_data = self._get_cluster_data(time_point, label)
        corr_matrix = cluster_data[['IntDen', 'Area']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')

        # Plot 3: Spatial autocorrelation
        ax = axes[1, 0]
        self._plot_spatial_autocorrelation(cluster_data, ax)
        ax.set_title('Spatial Autocorrelation')

        # Plot 4: Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = []
        for metric, values in correlations['spatial_correlations'].items():
            metrics_text.append(f"{metric}:")
            metrics_text.append(f"  Pearson: {values['pearson']:.3f}")
            metrics_text.append(f"  Spearman: {values['spearman']:.3f}")
        metrics_text.append(f"\nSpatial Autocorrelation: {correlations['spatial_autocorrelation']:.3f}")

        ax.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=12, va='center')

        plt.tight_layout()
        return fig

    def _calculate_spatial_autocorrelation(self, coordinates: np.ndarray,
                                           values: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation."""
        n = len(coordinates)
        if n < 2:
            return 0.0

        # Calculate distance matrix
        distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))

        # Create weight matrix (inverse distance)
        np.fill_diagonal(distances, np.inf)  # Avoid division by zero
        weights = 1 / distances

        # Normalize weights
        weights /= weights.sum(axis=1)[:, None]
        np.fill_diagonal(weights, 0)

        # Calculate Moran's I
        mean = values.mean()
        numerator = np.sum(weights * np.outer(values - mean, values - mean))
        denominator = np.sum((values - mean) ** 2)

        return (n * numerator) / (np.sum(weights) * denominator)

    def _calculate_distance_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between distance-based metrics."""
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        # Calculate nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
        nn_distances, _ = nbrs.kneighbors(cluster_data[['X', 'Y']])

        return {
            'nn_intensity_corr': pearsonr(nn_distances[:, 1],
                                          cluster_data['IntDen'])[0],
            'nn_area_corr': pearsonr(nn_distances[:, 1],
                                     cluster_data['Area'])[0]
        }

    def _compare_distance_correlations(self, corr1: Dict, corr2: Dict) -> Dict[str, float]:
        """Compare distance-based correlations between clusters."""
        return {key: abs(corr1[key] - corr2[key])
                for key in corr1.keys()}

    def _plot_spatial_autocorrelation(self, cluster_data: pd.DataFrame, ax: plt.Axes):
        """Plot spatial autocorrelation analysis."""
        coordinates = cluster_data[['X', 'Y']].values
        values = cluster_data['IntDen'].values

        # Calculate local Moran's I
        local_morans = np.zeros(len(coordinates))
        for i in range(len(coordinates)):
            distances = np.sqrt(((coordinates - coordinates[i]) ** 2).sum(axis=1))
            weights = 1 / distances
            weights[i] = 0  # Zero weight for self
            weights /= weights.sum()

            z_i = values[i] - values.mean()
            z_j = values - values.mean()
            local_morans[i] = z_i * np.sum(weights * z_j)

        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                             c=local_morans, cmap='RdBu')
        plt.colorbar(scatter, ax=ax, label="Local Moran's I")
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

    def _calculate_distribution_overlap(self, dist1: np.ndarray,
                                        dist2: np.ndarray, n_bins: int = 50) -> float:
        """Calculate overlap between two distributions using histogram intersection."""
        hist1, edges = np.histogram(dist1, bins=n_bins, density=True)
        hist2, _ = np.histogram(dist2, bins=edges, density=True)
        return np.sum(np.minimum(hist1, hist2)) * (edges[1] - edges[0])

    # ==================== CORRELATION ANALYSIS ====================

    def analyze_correlation(self, time_point: int, label: int) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlations between different metrics within a cluster.

        Args:
            time_point: Time point to analyze
            label: Cluster label to analyze

        Returns:
            Dictionary containing various correlation measures and statistical analyses
        """
        cluster_data = self._get_cluster_data(time_point, label)
        if len(cluster_data) < 3:
            return {}

        # Calculate centers and distances
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        # Calculate local densities
        kde = gaussian_kde(cluster_data[['X', 'Y']].values.T)
        local_densities = kde(cluster_data[['X', 'Y']].values.T)

        # Basic correlations
        spatial_correlations = {
            'intensity_area': {
                'pearson': pearsonr(cluster_data['IntDen'], cluster_data['Area'])[0],
                'spearman': spearmanr(cluster_data['IntDen'], cluster_data['Area'])[0]
            },
            'intensity_distance': {
                'pearson': pearsonr(cluster_data['IntDen'], distances)[0],
                'spearman': spearmanr(cluster_data['IntDen'], distances)[0]
            },
            'intensity_density': {
                'pearson': pearsonr(cluster_data['IntDen'], local_densities)[0],
                'spearman': spearmanr(cluster_data['IntDen'], local_densities)[0]
            },
            'area_density': {
                'pearson': pearsonr(cluster_data['Area'], local_densities)[0],
                'spearman': spearmanr(cluster_data['Area'], local_densities)[0]
            }
        }

        # Calculate spatial autocorrelation
        spatial_autocorr = self._calculate_spatial_autocorrelation(
            cluster_data[['X', 'Y']].values,
            cluster_data['IntDen'].values
        )

        # Calculate distance-based correlations
        distance_correlations = self._calculate_distance_correlations(cluster_data)

        return {
            'spatial_correlations': spatial_correlations,
            'spatial_autocorrelation': spatial_autocorr,
            'distance_correlations': distance_correlations
        }

    def compare_correlations(self, time_point: int, label1: int, label2: int) -> Dict:
        """
        Compare correlation patterns between two clusters.

        Args:
            time_point: Time point to analyze
            label1, label2: Cluster labels to compare
        """
        corr1 = self.analyze_correlation(time_point, label1)
        corr2 = self.analyze_correlation(time_point, label2)

        if not corr1 or not corr2:
            return {}

        # Calculate differences in correlation patterns
        correlation_differences = {}
        for metric, values in corr1['spatial_correlations'].items():
            correlation_differences[metric] = {
                'pearson_diff': abs(values['pearson'] -
                                    corr2['spatial_correlations'][metric]['pearson']),
                'spearman_diff': abs(values['spearman'] -
                                     corr2['spatial_correlations'][metric]['spearman'])
            }

        return {
            'correlation_differences': correlation_differences,
            'autocorr_difference': abs(corr1['spatial_autocorrelation'] -
                                       corr2['spatial_autocorrelation']),
            'distance_correlation_difference': self._compare_distance_correlations(
                corr1['distance_correlations'],
                corr2['distance_correlations']
            )
        }

    def plot_correlation_analysis(self, time_point: int, label: int,
                                  include_comparison: Optional[int] = None):
        """
        Create comprehensive visualization of correlation analysis.

        Args:
            time_point: Time point to analyze
            label: Primary cluster label to analyze
            include_comparison: Optional second cluster label for comparison
        """
        correlations = self.analyze_correlation(time_point, label)
        if not correlations:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Intensity vs Distance scatter
        ax = axes[0, 0]
        cluster_data = self._get_cluster_data(time_point, label)
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        ax.scatter(distances, cluster_data['IntDen'], alpha=0.5,
                   label=f'Cluster {label}')
        if include_comparison is not None:
            comp_data = self._get_cluster_data(time_point, include_comparison)
            comp_center = comp_data[['X', 'Y']].mean()
            comp_distances = np.sqrt(((comp_data[['X', 'Y']] - comp_center) ** 2).sum(axis=1))
            ax.scatter(comp_distances, comp_data['IntDen'], alpha=0.5,
                       label=f'Cluster {include_comparison}')
        ax.set_xlabel('Distance from Center')
        ax.set_ylabel('Intensity')
        ax.set_title('Intensity vs Distance')
        ax.legend()

        # Plot 2: Correlation matrix heatmap
        ax = axes[0, 1]
        cluster_data = self._get_cluster_data(time_point, label)
        corr_matrix = cluster_data[['IntDen', 'Area']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')

        # Plot 3: Spatial autocorrelation
        ax = axes[1, 0]
        self._plot_spatial_autocorrelation(cluster_data, ax)
        ax.set_title('Spatial Autocorrelation')

        # Plot 4: Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = []
        for metric, values in correlations['spatial_correlations'].items():
            metrics_text.append(f"{metric}:")
            metrics_text.append(f"  Pearson: {values['pearson']:.3f}")
            metrics_text.append(f"  Spearman: {values['spearman']:.3f}")
        metrics_text.append(f"\nSpatial Autocorrelation: {correlations['spatial_autocorrelation']:.3f}")

        ax.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=12, va='center')

        plt.tight_layout()
        return fig

    def _calculate_spatial_autocorrelation(self, coordinates: np.ndarray,
                                           values: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation."""
        n = len(coordinates)
        if n < 2:
            return 0.0

        # Calculate distance matrix
        distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))

        # Create weight matrix (inverse distance)
        np.fill_diagonal(distances, np.inf)  # Avoid division by zero
        weights = 1 / distances

        # Normalize weights
        weights /= weights.sum(axis=1)[:, None]
        np.fill_diagonal(weights, 0)

        # Calculate Moran's I
        mean = values.mean()
        numerator = np.sum(weights * np.outer(values - mean, values - mean))
        denominator = np.sum((values - mean) ** 2)

        return (n * numerator) / (np.sum(weights) * denominator)

    def _calculate_distance_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between distance-based metrics."""
        center = cluster_data[['X', 'Y']].mean()
        distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))

        # Calculate nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
        nn_distances, _ = nbrs.kneighbors(cluster_data[['X', 'Y']])

        return {
            'nn_intensity_corr': pearsonr(nn_distances[:, 1],
                                          cluster_data['IntDen'])[0],
            'nn_area_corr': pearsonr(nn_distances[:, 1],
                                     cluster_data['Area'])[0]
        }

    def _compare_distance_correlations(self, corr1: Dict, corr2: Dict) -> Dict[str, float]:
        """Compare distance-based correlations between clusters."""
        return {key: abs(corr1[key] - corr2[key])
                for key in corr1.keys()}

    def _plot_spatial_autocorrelation(self, cluster_data: pd.DataFrame, ax: plt.Axes):
        """Plot spatial autocorrelation analysis."""
        coordinates = cluster_data[['X', 'Y']].values
        values = cluster_data['IntDen'].values

        # Calculate local Moran's I
        local_morans = np.zeros(len(coordinates))
        for i in range(len(coordinates)):
            distances = np.sqrt(((coordinates - coordinates[i]) ** 2).sum(axis=1))
            weights = 1 / distances
            weights[i] = 0  # Zero weight for self
            weights /= weights.sum()

            z_i = values[i] - values.mean()
            z_j = values - values.mean()
            local_morans[i] = z_i * np.sum(weights * z_j)

        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                             c=local_morans, cmap='RdBu')
        plt.colorbar(scatter, ax=ax, label="Local Moran's I")
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

    def _calculate_distribution_overlap(self, dist1: np.ndarray,
                                        dist2: np.ndarray, n_bins: int = 50) -> float:
        """Calculate overlap between two distributions using histogram intersection."""
        hist1, edges = np.histogram(dist1, bins=n_bins, density=True)
        hist2, _ = np.histogram(dist2, bins=edges, density=True)
        return np.sum(np.minimum(hist1, hist2)) * (edges[1] - edges[0])


# ==================== CORRELATION ANALYSIS ====================

def analyze_correlation(self, time_point: int, label: int) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between metrics within a cluster.

    Args:
        time_point: Time point to analyze
        label: Cluster label to analyze

    Returns:
        Dictionary containing correlation measures between different metrics
    """
    cluster_data = self._get_cluster_data(time_point, label)
    if len(cluster_data) < 3:
        return {}

    # Calculate centers and normalized distances
    center = cluster_data[['X', 'Y']].mean()
    distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))
    norm_distances = distances / distances.max() if distances.max() > 0 else distances

    # Calculate local densities
    kde = gaussian_kde(cluster_data[['X', 'Y']].values.T)
    local_densities = kde(cluster_data[['X', 'Y']].values.T)

    # Basic correlations between metrics
    basic_correlations = {
        'intensity_area': pearsonr(cluster_data['IntDen'], cluster_data['Area'])[0],
        'intensity_distance': pearsonr(cluster_data['IntDen'], norm_distances)[0],
        'intensity_density': pearsonr(cluster_data['IntDen'], local_densities)[0],
        'area_density': pearsonr(cluster_data['Area'], local_densities)[0]
    }

    # Spatial correlations
    spatial_correlations = self._calculate_spatial_correlations(cluster_data)

    # Nearest neighbor correlations
    nn_correlations = self._calculate_nn_correlations(cluster_data)

    return {
        'basic_correlations': basic_correlations,
        'spatial_correlations': spatial_correlations,
        'nearest_neighbor': nn_correlations
    }


def compare_correlations(self, time_point: int, label1: int, label2: int) -> Dict[str, float]:
    """
    Compare correlation patterns between two clusters.
    """
    corr1 = self.analyze_correlation(time_point, label1)
    corr2 = self.analyze_correlation(time_point, label2)

    if not corr1 or not corr2:
        return {}

    # Calculate differences in correlation patterns
    differences = {}
    for category in ['basic_correlations', 'spatial_correlations', 'nearest_neighbor']:
        differences[f'{category}_diff'] = {
            key: abs(corr1[category][key] - corr2[category][key])
            for key in corr1[category]
        }

    return differences


def plot_correlation_analysis(self, time_point: int, label: int):
    """
    Create visualization of correlation analysis results.
    """
    correlations = self.analyze_correlation(time_point, label)
    if not correlations:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 1: Basic correlation matrix
    ax = axes[0, 0]
    self._plot_correlation_matrix(correlations['basic_correlations'], ax)

    # Plot 2: Spatial correlations
    ax = axes[0, 1]
    self._plot_spatial_correlations(time_point, label, ax)

    # Plot 3: Nearest neighbor correlations
    ax = axes[1, 0]
    self._plot_nn_correlations(time_point, label, ax)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    self._plot_correlation_summary(correlations, ax)

    plt.tight_layout()
    return fig


def _calculate_spatial_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, ndarray]:
    """Calculate spatial correlation metrics."""
    # Calculate distances between all pairs of points
    points = cluster_data[['X', 'Y']].values
    distances = squareform(pdist(points))
    np.fill_diagonal(distances, np.inf)  # Avoid self-comparisons

    # Calculate correlations based on spatial relationships
    intensity_corr = np.zeros_like(distances)
    area_corr = np.zeros_like(distances)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            intensity_diff = abs(cluster_data['IntDen'].iloc[i] - cluster_data['IntDen'].iloc[j])
            area_diff = abs(cluster_data['Area'].iloc[i] - cluster_data['Area'].iloc[j])
            intensity_corr[i, j] = intensity_diff / distances[i, j]
            area_corr[i, j] = area_diff / distances[i, j]

    return {
        'intensity_spatial_corr': np.nanmean(intensity_corr),
        'area_spatial_corr': np.nanmean(area_corr)
    }


def _calculate_nn_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate nearest neighbor correlations."""
    from sklearn.neighbors import NearestNeighbors

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
    distances, indices = nbrs.kneighbors(cluster_data[['X', 'Y']])

    # Calculate correlations with nearest neighbor properties
    nn_correlations = {
        'nn_intensity': pearsonr(
            cluster_data['IntDen'].values,
            cluster_data['IntDen'].values[indices[:, 1]]
        )[0],
        'nn_area': pearsonr(
            cluster_data['Area'].values,
            cluster_data['Area'].values[indices[:, 1]]
        )[0]
    }

    return nn_correlations


def _plot_correlation_matrix(self, correlations: Dict[str, float], ax: plt.Axes):
    """Plot correlation matrix heatmap."""
    corr_matrix = pd.DataFrame([correlations]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')


def _plot_spatial_correlations(self, time_point: int, label: int, ax: plt.Axes):
    """Plot spatial correlation patterns."""
    cluster_data = self._get_cluster_data(time_point, label)
    points = cluster_data[['X', 'Y']].values

    # Create spatial correlation plot
    scatter = ax.scatter(points[:, 0], points[:, 1],
                         c=cluster_data['IntDen'],
                         cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Intensity')
    ax.set_title('Spatial Correlation Pattern')


def _plot_nn_correlations(self, time_point: int, label: int, ax: plt.Axes):
    """Plot nearest neighbor correlation patterns."""
    cluster_data = self._get_cluster_data(time_point, label)
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
    distances, indices = nbrs.kneighbors(cluster_data[['X', 'Y']])

    # Plot correlation between nearest neighbor intensities
    ax.scatter(cluster_data['IntDen'],
               cluster_data['IntDen'].values[indices[:, 1]],
               alpha=0.5)
    ax.set_xlabel('Cell Intensity')
    ax.set_ylabel('Nearest Neighbor Intensity')
    ax.set_title('Nearest Neighbor Intensity Correlation')


def _plot_correlation_summary(self, correlations: Dict, ax: plt.Axes):
    """Plot summary of correlation metrics."""
    ax.axis('off')
    summary = []

    for category, metrics in correlations.items():
        summary.append(f"\n{category.replace('_', ' ').title()}:")
        for name, value in metrics.items():
            summary.append(f"  {name}: {value:.3f}")

    ax.text(0.1, 0.5, '\n'.join(summary), fontsize=10, va='center')


# ==================== CORRELATION ANALYSIS ====================

def analyze_correlation(self, time_point: int, label: int) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between metrics within a cluster.

    Args:
        time_point: Time point to analyze
        label: Cluster label to analyze

    Returns:
        Dictionary containing correlation measures between different metrics
    """
    cluster_data = self._get_cluster_data(time_point, label)
    if len(cluster_data) < 3:
        return {}

    # Calculate centers and normalized distances
    center = cluster_data[['X', 'Y']].mean()
    distances = np.sqrt(((cluster_data[['X', 'Y']] - center) ** 2).sum(axis=1))
    norm_distances = distances / distances.max() if distances.max() > 0 else distances

    # Calculate local densities
    kde = gaussian_kde(cluster_data[['X', 'Y']].values.T)
    local_densities = kde(cluster_data[['X', 'Y']].values.T)

    # Basic correlations between metrics
    basic_correlations = {
        'intensity_area': pearsonr(cluster_data['IntDen'], cluster_data['Area'])[0],
        'intensity_distance': pearsonr(cluster_data['IntDen'], norm_distances)[0],
        'intensity_density': pearsonr(cluster_data['IntDen'], local_densities)[0],
        'area_density': pearsonr(cluster_data['Area'], local_densities)[0]
    }

    # Spatial correlations
    spatial_correlations = self._calculate_spatial_correlations(cluster_data)

    # Nearest neighbor correlations
    nn_correlations = self._calculate_nn_correlations(cluster_data)

    return {
        'basic_correlations': basic_correlations,
        'spatial_correlations': spatial_correlations,
        'nearest_neighbor': nn_correlations
    }


def compare_correlations(self, time_point: int, label1: int, label2: int) -> Dict[str, float]:
    """
    Compare correlation patterns between two clusters.
    """
    corr1 = self.analyze_correlation(time_point, label1)
    corr2 = self.analyze_correlation(time_point, label2)

    if not corr1 or not corr2:
        return {}

    # Calculate differences in correlation patterns
    differences = {}
    for category in ['basic_correlations', 'spatial_correlations', 'nearest_neighbor']:
        differences[f'{category}_diff'] = {
            key: abs(corr1[category][key] - corr2[category][key])
            for key in corr1[category]
        }

    return differences


def plot_correlation_analysis(self, time_point: int, label: int):
    """
    Create visualization of correlation analysis results.
    """
    correlations = self.analyze_correlation(time_point, label)
    if not correlations:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 1: Basic correlation matrix
    ax = axes[0, 0]
    self._plot_correlation_matrix(correlations['basic_correlations'], ax)

    # Plot 2: Spatial correlations
    ax = axes[0, 1]
    self._plot_spatial_correlations(time_point, label, ax)

    # Plot 3: Nearest neighbor correlations
    ax = axes[1, 0]
    self._plot_nn_correlations(time_point, label, ax)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    self._plot_correlation_summary(correlations, ax)

    plt.tight_layout()
    return fig


def _calculate_spatial_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate spatial correlation metrics."""
    # Calculate distances between all pairs of points
    points = cluster_data[['X', 'Y']].values
    distances = squareform(pdist(points))
    np.fill_diagonal(distances, np.inf)  # Avoid self-comparisons

    # Calculate correlations based on spatial relationships
    intensity_corr = np.zeros_like(distances)
    area_corr = np.zeros_like(distances)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            intensity_diff = abs(cluster_data['IntDen'].iloc[i] - cluster_data['IntDen'].iloc[j])
            area_diff = abs(cluster_data['Area'].iloc[i] - cluster_data['Area'].iloc[j])
            intensity_corr[i, j] = intensity_diff / distances[i, j]
            area_corr[i, j] = area_diff / distances[i, j]

    return {
        'intensity_spatial_corr': np.nanmean(intensity_corr),
        'area_spatial_corr': np.nanmean(area_corr)
    }


def _calculate_nn_correlations(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate nearest neighbor correlations."""
    from sklearn.neighbors import NearestNeighbors

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
    distances, indices = nbrs.kneighbors(cluster_data[['X', 'Y']])

    # Calculate correlations with nearest neighbor properties
    nn_correlations = {
        'nn_intensity': pearsonr(
            cluster_data['IntDen'].values,
            cluster_data['IntDen'].values[indices[:, 1]]
        )[0],
        'nn_area': pearsonr(
            cluster_data['Area'].values,
            cluster_data['Area'].values[indices[:, 1]]
        )[0]
    }

    return nn_correlations


def _plot_correlation_matrix(self, correlations: Dict[str, float], ax: plt.Axes):
    """Plot correlation matrix heatmap."""
    corr_matrix = pd.DataFrame([correlations]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')


def _plot_spatial_correlations(self, time_point: int, label: int, ax: plt.Axes):
    """Plot spatial correlation patterns."""
    cluster_data = self._get_cluster_data(time_point, label)
    points = cluster_data[['X', 'Y']].values

    # Create spatial correlation plot
    scatter = ax.scatter(points[:, 0], points[:, 1],
                         c=cluster_data['IntDen'],
                         cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Intensity')
    ax.set_title('Spatial Correlation Pattern')


def _plot_nn_correlations(self, time_point: int, label: int, ax: plt.Axes):
    """Plot nearest neighbor correlation patterns."""
    cluster_data = self._get_cluster_data(time_point, label)
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data[['X', 'Y']])
    distances, indices = nbrs.kneighbors(cluster_data[['X', 'Y']])

    # Plot correlation between nearest neighbor intensities
    ax.scatter(cluster_data['IntDen'],
               cluster_data['IntDen'].values[indices[:, 1]],
               alpha=0.5)
    ax.set_xlabel('Cell Intensity')
    ax.set_ylabel('Nearest Neighbor Intensity')
    ax.set_title('Nearest Neighbor Intensity Correlation')


def _plot_correlation_summary(self, correlations: Dict, ax: plt.Axes):
    """
    Plot summary of correlation metrics.

    Args:
        correlations: Dictionary containing correlation analysis results
        ax: Matplotlib axes for plotting
    """
    ax.axis('off')
    summary = []

    for category, metrics in correlations.items():
        summary.append(f"\n{category.replace('_', ' ').title()}:")
        for name, value in metrics.items():
            summary.append(f"  {name}: {value:.3f}")

    ax.text(0.1, 0.5, '\n'.join(summary), fontsize=10, va='center')


def _calculate_significance(self, corr_value: float, n: int) -> float:
    """
    Calculate statistical significance of correlation.

    Args:
        corr_value: Correlation coefficient
        n: Sample size

    Returns:
        p-value for correlation significance
    """
    # Calculate t-statistic
    t = corr_value * np.sqrt((n - 2) / (1 - corr_value ** 2))
    # Calculate two-tailed p-value
    return 2 * (1 - stats.t.cdf(abs(t), n - 2))


def _adjust_pvalues(self, pvalues: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg correction for multiple testing.

    Args:
        pvalues: List of p-values to adjust

    Returns:
        List of adjusted p-values
    """
    n = len(pvalues)
    if n == 0:
        return []

    # Sort p-values and get original indices
    sorted_pairs = sorted(enumerate(pvalues), key=lambda x: x[1])
    sorted_indices = [pair[0] for pair in sorted_pairs]
    sorted_pvalues = [pair[1] for pair in sorted_pairs]

    # Calculate adjusted values
    adjusted = []
    for i, p in enumerate(sorted_pvalues):
        adjusted.append(min(1, p * n / (i + 1)))

    # Make sure they're monotonically decreasing
    for i in range(len(adjusted) - 1, 0, -1):
        adjusted[i - 1] = min(adjusted[i - 1], adjusted[i])

    # Return to original order
    original_order = [0] * n
    for i, idx in enumerate(sorted_indices):
        original_order[idx] = adjusted[i]

    return original_order


def _validate_correlations(self, correlations: Dict[str, float],
                           sample_size: int, alpha: float = 0.05) -> Dict[str, bool]:
    """
    Validate correlation significance with multiple testing correction.

    Args:
        correlations: Dictionary of correlation values
        sample_size: Number of samples used for correlation
        alpha: Significance level

    Returns:
        Dictionary indicating which correlations are significant
    """
    # Calculate p-values
    pvalues = [self._calculate_significance(corr, sample_size)
               for corr in correlations.values()]

    # Apply multiple testing correction
    adjusted_pvalues = self._adjust_pvalues(pvalues)

    # Create significance dictionary
    return {name: (adj_p < alpha)
            for name, adj_p in zip(correlations.keys(), adjusted_pvalues)}


def _plot_correlation_significance(self, correlations: Dict[str, float],
                                   significance: Dict[str, bool], ax: plt.Axes):
    """
    Plot significance of correlation values.

    Args:
        correlations: Dictionary of correlation values
        significance: Dictionary of significance results
        ax: Matplotlib axes for plotting
    """
    names = list(correlations.keys())
    values = list(correlations.values())
    colors = ['red' if sig else 'blue' for sig in significance.values()]

    ax.barh(names, values, color=colors)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Correlation Significance')

    # Add significance labels
    for i, sig in enumerate(significance.values()):
        ax.text(0.1, i, '*' if sig else '', va='center', ha='left')

    return ax.figure


# ==================== TEMPORAL ANALYSIS ====================

def analyze_temporal(self, label: int, time_range: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Analyze how a single cluster changes over time.

    Args:
        label: Cluster label to analyze
        time_range: Optional tuple of (start_time, end_time)

    Returns:
        Dictionary containing temporal changes in cluster properties
    """
    # Get cluster data for all time points
    cluster_data = self.df[self.df['Labels'] == label]
    if time_range:
        cluster_data = cluster_data[
            (cluster_data['Time'] >= time_range[0]) &
            (cluster_data['Time'] <= time_range[1])
            ]

    time_points = sorted(cluster_data['Time'].unique())
    if len(time_points) < 2:
        return {}

    # Track metrics over time
    temporal_metrics = {
        'times': time_points,
        'cell_count': [],
        'mean_intensity': [],
        'total_area': [],
        'density': [],
        'center_x': [],
        'center_y': [],
        'radius': []
    }

    for t in time_points:
        time_slice = cluster_data[cluster_data['Time'] == t]

        # Basic metrics
        temporal_metrics['cell_count'].append(len(time_slice))
        temporal_metrics['mean_intensity'].append(time_slice['IntDen'].mean())
        temporal_metrics['total_area'].append(time_slice['Area'].sum())
        temporal_metrics['density'].append(len(time_slice) / time_slice['Area'].sum())

        # Spatial metrics
        center_x = time_slice['X'].mean()
        center_y = time_slice['Y'].mean()
        temporal_metrics['center_x'].append(center_x)
        temporal_metrics['center_y'].append(center_y)

        # Calculate radius (distance from center to furthest point)
        distances = np.sqrt(((time_slice['X'] - center_x) ** 2 +
                             (time_slice['Y'] - center_y) ** 2))
        temporal_metrics['radius'].append(np.max(distances))

    # Calculate rates of change
    rates = self._calculate_rates_of_change(temporal_metrics)
    temporal_metrics['rates'] = rates

    return temporal_metrics


def plot_temporal_analysis(self, label: int, time_range: Optional[Tuple[int, int]] = None):
    """
    Create visualization of how cluster properties change over time.
    """
    metrics = self.analyze_temporal(label, time_range)
    if not metrics:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot 1: Cell count and density over time
    ax = axes[0, 0]
    times = metrics['times']
    ax.plot(times, metrics['cell_count'], 'b-', label='Cell Count')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax2 = ax.twinx()
    ax2.plot(times, metrics['density'], 'r-', label='Density')
    ax2.set_ylabel('Density')
    ax.set_title('Population Changes')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    # Plot 2: Intensity changes
    ax = axes[0, 1]
    ax.plot(times, metrics['mean_intensity'], 'g-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('Intensity Changes')

    # Plot 3: Spatial trajectory
    ax = axes[1, 0]
    ax.plot(metrics['center_x'], metrics['center_y'], 'k-')
    ax.scatter(metrics['center_x'], metrics['center_y'],
               c=range(len(times)), cmap='viridis')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Cluster Movement')
    plt.colorbar(ax.collections[0], ax=ax, label='Time')

    # Plot 4: Area and radius changes
    ax = axes[1, 1]
    ax.plot(times, metrics['total_area'], 'b-', label='Total Area')
    ax2 = ax.twinx()
    ax2.plot(times, metrics['radius'], 'r-', label='Radius')
    ax.set_xlabel('Time')
    ax.set_ylabel('Area')
    ax2.set_ylabel('Radius')
    ax.set_title('Size Changes')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    return fig


def _calculate_rates_of_change(self, metrics: Dict) -> Dict:
    """
    Calculate rates of change for various metrics.

    Args:
        metrics: Dictionary containing temporal metrics

    Returns:
        Dictionary containing rates of change
    """
    times = np.array(metrics['times'])
    dt = np.diff(times)

    rates = {}

    # Growth rate (relative change in cell count)
    counts = np.array(metrics['cell_count'])
    rates['growth_rate'] = np.mean(np.diff(np.log(counts + 1)) / dt)

    # Intensity change rate
    intensities = np.array(metrics['mean_intensity'])
    rates['intensity_change'] = np.mean(np.diff(intensities) / dt)

    # Area expansion rate
    areas = np.array(metrics['total_area'])
    rates['area_change'] = np.mean(np.diff(areas) / dt)

    # Migration speed (distance moved per time)
    dx = np.diff(metrics['center_x'])
    dy = np.diff(metrics['center_y'])
    displacement = np.sqrt(dx ** 2 + dy ** 2)
    rates['migration_speed'] = np.mean(displacement / dt)

    return rates