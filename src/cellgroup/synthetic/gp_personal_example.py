import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class SyntheticNucleus:
    def __init__(
            self,
            centroid: Tuple[float, float],
            radii: Tuple[float, float],
            orientation: float = 0,
            intensity_mean: float = 1000,
            intensity_std: float = 200
    ):
        """
        Create a synthetic nucleus with elliptical shape and Gaussian intensity distribution.

        Args:
            centroid: (x, y) coordinates of nucleus center
            radii: (major_radius, minor_radius) of the ellipse
            orientation: Rotation angle in radians
            intensity_mean: Mean intensity value for the nucleus
            intensity_std: Standard deviation of intensity values
        """
        self.centroid = centroid
        self.radii = radii
        self.orientation = orientation
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std

    def generate_intensity_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate a 2D intensity map for the nucleus."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        pos = np.dstack((x, y))

        # Create covariance matrix for elliptical Gaussian
        major_sigma = self.radii[0] / 3  # Convert radius to standard deviation
        minor_sigma = self.radii[1] / 3

        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(self.orientation), -np.sin(self.orientation)],
            [np.sin(self.orientation), np.cos(self.orientation)]
        ])

        # Create covariance matrix
        cov = np.array([[major_sigma ** 2, 0], [0, minor_sigma ** 2]])
        cov = rot_matrix @ cov @ rot_matrix.T

        # Generate Gaussian distribution
        rv = multivariate_normal(self.centroid, cov)
        intensity = rv.pdf(pos)

        # Scale intensity to desired range
        intensity = intensity / np.max(intensity) * self.intensity_mean

        # Add random noise
        noise = np.random.normal(0, self.intensity_std, intensity.shape)
        intensity = np.maximum(intensity + noise, 0)  # Ensure non-negative values

        return intensity


class SyntheticCluster:
    def __init__(
            self,
            center: Tuple[float, float],
            radius: float,
            n_nuclei: int,
            nuclei_size_range: Tuple[float, float] = (10, 15),
            intensity_range: Tuple[float, float] = (800, 1200)
    ):
        """
        Create a cluster of synthetic nuclei.

        Args:
            center: (x, y) coordinates of cluster center
            radius: Radius of the cluster
            n_nuclei: Number of nuclei in the cluster
            nuclei_size_range: (min, max) size range for nuclei
            intensity_range: (min, max) intensity range for nuclei
        """
        self.center = center
        self.radius = radius
        self.n_nuclei = n_nuclei
        self.nuclei_size_range = nuclei_size_range
        self.intensity_range = intensity_range
        self.nuclei: List[SyntheticNucleus] = []

        self._generate_nuclei()

    def _generate_nuclei(self):
        """Generate nuclei positions and properties within the cluster."""
        for _ in range(self.n_nuclei):
            # Generate random position within cluster (with margin)
            margin = max(self.nuclei_size_range)
            r = np.random.uniform(0, self.radius - margin)
            theta = np.random.uniform(0, 2 * np.pi)

            x = self.center[0] + r * np.cos(theta)
            y = self.center[1] + r * np.sin(theta)

            # Generate random nucleus properties
            major_radius = np.random.uniform(*self.nuclei_size_range)
            minor_radius = major_radius * np.random.uniform(0.7, 1.0)  # Slight ellipticity
            orientation = np.random.uniform(0, np.pi)
            intensity = np.random.uniform(*self.intensity_range)

            nucleus = SyntheticNucleus(
                centroid=(x, y),
                radii=(major_radius, minor_radius),
                orientation=orientation,
                intensity_mean=intensity
            )
            self.nuclei.append(nucleus)

    def generate_intensity_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate combined intensity map for all nuclei in the cluster."""
        intensity_map = np.zeros(shape)
        for nucleus in self.nuclei:
            intensity_map += nucleus.generate_intensity_map(shape)
        return intensity_map


class SyntheticImage:
    def __init__(
            self,
            shape: Tuple[int, int],
            background_noise: float = 50,
    ):
        """
        Create a synthetic microscopy image with multiple clusters.

        Args:
            shape: (height, width) of the image
            background_noise: Standard deviation of background noise
        """
        self.shape = shape
        self.background_noise = background_noise
        self.clusters: List[SyntheticCluster] = []

    def add_cluster(self, cluster: SyntheticCluster):
        """Add a cluster to the image."""
        self.clusters.append(cluster)

    def generate_image(self) -> np.ndarray:
        """Generate the final synthetic image."""
        # Generate intensity map
        intensity_map = np.zeros(self.shape)
        for cluster in self.clusters:
            intensity_map += cluster.generate_intensity_map(self.shape)

        # Add background noise
        noise = np.random.normal(0, self.background_noise, self.shape)
        intensity_map = np.maximum(intensity_map + noise, 0)

        return intensity_map


def generate_test_image(
        image_size: Tuple[int, int] = (512, 512),
        n_clusters: int = 8,
        cluster_size_range: Tuple[float, float] = (50, 100),
        nuclei_per_cluster_range: Tuple[int, int] = (5, 15)
) -> np.ndarray:
    """
    Generate a test image with multiple clusters.

    Args:
        image_size: Size of the output image
        n_clusters: Number of clusters to generate
        cluster_size_range: (min, max) radius range for clusters
        nuclei_per_cluster_range: (min, max) range for number of nuclei per cluster

    Returns:
        Synthetic microscopy image as numpy array
    """
    # Create synthetic image
    image = SyntheticImage(image_size)

    # Generate clusters
    margin = max(cluster_size_range)
    for _ in range(n_clusters):
        # Random cluster position
        x = np.random.uniform(margin, image_size[1] - margin)
        y = np.random.uniform(margin, image_size[0] - margin)

        # Random cluster properties
        radius = np.random.uniform(*cluster_size_range)
        n_nuclei = np.random.randint(*nuclei_per_cluster_range)

        cluster = SyntheticCluster(
            center=(x, y),
            radius=radius,
            n_nuclei=n_nuclei
        )

        image.add_cluster(cluster)

    return image.generate_image()


# Example usage:
if __name__ == "__main__":
    # Generate test image
    test_image = generate_test_image()

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(test_image, cmap='gray')
    plt.title("Synthetic Cell Image")
    plt.colorbar()
    plt.show()