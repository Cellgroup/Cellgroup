from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict
import numpy as np
from numpy.typing import NDArray

from cellgroup.synthetic import Cluster, Space


class Sample(BaseModel):
    """Defines a sample with multiple clusters that evolve over time."""

    clusters: list[Cluster] = Field(
        default_factory=list,
        description="List of clusters in the sample"
    )

    space: Space = Field(
        description="Space where the sample exists"
    )

    timestep: int = Field(
        default=0,
        description="Current timestep of the simulation"
    )

    # Optional properties for cluster interactions
    cluster_interaction_range: float = Field(
        default=100.0,
        description="Maximum distance for cluster interactions"
    )

    cluster_merge_threshold: float = Field(
        default=20.0,
        description="Distance threshold for merging clusters"
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def count(self) -> int:
        """Number of clusters in sample."""
        return len(self.clusters)

    @property
    def total_nuclei(self) -> int:
        """Total number of nuclei across all clusters."""
        return sum(cluster.count for cluster in self.clusters)

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Calculate sample centroid."""
        if not self.clusters:
            return (0.0, 0.0, 0.0)

        weighted_positions = [
            (c.centroid, c.count) for c in self.clusters
        ]
        total_nuclei = sum(weight for _, weight in weighted_positions)

        if total_nuclei == 0:
            return (0.0, 0.0, 0.0)

        weighted_sum = [
            sum(pos[i] * weight for pos, weight in weighted_positions)
            for i in range(3)
        ]

        return tuple(ws / total_nuclei for ws in weighted_sum)

    def _check_cluster_merge(self) -> List[tuple[int, int]]:
        """Find clusters that should be merged."""
        merge_pairs = []

        for i, cluster1 in enumerate(self.clusters):
            for j, cluster2 in enumerate(self.clusters[i + 1:], i + 1):
                # Calculate distance between cluster centroids
                c1 = np.array(cluster1.centroid[:2])
                c2 = np.array(cluster2.centroid[:2])
                distance = np.linalg.norm(c2 - c1)

                if distance < self.cluster_merge_threshold:
                    merge_pairs.append((i, j))

        return merge_pairs

    def _merge_clusters(self, cluster_pairs: List[tuple[int, int]]):
        """Merge specified cluster pairs."""
        if not cluster_pairs:
            return

        # Process merges in reverse order to maintain indices
        cluster_pairs = sorted(cluster_pairs, reverse=True)

        for i, j in cluster_pairs:
            # Merge nuclei lists
            self.clusters[i].nuclei.extend(self.clusters[j].nuclei)

            # Update cluster properties
            self.clusters[i].max_radius = tuple(
                max(r1, r2) for r1, r2 in
                zip(self.clusters[i].max_radius, self.clusters[j].max_radius)
            )

            # Remove merged cluster
            self.clusters.pop(j)

    def update(self):
        """Update the status of clusters in the sample."""
        self.timestep += 1

        # Update individual clusters
        for cluster in self.clusters:
            cluster.update()

        # Remove empty clusters
        self.clusters = [c for c in self.clusters if c.count > 0]

        # Check for cluster merging
        merge_pairs = self._check_cluster_merge()
        self._merge_clusters(merge_pairs)

    def render(self) -> NDArray:
        """Render the sample."""
        if not self.clusters:
            return np.zeros(self.space.space[:2])

        # Initialize image
        image = np.zeros(self.space.space[:2])

        # Render each cluster
        for cluster in self.clusters:
            image += cluster.render()

        return image

    def get_cluster_metrics(self) -> Dict:
        """Calculate various metrics for the sample."""
        if not self.clusters:
            return {}

        metrics = {
            'timestep': self.timestep,
            'n_clusters': self.count,
            'total_nuclei': self.total_nuclei,
            'mean_cluster_size': self.total_nuclei / self.count,
            'total_volume': sum(c.volume for c in self.clusters),
            'cluster_distances': [],
            'cluster_sizes': [c.count for c in self.clusters]
        }

        # Calculate inter-cluster distances
        centroids = [np.array(c.centroid[:2]) for c in self.clusters]
        for i, c1 in enumerate(centroids):
            for c2 in centroids[i + 1:]:
                distance = np.linalg.norm(c2 - c1)
                metrics['cluster_distances'].append(distance)

        return metrics

    @classmethod
    def create_random_sample(cls,
                             space: Space,
                             n_clusters: int,
                             nuclei_per_cluster: int,
                             min_separation: float = 100.0,
                             **kwargs) -> 'Sample':
        """Create a sample with randomly positioned clusters."""
        clusters = []
        attempts = 0
        max_attempts = 100

        while len(clusters) < n_clusters and attempts < max_attempts:
            # Generate random center
            center = (
                np.random.uniform(100, space.space[0] - 100),
                np.random.uniform(100, space.space[1] - 100)
            )

            # Check distance from existing clusters
            too_close = False
            for cluster in clusters:
                dist = np.linalg.norm(
                    np.array(center) - np.array(cluster.centroid[:2])
                )
                if dist < min_separation:
                    too_close = True
                    break

            if too_close:
                attempts += 1
                continue

            # Create new cluster
            cluster = Cluster.create_random_cluster(
                space=space,
                n_nuclei=nuclei_per_cluster,
                center=center,
                radius=min_separation / 2
            )
            clusters.append(cluster)

        return cls(
            clusters=clusters,
            space=space,
            **kwargs
        )