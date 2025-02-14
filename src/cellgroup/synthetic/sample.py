from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

from cellgroup.synthetic.cluster import Cluster
from cellgroup.synthetic.space import Space


class Sample(BaseModel):
    """Defines a sample with multiple clusters that evolve over time."""
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    space: Space
    "Space where the sample exists"

    time: int = 0
    "Current timestep of the simulation"

    clusters: list[Cluster] = Field(default_factory=list)
    "List of clusters in the sample"
    
    dead_clusters: list[Cluster] = Field(default_factory=list)
    "List of dead (i.e., all nuclei died) clusters in the sample"
    # might be useful for analysis (e.g., lineage)
    
    merged_clusters: list[Cluster] = Field(default_factory=list)
    "List of merged clusters in the sample"
    # might be useful for analysis (e.g., lineage) --> #TODO: add lineage to clusters

    # Optional properties for cluster interactions
    cluster_interaction_range: float = 100.0
    "Maximum distance for cluster interactions"

    cluster_merge_threshold: float = 20.0
    "Distance threshold for merging clusters"
    
    @field_validator("clusters", "dead_clusters", "merged_clusters")
    def _validate_clusters_ndims(cls, v: list[Cluster]) -> list[Cluster]:
        """Check if all clusters have the same dimensionality."""
        if not v:
            return v

        ndims = v[0].ndims
        if not all(n.ndims == ndims for n in v[1:]):
            raise ValueError("All clusters must have the same dimensionality!")
        return v
    
    # TODO: implement more validators w.r.t. Space
    
    #TODO: implement nice __repr__ method to get a summary of the sample
    
    @property
    def is_3D(self) -> bool:
        """Check if sample is 3D."""
        return self.space.ndim == 3
    
    @property
    def ndims(self) -> int:
        """Number of dimensions (2D or 3D)."""
        return self.space.ndims
    
    @property
    def count(self) -> int:
        """Number of clusters in sample."""
        return len(self.clusters)

    @property
    def total_nuclei(self) -> int:
        """Total number of nuclei across all clusters."""
        return sum(cluster.count for cluster in self.clusters)

    @property
    def nuclei_count(self) -> dict[int, int]:
        """Number of nuclei in each cluster."""
        return {c.idx: c.count for c in self.clusters}

    #TODO: necessary to have this?
    @property
    def centroid(self) -> tuple[float, ...]:
        """Calculate sample centroid."""
        if not self.clusters:
            return (0.0,) * self.clusters.ndims #TODO: adapt to 2D/3D

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

    def _check_cluster_merge(self) -> list[tuple[int, int]]:
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

    def _merge_clusters(self):
        """Merge specified cluster pairs."""
        cluster_pairs = self._check_cluster_merge()
        
        if not cluster_pairs:
            return

        # Process merges in reverse order to maintain indices
        cluster_pairs = sorted(cluster_pairs, reverse=True)

        for i, j in cluster_pairs:
            #TODO: here I would create a new cluster instance and add the old ones to
            # a list of merged clusters, to keep track of evolution history
            c1 = self.clusters[i]
            c2 = self.clusters[j]
            
            # Create new cluster
            new_cluster = Cluster(
                nuclei=c1.nuclei + c2.nuclei,
                dead_nuclei=c1.dead_nuclei + c2.dead_nuclei,
                #TODO: add other properties
            )

            # Update cluster properties
            new_cluster.max_radius = tuple(
                max(r1, r2) for r1, r2 in zip(c1.max_radius, c2.max_radius)
            )

            # Update cluster lists
            self.merged_clusters.append(c1)
            self.merged_clusters.append(c2)
            self.clusters.pop(i)
            self.clusters.pop(j)
            self.clusters.append(new_cluster)
            

    def update(self) -> None:
        """Update the status of clusters in the sample."""
        self.timestep += 1

        # Update individual clusters
        active_clusters = []
        for cluster in self.clusters:
            cluster.update()
            
            # Check for empty clusters
            if cluster.count == 0:
                self.dead_clusters.append(cluster)
            else:
                active_clusters.append(cluster)
        
        self.clusters = active_clusters

        # Check for cluster merging
        self._merge_clusters()

    def render(self) -> NDArray:
        """Render the sample."""
        if not self.clusters:
            return np.zeros(self.space.size)

        # Initialize image
        image = np.zeros(self.space.size)

        # Render each cluster
        for cluster in self.clusters:
            image += cluster.render()

        return image

    def get_cluster_metrics(self) -> dict[str, Any]:
        """Calculate various metrics for the sample."""
        raise NotImplementedError("Method not yet implemented.")
        if not self.clusters:
            return {}

        metrics = {
            'timestep': self.timestep,
            'n_clusters': self.count,
            'total_nuclei': self.total_nuclei,
            'mean_cluster_size': self.total_nuclei / self.count,
            'total_volume': sum(c.size for c in self.clusters),
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
    def create_random_sample(
        cls,
        space: Space,
        time: int,
        n_clusters: int,
        cluster_centers: list[tuple[int, ...]],
        cluster_radii: list[tuple[int, ...]],
        nuclei_per_cluster_range: tuple[int, int],
        nuclei_semi_axes_range: tuple[int, int],
        **kwargs
    ) -> 'Sample':
        """Create a sample with randomly positioned clusters."""
        clusters: list[Cluster] = []

        #FIXME: this is annoyingly slow, let's do it manually for now
        # attempts = 0
        # max_attempts = 100
        # while len(clusters) < n_clusters and attempts < max_attempts:
            # Generate random center
            # center = np.random.uniform(100, space.size - 100)
            #
            # # Check distance from existing clusters
            # too_close = False
            # for cluster in clusters:
            #     dist = np.linalg.norm(
            #         np.array(center) - np.array(cluster.centroid)
            #     )
            #     if dist < min_separation:
            #         too_close = True
            #         break

            # if too_close:
            #     attempts += 1
            #     continue

        for i in range(n_clusters):
            # Sample random properties
            n_nuclei = np.random.randint(*nuclei_per_cluster_range)
            
            # Create new cluster
            cluster = Cluster.create_random_cluster(
                space=space,
                time=time,
                idx=i,
                n_nuclei=n_nuclei,
                center=cluster_centers[i],
                radii=cluster_radii[i],
                semi_axes_range=nuclei_semi_axes_range,   
            )
            clusters.append(cluster)

        return cls(
            clusters=clusters,
            space=space,
            time=time,
            **kwargs
        )