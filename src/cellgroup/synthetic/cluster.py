from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space


class Cluster(BaseModel):
    """Defines a cluster of nuclei with spatial organization and evolution."""
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    idx: int
    "Unique cluster index."

    nuclei: list[Nucleus] = Field(default_factory=list)
    "List of active nuclei in the cluster."
    
    dead_nuclei: list[Nucleus] = Field(default_factory=list)
    "List of dead nuclei in the cluster."
    # might be useful for analysis (e.g., lineage)
    # it is more efficient to keep them separate than to avoid calling update() on them

    # Geometric properties
    max_radius: tuple[int, ...]
    "Maximum radius in each dimension."

    concentration: float = Field(gt=0.0) # lt=1.0)
    "Density of nuclei in the cluster."

    # Optional properties for cluster behavior
    repulsion_strength: Optional[float] = 50.0
    "Repulsion strength between nuclei. If None, no repulsion."

    adhesion_strength: Optional[float] = 10.0
    "Adhesion strength between nuclei. If None, no adhesion."

    #TODO: naming is not clear
    noise_strength: float = 1.0
    "Strength of random motion."
    
    #TODO: implement nice __repr__ method to get a summary of the cluster
    
    @field_validator("nuclei")
    def _validate_nuclei_ndims(cls, v: list[Nucleus]) -> list[Nucleus]:
        """Check if all nuclei have the same dimensionality."""
        if not v:
            return v

        ndims = v[0].ndims
        if not all(n.ndims == ndims for n in v[1:]):
            raise ValueError("All nuclei must have the same dimensionality!")
        return v
    
    @model_validator(mode="after")
    def _validate_cluster_ndims(self):
        """Check if cluster is in 2D or 3D space."""
        nuclei_ndims = self.nuclei[0].ndims
        if nuclei_ndims not in (2, 3):
            raise ValueError("Cluster must have nuclei in 2D or 3D space!")
        if nuclei_ndims != len(self.max_radius):
            raise ValueError(
                "Number of `max_radius` dimensions must match the one of nuclei!"
            )
        return self
    
    @property
    def is_3D(self) -> bool:
        """Check if cluster is in a 3D space."""
        return len(self.nuclei[0].centroid) == 3 # or use max radius

    @property
    def ndims(self) -> int:
        """Number of dimensions in cluster space."""
        return len(self.nuclei[0].centroid) # or use max radius
    
    @property
    def count(self) -> int:
        """Number of nuclei in cluster."""
        return len(self.nuclei)
    
    @property
    def is_empty(self) -> bool:
        """Check if cluster is empty."""
        return self.count == 0
    
    @property
    def bounding_box(self) -> Optional[tuple[tuple[int, int], ...]]:
        """Get bounding box of the cluster."""
        raise NotImplementedError("Bounding box calculation not implemented yet!")

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Calculate cluster centroid."""
        if self.is_empty:
            return None

        centroids = np.stack([n.centroid for n in self.nuclei])
        return np.mean(centroids, axis=0)

    @property
    def size(self) -> float:
        """Calculate approximate cluster size (area in 2D, volume in 3D)."""
        if not self.nuclei:
            return 0.0

        # Get bounding box
        centroids = np.stack([n.centroid for n in self.nuclei])
        min_coords = np.min(centroids, axis=0)
        max_coords = np.max(centroids, axis=0)

        # Calculate dimensions in pixels
        dimensions = (max_coords - min_coords)
        return np.prod(dimensions)

    def _calculate_forces(self) -> list[tuple[float, ...]]:
        """Calculate forces between nuclei."""
        forces = [(0.0, 0.0) for _ in self.nuclei]

        # Calculate pairwise forces
        for i, nucleus1 in enumerate(self.nuclei):
            for j, nucleus2 in enumerate(self.nuclei[i + 1:], i + 1):
                # Calculate distance between nuclei
                dx = nucleus2.XM - nucleus1.XM
                dy = nucleus2.YM - nucleus1.YM
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance == 0:
                    continue

                # Normalized direction
                dx /= distance
                dy /= distance

                # Repulsive force (decreases with distance)
                repulsion = self.repulsion_strength / (distance ** 2)

                # Adhesive force (increases then decreases with distance)
                optimal_distance = (nucleus1.Major + nucleus2.Major) / 4
                adhesion = self.adhesion_strength * (distance - optimal_distance) * np.exp(-distance / optimal_distance)

                # Total force
                force = repulsion - adhesion

                # Add to force vectors
                forces[i] = (forces[i][0] - force * dx, forces[i][1] - force * dy)
                forces[j] = (forces[j][0] + force * dx, forces[j][1] + force * dy)

        return forces

    def update(self):
        """Update the status of nuclei in the cluster."""
        if self.count == 0:
            return

        # Update individual nuclei
        alive_nuclei = []
        for nucleus in self.nuclei:
            nucleus.update()
            
            #TODO: implement check for new position
            
            # Remove dead nuclei
            if not nucleus.is_alive:
                alive_nuclei.append(nucleus)                
            else:
                self.dead_nuclei.append(nucleus)

            # Check for division #TODO: for coherence, this should happen in nucleus.update()
            if nucleus.Area >= nucleus.min_division_size:
                daughter = nucleus.divide()
                if daughter != nucleus:
                    alive_nuclei.append(daughter)

        self.nuclei = alive_nuclei

        # Calculate and apply forces
        forces = self._calculate_forces()

        # Update positions based on forces #TODO: adapt to 3D
        for nucleus, (fx, fy) in zip(self.nuclei, forces):
            # Add random noise #TODO: isn't this already in nucleus.update()?
            fx += np.random.normal(0, self.noise_strength)
            fy += np.random.normal(0, self.noise_strength)

            # Update position
            new_x = nucleus.XM + fx
            new_y = nucleus.YM + fy

            # Keep within bounds
            new_x = np.clip(new_x, 0, self.space.size[0])
            new_y = np.clip(new_y, 0, self.space.size[1])

            nucleus.XM = new_x
            nucleus.YM = new_y
            nucleus.centroid = (int(new_x), int(new_y), 0)
            
            #TODO: can a nucleus be kicked out of the cluster due to repulsion?

    def render(self, space: Space) -> NDArray:
        """Render the cluster."""
        if self.is_empty:
            return np.zeros(space.size)

        # Render each nucleus
        image = np.zeros(space.size)
        for nucleus in self.nuclei:
            image += nucleus.render(space)

        return image

    @classmethod
    def create_random_cluster(
        cls,
        space: Space,
        n_nuclei: int,
        center: tuple[float, float],
        radius: float,
        **kwargs
    ) -> 'Cluster':
        """Create a cluster with randomly positioned nuclei."""
        nuclei = []
        for _ in range(n_nuclei):
            # Generate random position within radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)

            # Create nucleus at position
            nucleus = Nucleus(
                id=len(nuclei),
                dim="2D",
                XM=x,
                YM=y,
                centroid=(int(x), int(y), 0),
                Major=np.random.uniform(10, 20),
                Minor=np.random.uniform(8, 15),
                Angle=np.random.uniform(0, 360),
                RawIntDen=1000,
                Labels=0,
                Time=0
            )
            nuclei.append(nucleus)

        return cls(
            nuclei=nuclei,
            space=space,
            max_radius=(radius, radius, 1),
            concentration=n_nuclei / (np.pi * radius ** 2),
            **kwargs
        )