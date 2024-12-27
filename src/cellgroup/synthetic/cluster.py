from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from cellgroup.synthetic import NucleusFluorophoreDistribution, Space


class Cluster(BaseModel):
    """Defines a cluster of nuclei with spatial organization and evolution."""

    nuclei: list[NucleusFluorophoreDistribution] = Field(
        default_factory=list,
        description="List of nuclei in the cluster"
    )

    space: Space = Field(
        description="Space where the cluster exists"
    )

    # Geometric properties
    max_radius: tuple[int, int, int] = Field(
        description="Maximum radius in each dimension"
    )

    concentration: float = Field(
        description="Density of nuclei in the cluster",
        gt=0.0
    )

    # Optional properties for cluster behavior
    repulsion_strength: float = Field(
        default=50.0,
        description="Strength of repulsion between nuclei"
    )

    adhesion_strength: float = Field(
        default=10.0,
        description="Strength of adhesion between nuclei"
    )

    noise_strength: float = Field(
        default=1.0,
        description="Strength of random motion"
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def validate_cluster(self) -> 'Cluster':
        """Validate cluster configuration."""
        if len(self.max_radius) != len(self.space.space):
            raise ValueError("max_radius dimensions must match space dimensions")
        return self

    @property
    def count(self) -> int:
        """Number of nuclei in cluster."""
        return len(self.nuclei)

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Calculate cluster centroid."""
        if not self.nuclei:
            return (0.0, 0.0, 0.0)

        positions = np.array([[n.XM, n.YM, 0] for n in self.nuclei])
        return tuple(np.mean(positions, axis=0))

    @property
    def volume(self) -> float:
        """Calculate approximate cluster volume."""
        if not self.nuclei:
            return 0.0

        # Get bounding box
        positions = np.array([[n.XM, n.YM] for n in self.nuclei])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        # Calculate dimensions in microns
        dimensions = [
            (max_coords[i] - min_coords[i]) * self.space.scale[i]
            for i in range(2)
        ]

        # For 2D, use area * unit depth
        if len(self.space.space) == 2:
            return dimensions[0] * dimensions[1]

        # For 3D, use volume
        return dimensions[0] * dimensions[1] * dimensions[2]

    def _calculate_forces(self) -> List[tuple[float, float]]:
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
        if not self.nuclei:
            return

        # Update individual nuclei
        alive_nuclei = []
        for nucleus in self.nuclei:
            if nucleus.update():  # Returns False if nucleus dies
                alive_nuclei.append(nucleus)

                # Check for division
                if nucleus.Area >= nucleus.min_division_size:
                    daughter = nucleus.divide()
                    if daughter != nucleus:
                        alive_nuclei.append(daughter)

        self.nuclei = alive_nuclei

        # Calculate and apply forces
        forces = self._calculate_forces()

        # Update positions based on forces
        for nucleus, (fx, fy) in zip(self.nuclei, forces):
            # Add random noise
            fx += np.random.normal(0, self.noise_strength)
            fy += np.random.normal(0, self.noise_strength)

            # Update position
            new_x = nucleus.XM + fx
            new_y = nucleus.YM + fy

            # Keep within bounds
            new_x = np.clip(new_x, 0, self.space.space[0])
            new_y = np.clip(new_y, 0, self.space.space[1])

            nucleus.XM = new_x
            nucleus.YM = new_y
            nucleus.centroid = (int(new_x), int(new_y), 0)

    def render(self) -> NDArray:
        """Render the cluster."""
        if not self.nuclei:
            return np.zeros(self.space.space[:2])

        # Initialize image
        image = np.zeros(self.space.space[:2])

        # Render each nucleus
        for nucleus in self.nuclei:
            image += nucleus.render(self.space)

        return image

    @classmethod
    def create_random_cluster(cls,
                              space: Space,
                              n_nuclei: int,
                              center: tuple[float, float],
                              radius: float,
                              **kwargs) -> 'Cluster':
        """Create a cluster with randomly positioned nuclei."""
        nuclei = []
        for _ in range(n_nuclei):
            # Generate random position within radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)

            # Create nucleus at position
            nucleus = NucleusFluorophoreDistribution(
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