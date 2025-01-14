from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space


class Cluster(BaseModel):
    """Defines a cluster of nuclei with spatial organization and evolution."""
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    nuclei: list[Nucleus] = Field(
        default_factory=list,
        description="List of active nuclei in the cluster."
    )
    
    dead_nuclei: list[Nucleus] = Field(
        default_factory=list,
        description="List of dead nuclei in the cluster."
    ) # might be useful for analysis (e.g., lineage)

    #TODO: Space is common to all clusters, so it should not be a field
    # of a specific cluster (unless simplifies the code)
    space: Space = Field(
        description="Space where the cluster exists."
    )

    # Geometric properties
    max_radius: tuple[int, ...] = Field(
        description="Maximum radius in each dimension."
    )

    concentration: float = Field(
        gt=0.0, # lt=1.0, #TODO: not sure about this
        description="Density of nuclei in the cluster",
    )

    # Optional properties for cluster behavior
    repulsion_strength: Optional[float] = Field(
        default=50.0,
        description="Repulsion strength between nuclei. If None, no repulsion."
    )

    adhesion_strength: Optional[float] = Field(
        default=10.0,
        description="Adhesion strength between nuclei. If None, no adhesion."
    )

    #TODO: naming is not clear
    noise_strength: float = Field(
        default=1.0,
        description="Strength of random motion"
    )

    @model_validator(mode='after')
    def validate_cluster(self) -> 'Cluster':
        """Validate cluster configuration."""
        if len(self.max_radius) != len(self.space.size):
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
    def size(self) -> float:
        """Calculate approximate cluster size (area in 2D, volume in 3D)."""
        if not self.nuclei:
            return 0.0

        # Get bounding box #TODO: this can become a method since it can used in many places
        positions = np.array([[n.XM, n.YM] for n in self.nuclei])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        # Calculate dimensions in microns #TODO: keep in pixels
        dimensions = (max_coords - min_coords) * np.array(self.space.scale) #TODO: ugly
        
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

    #TODO: this should be in the FP class, here we just simulate the geometry
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