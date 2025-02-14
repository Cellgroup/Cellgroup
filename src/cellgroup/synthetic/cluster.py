from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tqdm import tqdm

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space
from cellgroup.synthetic.utils import Status


class Cluster(BaseModel):
    """Defines a cluster of nuclei with spatial organization and evolution."""
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    idx: int
    "Unique cluster index."
    
    time: int
    "Current timestep of the simulation."
    
    space: Space
    "Space where the cluster exists."

    nuclei: list[Nucleus] = Field(default_factory=list)
    "List of active nuclei in the cluster."
    
    dead_nuclei: list[Nucleus] = Field(default_factory=list)
    "List of dead nuclei in the cluster."
    
    divided_nuclei: list[Nucleus] = Field(default_factory=list)
    "List of nuclei in the cluster that have undergone division."

    # Geometric properties
    max_radius: tuple[float, ...]
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
    
    @field_validator("nuclei", "dead_nuclei", "divided_nuclei")
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
                # Access X coordinate as centroid[2], Y as centroid[1]
                dx = nucleus2.centroid[2] - nucleus1.centroid[2]  # Changed from XM
                dy = nucleus2.centroid[1] - nucleus1.centroid[1]  # Changed from YM
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance == 0:
                    continue

                # Rest of the force calculation remains the same
                dx /= distance
                dy /= distance

                repulsion = self.repulsion_strength / (distance ** 2)
                optimal_distance = (nucleus1.Major + nucleus2.Major) / 4
                adhesion = self.adhesion_strength * (distance - optimal_distance) * np.exp(-distance / optimal_distance)

                force = repulsion - adhesion

                forces[i] = (forces[i][0] - force * dx, forces[i][1] - force * dy)
                forces[j] = (forces[j][0] + force * dx, forces[j][1] + force * dy)

        return forces
    
    def apply_forces(self) -> None:
        """Apply forces to nuclei in the cluster."""
        raise NotImplementedError("Force application not implemented yet!")
        # Calculate and apply forces
        forces = self._calculate_forces()

        # Update positions based on forces
        for nucleus, (fx, fy) in zip(self.nuclei, forces):
            # Add random noise
            # TODO: isn't this already in nucleus.update()?
            fx += np.random.normal(0, self.noise_strength)
            fy += np.random.normal(0, self.noise_strength)

            # Update position
            new_x = nucleus.centroid[2] + fx  # Changed from XM
            new_y = nucleus.centroid[1] + fy  # Changed from YM
            # TODO: implement check for new position
            # Keep within bounds
            new_x = np.clip(new_x, 0, self.max_radius[0])
            new_y = np.clip(new_y, 0, self.max_radius[1])

            # Update nucleus position in Z,Y,X order
            nucleus.centroid = (0, new_y, new_x)  # Changed from (new_x, new_y, 0)

            # TODO: adapt to 3D
            # TODO: can a nucleus be kicked out of the cluster due to repulsion?

    def update(self) -> None:
        """Update the status of nuclei in the cluster."""
        if self.count == 0:
            return

        # --- Update individual nuclei
        alive_nuclei = []
        for nucleus in self.nuclei:
            curr_status = nucleus.update()
            
            #TODO: implement check for new position (or at sample level)
            
            # --- Remove dead & divided nuclei
            if curr_status == Status.ALIVE:
                alive_nuclei.append(nucleus)                
            elif curr_status == Status.DIVIDED:
                # If nucleus has divided, compute the daugters
                d1, d2 = nucleus.divide()
                alive_nuclei.append(d1)
                alive_nuclei.append(d2)
                self.divided_nuclei.append(nucleus)
            elif curr_status == Status.DEAD:
                self.dead_nuclei.append(nucleus)

        self.nuclei = alive_nuclei

        # --- Apply inter-cluster forces
        # self.apply_forces()
    

    def render(self, border: bool = False) -> NDArray:
        """Render the cluster."""
        if self.is_empty:
            return np.zeros(self.space.size)

        # Render each nucleus
        # TODO: vectorize rendering (especially for 3D)
        image = np.zeros(self.space.size)
        for nucleus in tqdm(
            self.nuclei, desc=f"Rendering nuclei in cluster {self.idx}"
        ):
            image += nucleus.render()
        print("------------------------------")

        # Add border if requested
        if border:
            raise NotImplementedError("Border rendering not implemented yet!")
        
        return image

    @classmethod
    def create_random_cluster(
        cls,
        space: Space,
        time: int,
        idx: int,
        n_nuclei: int,
        center: tuple[float, ...],
        radii: tuple[float, ...],
        semi_axes_range: tuple[float, float],
        **kwargs
    ) -> 'Cluster':
        """Create a cluster with randomly positioned nuclei."""
        assert len(center) == len(radii) == len(space.size), (
            "Center and radius must match space dimensions!"
        )
        dim = "3D" if len(center) == 3 else "2D"
        
        nuclei = []
        for i in range(n_nuclei):
            # Generate random position within radius
            if dim == "3D":
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(0, radii)
                centroid = center + r * np.array([
                    np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
                ])
            elif dim == "2D":
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, radii)
                centroid = center + r * np.array([np.cos(theta), np.sin(theta)])
            semi_axes = np.random.uniform(*semi_axes_range, size=len(center))
            angles = {"theta": np.random.uniform(0, np.pi)}
            if dim == "3D":
                angles["phi"] = np.random.uniform(0, np.pi)
                angles["psi"] = np.random.uniform(0, np.pi)

            # Create nucleus at position with Z,Y,X ordering
            nucleus = Nucleus(
                idx=i,
                space=space,
                dim=dim,
                time=time,
                centroid=centroid,
                semi_axes=semi_axes,
                **angles,
                # TODO: pass other args in a pydantic config
            )
            nuclei.append(nucleus)
            
        # FIXME: add function call to avoid overlap of nuclei

        return cls(
            nuclei=nuclei,
            idx=idx,
            space=space,
            time=time,
            max_radius=radii,
            concentration=n_nuclei / np.prod(radii),    
            **kwargs
        )