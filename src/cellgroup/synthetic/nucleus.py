from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Union

from numpy.typing import NDArray

from cellgroup.synthetic import Space


class Nucleus(BaseModel):
    """Defines a nucleus instance with geometric properties."""
    
    id: int
    """Unique nucleus ID."""
    # TODO: find a way to generate unique IDs
    
    dim: Literal["2D", "3D"]
    
    timestep: int
    """Overall timestep of simulation."""
    
    eta: int = 0
    """Number of time steps since the nucleus was created."""
    
    centroid: tuple[int, int, int]
    
    # --- geometric properties ---
    aspect_ratio: float
    
    radii: Union[int, tuple[int, ...]]
    # TODO: or use eccentricty for ellipsoid?
    
    # and others geometric params ...
    
    # --- evolutionary properties ---
    lineage: list[int] = Field(default_factory=list)
    """Lineage, i.e. list of parent nuclei IDs."""
    # nice to keep track of the lineage
    
    death_prob: float = Field(0.0, ge=0.0, le=1.0)
    
    division_prob: float = Field(0.0, ge=0.0, le=1.0)
    
    ...
    
    # --- computed properties ---
    @property
    def surface_area(self) -> float:
        # given the params, calculate the area of the nucleus
        raise NotImplementedError
    
    @property
    def volume(self) -> float:
        # given the params, calculate the volume of the nucleus
        raise NotImplementedError
    
    
    # --- methods ---
    def divide(self) -> "Nucleus":
        # divide the nucleus into two daughter nuclei
        raise NotImplementedError
        return self.copy()
    
    def update(self):
        # update the nucleus properties depending on the timestep
        self.timestep += 1
        raise NotImplementedError
    
    
class NucleusFluorophoreDistribution(Nucleus):
    """Defines a fluorophore density distribution over the nucleus."""
    
    fluorophore_density: NDArray
    """Fluorophore density distribution."""
    
    def render(self, space: Space) -> NDArray:
        """Render the nucleus, given its properties and the space object."""
        raise NotImplementedError
    
    

    
    
