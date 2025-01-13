from __future__ import annotations

from typing import Optional, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space 

#TODO: we should consider supporting 3D simulation from the start

class Nucleus(BaseModel):
    """Defines a nucleus instance with minimal core properties and growth dynamics."""

    # TODO: refactor names -> capital letters not common in Python
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True, # TODO: why is this needed?
        validate_assignment=True,
        validate_default=True,
    )
    
    # Essential identification and tracking
    id: int = Field(description="Unique nucleus ID")
    Labels: int = Field(description="Cluster label")
    Time: int = Field(description="Temporal information")
    eta: int = Field(default=0, description="Age of nucleus in timesteps")

    # Core positional and geometric properties
    XM: float = Field(description="X coordinate of mass center")
    YM: float = Field(description="Y coordinate of mass center")
    Major: float = Field(description="Length of major axis")
    Minor: float = Field(description="Length of minor axis")
    Angle: float = Field(description="Orientation angle in degrees")

    # TODO: descriptions can be more detailed and put in """ """ under Field
    
    # Core intensity properties
    RawIntDen: float = Field(description="Raw integrated density") # TODO: not clear

    # TODO: in some simulations we might want to disable some properties.
    # Therefore we need to put `Optional` for all properties and set default to None.
    
    # Growth and death properties
    # TODO: it would be nice to set ranges for these values to avoid unrealistic values
    growth_rate: float = Field(default=0.1, description="Base growth rate")
    max_size: float = Field(default=1000.0, description="Maximum area")
    min_division_size: float = Field(default=500.0, description="Minimum size for division")
    min_viable_size: float = Field(default=50.0, description="Minimum viable size")
    max_age: int = Field(default=200, description="Maximum age in timesteps")
    is_alive: bool = Field(default=True, description="Viability status")

    # Optional evolutionary properties
    lineage: list[int] = Field(default_factory=list, description="List of parent nuclei IDs")
    death_prob: float = Field(0.0, ge=0.0, le=1.0)
    division_prob: float = Field(0.0, ge=0.0, le=1.0)

    # --- Calculated geometric properties ---
    @property
    def Area(self) -> float:
        """Calculate area using ellipse formula."""
        return np.pi * (self.Major / 2) * (self.Minor / 2)

    @property
    def AspectRatio(self) -> float:
        """Calculate aspect ratio."""
        return self.Major / self.Minor

    @property
    def Roundness(self) -> float:
        """Calculate roundness (4*Area/(π*Major^2))."""
        return (4 * self.Area) / (np.pi * self.Major ** 2)

    @property
    def perimeter(self) -> float:
        """Calculate perimeter using Ramanujan approximation."""
        a = self.Major
        b = self.Minor
        h = ((a - b) / (a + b)) ** 2
        return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    
    @property
    def Circ(self) -> float:
        """Calculate circularity (4π*Area/perimeter^2)."""
        return (4 * np.pi * self.Area) / (self.perimeter ** 2)

    @property
    def Solidity(self) -> float:
        """Approximate solidity (area/convex hull area)."""
        raise NotImplementedError

    # --- Calculated intensity properties ---
    @property
    def IntDen(self) -> float: # TODO: not clear
        """Calculate integrated density."""
        return self.RawIntDen

    @property
    def Mean(self) -> float: # TODO: naming is misleading
        """Calculate mean intensity."""
        return self.RawIntDen / self.Area

    @property
    def centroid(self) -> tuple[float, float]:
        """Return centroid coordinates."""
        return (self.XM, self.YM)

    def _calculate_growth_factor(self) -> float:
        """Calculate growth factor based on current size and conditions."""
        # Logistic growth factor
        size_factor = 1 - (self.Area / self.max_size)

        # Age-dependent modulation
        age_factor = np.exp(-self.eta / 100)  # Decreases with age

        # Calculate base growth increment
        growth_increment = self.growth_rate * size_factor * age_factor

        # Add some random variation
        noise = np.random.normal(0, 0.02)  # 2% random variation 
        #TODO: check if this is a good value to hardcode, or if it should be a parameter

        return 1 + growth_increment + noise

    def _check_death(self) -> bool:
        """Check if the nucleus should die based on various conditions."""
        if not self.is_alive:
            return False

        # Death conditions
        if (
            self.Area < self.min_viable_size or  # Too small
            self.eta > self.max_age or  # Too old
            np.random.random() < self.death_prob
        ):  # Random death
            return True

        return False

    def die(self):
        """Handle death of nucleus."""
        self.is_alive = False
        # Rapid size decrease #TODO: not clear
        shrink_factor = 0.5
        self.Major *= shrink_factor
        self.Minor *= shrink_factor
        self.RawIntDen *= shrink_factor

    def update(self) -> bool: #TODO: is there a reason why we want to return a boolean?
        """Update nucleus properties for one timestep. Returns False if nucleus dies."""
        if not self.is_alive:
            return False

        # Increment age
        self.eta += 1

        # Check for death #TODO: this is ok, but can be made more readable
        if self._check_death():
            self.die()
            return False

        # Calculate growth factor
        growth_factor = self._calculate_growth_factor()

        # Update size while maintaining aspect ratio
        #TODO: also implement non-isotropic growth
        self.Major *= np.sqrt(growth_factor)
        self.Minor *= np.sqrt(growth_factor)

        # Update intensity proportionally to area
        self.RawIntDen *= growth_factor

        # Random movement (Brownian motion)
        diffusion_coefficient = 1.0  # Can be adjusted #TODO: move into parameters
        dx = np.random.normal(0, np.sqrt(2 * diffusion_coefficient))
        dy = np.random.normal(0, np.sqrt(2 * diffusion_coefficient))
        self.XM += dx
        self.YM += dy
        # TODO: checks to implement:
        # - make sure nucleus does not leave the image Space
        # - make sure nucleus does not overlap with other nuclei
        # - make sure nucleus does not exit the cluster (or maybe it could?)

        # Random rotation
        rotation_rate = 0.1  # Degrees per timestep #TODO: move into parameters
        dangle = np.random.normal(0, rotation_rate)
        self.Angle = (self.Angle + dangle) % 360

        # Update division probability based on size and age
        size_factor = max(0, (self.Area - self.min_division_size) / self.min_division_size)
        age_factor = np.exp(-self.eta / 50)  # Decreases with age
        self.division_prob = 0.1 * size_factor * age_factor  # Base rate * factors

        # Update death probability based on age and size
        stress_factor = max(0, (self.Area - self.max_size) / self.max_size)
        age_factor = self.eta / self.max_age
        self.death_prob = min(0.8, age_factor + stress_factor)

        return True

    def divide(self) -> "Nucleus":
        # TODO: shouldn't we return a pair of `Nucleus` and make mother die?
        # This is also a biological question. Whatever makes more sense from a 
        # biological perspective is the better choice.
        """Divide nucleus into two daughter nuclei if conditions are met."""
        if self.Area < self.min_division_size or np.random.random() > self.division_prob:
            return self

        # Create daughter nucleus with same properties
        daughter = self.model_copy()
        daughter.id = self.id + 1000 
        # TODO: not clear -> do we want a hash function that generates unique IDs maybe?
        # Could be useful to retrieve nuclei then.
        daughter.eta = 0
        daughter.lineage = self.lineage + [self.id]

        # Calculate division axis (perpendicular to major axis)
        division_angle = np.radians(self.Angle + 90)
        displacement = self.Minor / 2

        # Calculate new positionsc #TODO: not clear
        dx = displacement * np.cos(division_angle)
        dy = displacement * np.sin(division_angle)

        # Update positions
        self.XM -= dx
        self.YM -= dy
        daughter.XM = self.XM + 2 * dx
        daughter.YM = self.YM + 2 * dy

        # Scale down sizes (maintain total area)
        scale_factor = 1 / np.sqrt(2)
        self.Major *= scale_factor
        self.Minor *= scale_factor
        daughter.Major *= scale_factor
        daughter.Minor *= scale_factor

        # Divide intensity roughly equally (with some noise)
        intensity_ratio = np.random.normal(0.5, 0.05)
        self.RawIntDen *= intensity_ratio
        daughter.RawIntDen *= (1 - intensity_ratio)

        return daughter

    @classmethod
    def create_from_measurements(cls,
                                 measurements: dict) -> "Nucleus":
        """Create a nucleus instance from minimal required measurements."""
        return cls(
            id=measurements.get('id', np.random.randint(1, 100000)),
            Labels=measurements['Labels'],
            Time=measurements['Time'],
            XM=measurements['X'],
            YM=measurements['Y'],
            Major=measurements['Major'],
            Minor=measurements['Minor'],
            Angle=measurements['Angle'],
            RawIntDen=measurements['RawIntDen']
        )    

    
    
