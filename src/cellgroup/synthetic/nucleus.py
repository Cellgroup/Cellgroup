from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from typing import Optional, Tuple
from cellgroup.synthetic import Nucleus, Space 


class Nucleus(BaseModel):
    """Defines a nucleus instance with minimal core properties and growth dynamics."""

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

    # Core intensity properties
    RawIntDen: float = Field(description="Raw integrated density")

    # Growth and death properties
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

    class Config:
        arbitrary_types_allowed = True

    # --- Calculated geometric properties ---
    @property
    def Area(self) -> float:
        """Calculate area using ellipse formula."""
        return np.pi * (self.Major / 2) * (self.Minor / 2)

    @property
    def AR(self) -> float:
        """Calculate aspect ratio."""
        return self.Major / self.Minor

    @property
    def Round(self) -> float:
        """Calculate roundness (4*Area/(π*Major^2))."""
        return (4 * self.Area) / (np.pi * self.Major ** 2)

    @property
    def Circ(self) -> float:
        """Calculate circularity (4π*Area/perimeter^2)."""
        perimeter = np.pi * (self.Major + self.Minor) * (
                    1 + (3 * ((self.Major - self.Minor) / (self.Major + self.Minor)) ** 2) / (
                        10 + np.sqrt(4 - 3 * ((self.Major - self.Minor) / (self.Major + self.Minor)) ** 2)))
        return (4 * np.pi * self.Area) / (perimeter ** 2)

    @property
    def Solidity(self) -> float:
        """Approximate solidity (area/convex hull area)."""
        return 1.0

    # --- Calculated intensity properties ---
    @property
    def IntDen(self) -> float:
        """Calculate integrated density."""
        return self.RawIntDen

    @property
    def Mean(self) -> float:
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

        return 1 + growth_increment + noise

    def _check_death(self) -> bool:
        """Check if the nucleus should die based on various conditions."""
        if not self.is_alive:
            return False

        # Death conditions
        if (self.Area < self.min_viable_size or  # Too small
                self.eta > self.max_age or  # Too old
                np.random.random() < self.death_prob):  # Random death
            return True

        return False

    def die(self):
        """Handle death of nucleus."""
        self.is_alive = False
        # Rapid size decrease
        shrink_factor = 0.5
        self.Major *= shrink_factor
        self.Minor *= shrink_factor
        self.RawIntDen *= shrink_factor

    def update(self) -> bool:
        """Update nucleus properties for one timestep. Returns False if nucleus dies."""
        if not self.is_alive:
            return False

        # Increment age
        self.eta += 1

        # Check for death
        if self._check_death():
            self.die()
            return False

        # Calculate growth factor
        growth_factor = self._calculate_growth_factor()

        # Update size while maintaining aspect ratio
        self.Major *= np.sqrt(growth_factor)
        self.Minor *= np.sqrt(growth_factor)

        # Update intensity proportionally to area
        self.RawIntDen *= growth_factor

        # Random movement (Brownian motion)
        diffusion_coefficient = 1.0  # Can be adjusted
        dx = np.random.normal(0, np.sqrt(2 * diffusion_coefficient))
        dy = np.random.normal(0, np.sqrt(2 * diffusion_coefficient))
        self.XM += dx
        self.YM += dy

        # Random rotation
        rotation_rate = 0.1  # Degrees per timestep
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
        """Divide nucleus into two daughter nuclei if conditions are met."""
        if self.Area < self.min_division_size or np.random.random() > self.division_prob:
            return self

        # Create daughter nucleus with same properties
        daughter = self.model_copy()
        daughter.id = self.id + 1000
        daughter.eta = 0
        daughter.lineage = self.lineage + [self.id]

        # Calculate division axis (perpendicular to major axis)
        division_angle = np.radians(self.Angle + 90)
        displacement = self.Minor / 2

        # Calculate new positions
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




class NucleusFluorophoreDistribution(Nucleus):
    """Defines a fluorophore density distribution over the nucleus."""

    fluorophore_density: NDArray
    """Fluorophore density distribution."""

    distribution_type: str = "gaussian"  # or "uniform", "ring"

    # Distribution parameters
    intensity_center: float = 1.0  # Relative intensity at center
    intensity_edge: float = 0.3  # Relative intensity at edge
    noise_std: float = 0.1  # Standard deviation of noise
    background_level: float = 0.05  # Background intensity level

    class Config:
        arbitrary_types_allowed = True

    def _generate_gaussian_distribution(self, shape: Tuple[int, int]) -> NDArray:
        """Generate Gaussian fluorophore distribution."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        pos = np.dstack((x, y))

        # Create covariance matrix for elliptical distribution
        major_sigma = self.Major / 4  # Convert radius to standard deviation
        minor_sigma = self.Minor / 4

        # Rotation matrix
        theta = np.radians(self.Angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Create covariance matrix
        cov = np.array([[major_sigma ** 2, 0], [0, minor_sigma ** 2]])
        cov = rot_matrix @ cov @ rot_matrix.T

        # Generate distribution
        rv = multivariate_normal([self.XM, self.YM], cov)
        distribution = rv.pdf(pos)

        # Scale to desired intensity range
        distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def _generate_ring_distribution(self, shape: Tuple[int, int]) -> NDArray:
        """Generate ring-like fluorophore distribution."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]

        # Calculate distances from center
        dx = x - self.XM
        dy = y - self.YM

        # Apply rotation
        theta = np.radians(self.Angle)
        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        # Scale to create elliptical distance
        dx_scaled = dx_rot / (self.Major / 2)
        dy_scaled = dy_rot / (self.Minor / 2)
        distances = np.sqrt(dx_scaled ** 2 + dy_scaled ** 2)

        # Create ring pattern
        ring_radius = 0.7  # Position of peak intensity
        ring_width = 0.2  # Width of the ring
        distribution = np.exp(-((distances - ring_radius) / ring_width) ** 2)

        # Scale to desired intensity range
        distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def _generate_uniform_distribution(self, shape: Tuple[int, int]) -> NDArray:
        """Generate uniform fluorophore distribution with soft edges."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]

        # Calculate normalized distances from center
        dx = x - self.XM
        dy = y - self.YM

        # Apply rotation
        theta = np.radians(self.Angle)
        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        # Scale to create elliptical mask
        dx_scaled = dx_rot / (self.Major / 2)
        dy_scaled = dy_rot / (self.Minor / 2)
        distances = np.sqrt(dx_scaled ** 2 + dy_scaled ** 2)

        # Create soft mask
        sigma = 0.1  # Controls edge softness
        distribution = 1 / (1 + np.exp((distances - 1) / sigma))

        # Scale to desired intensity range
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def render(self, space: Space) -> NDArray:
        """Render the nucleus, given its properties and the space object."""
        if not self.is_alive:
            return np.zeros(space.space[:2])

        # Generate base distribution based on type
        if self.distribution_type == "gaussian":
            distribution = self._generate_gaussian_distribution(space.space[:2])
        elif self.distribution_type == "ring":
            distribution = self._generate_ring_distribution(space.space[:2])
        else:  # uniform
            distribution = self._generate_uniform_distribution(space.space[:2])

        # Add noise
        noise = np.random.normal(0, self.noise_std, distribution.shape)
        distribution = np.maximum(distribution + noise, 0)

        # Add background
        distribution += self.background_level

        # Scale by RawIntDen
        distribution *= (self.RawIntDen / distribution.sum())

        return distribution

    @classmethod
    def create_from_nucleus(cls,
                            nucleus: Nucleus,
                            distribution_type: str = "gaussian",
                            **kwargs) -> "NucleusFluorophoreDistribution":
        """Create a fluorophore distribution from a base nucleus."""
        # Create empty fluorophore density array
        fluorophore_density = np.array([])

        # Create instance with all nucleus properties
        nucleus_dict = nucleus.model_dump()
        instance = cls(
            **nucleus_dict,
            fluorophore_density=fluorophore_density,
            distribution_type=distribution_type,
            **kwargs
        )

        return instance
    

    
    
