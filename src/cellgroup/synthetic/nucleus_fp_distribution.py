from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space


class NucleusFluorophoreDistribution(BaseModel):
    """Defines a fluorophore density distribution over the nucleus."""

    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed = True
    )
    
    nucleus: Nucleus
    """Nucleus object on which FP distribution is rendered."""
    
    fluorophore_density: NDArray #TODO: not clear
    """Fluorophore density distribution."""

    distribution_type: Literal["gaussian", "uniform", "ring"] = "gaussian"

    # Distribution parameters
    intensity_center: float = 1.0  # Relative intensity at center
    intensity_edge: float = 0.3  # Relative intensity at edge
    noise_std: float = 0.1  # Standard deviation of noise
    background_level: float = 0.05  # Background intensity level

    def _generate_gaussian_distribution(self, shape: tuple[int, int]) -> NDArray:
        """Generate Gaussian fluorophore distribution."""
        #TODO: this step would need a brief explanation
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        pos = np.dstack((x, y))

        # --- Create covariance matrix for elliptical distribution
        major_sigma = self.nucleus.Major / 4  # Convert radius to standard deviation
        minor_sigma = self.nucleus.Minor / 4

        # Rotation matrix
        theta = np.radians(self.Angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Create covariance matrix
        cov = np.array([[major_sigma ** 2, 0], [0, minor_sigma ** 2]])
        cov = rot_matrix @ cov @ rot_matrix.T

        # --- Generate distribution
        rv = np.random.normal([self.nucleus.XM, self.nucleus.YM], cov)
        distribution = rv.pdf(pos)

        # Scale to desired intensity range
        distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def _generate_ring_distribution(self, shape: tuple[int, int]) -> NDArray:
        """Generate ring-like fluorophore distribution."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]

        # Calculate distances from center
        # TODO: adapt to 3D
        dx = x - self.nucleus.XM
        dy = y - self.nucleus.YM

        # Apply rotation
        theta = np.radians(self.Angle)
        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        # Scale to create elliptical distance
        dx_scaled = dx_rot / (self.nucleus.Major)
        dy_scaled = dy_rot / (self.nucleus.Minor)
        distances = np.sqrt(dx_scaled ** 2 + dy_scaled ** 2)

        # Create ring pattern
        ring_radius = 0.7  # Position of peak intensity #TODO: make this a parameter
        ring_width = 0.2  # Width of the ring #TODO: make this a parameter
        distribution = np.exp(-((distances - ring_radius) / ring_width) ** 2)

        # Scale to desired intensity range
        distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def _generate_uniform_distribution(self, shape: tuple[int, int]) -> NDArray:
        """Generate uniform fluorophore distribution with soft edges."""
        y, x = np.mgrid[0:shape[0], 0:shape[1]]

        # Calculate normalized distances from center
        dx = x - self.nucleus.XM
        dy = y - self.nucleus.YM

        # Apply rotation
        theta = np.radians(self.nucleus.Angle)
        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        # Scale to create elliptical mask
        dx_scaled = dx_rot / (self.nucleus.Major / 2)
        dy_scaled = dy_rot / (self.nucleus.Minor / 2)
        distances = np.sqrt(dx_scaled ** 2 + dy_scaled ** 2)

        # Create soft mask
        sigma = 0.1  # Controls edge softness #TODO: make this a parameter
        distribution = 1 / (1 + np.exp((distances - 1) / sigma))

        # Scale to desired intensity range
        distribution = self.intensity_edge + distribution * (self.intensity_center - self.intensity_edge)

        return distribution

    def render(self, space: Space) -> NDArray:
        """Render the nucleus FP distribution, given its properties and the space object."""
        if not self.nucleus.is_alive:
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

        # Scale by RawIntDen #TODO: not clear
        distribution *= (self.nucleus.RawIntDen / distribution.sum())

        return distribution

    #TODO: I don't think this is necessary, as we anyway need a nucleus to generate a
    # fluorophore distribution. Remove ?
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
