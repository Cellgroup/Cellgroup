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

    distribution_type: Literal["binary", "gaussian", "uniform", "ring"] = "gaussian"

    # Distribution parameters
    intensity_center: float = 1.0  # Relative intensity at center
    intensity_edge: float = 0.3  # Relative intensity at edge
    noise_std: float = 0.1  # Standard deviation of noise
    background_level: float = 0.05  # Background intensity level
    
    @staticmethod
    def _normalize(distribution: NDArray) -> NDArray:
        """Normalize distribution to sum to 1."""
        min_ = distribution.min()
        max_ = distribution.max()
        return (distribution - min_) / (max_ - min_)
    
    def _rescale(self, distribution: NDArray) -> NDArray:
        """Rescale distribution to desired intensity range."""
        return (
            self.intensity_edge + distribution * (
                self.intensity_center - self.intensity_edge
            )
        )

    def _generate_gaussian_distribution(self, space_shape: tuple[int, ...]) -> NDArray:
        """Generate Gaussian fluorophore distribution."""
        # Generate grid of coordinates
        coords = np.mgrid[tuple(slice(0, s) for s in space_shape)]
        pos = np.dstack(*coords)

        # Rotation matrix
        rot_matrix = self.nucleus._get_rotation_matrix()
        
        # --- Create covariance matrix for elliptical distribution
        sigmas = self.nucleus.semi_axes / 2  # Convert radius to standard deviation
        
        cov = np.eye(self.nucleus.ndims) * sigmas ** 2
        cov = rot_matrix @ cov @ rot_matrix.T

        # --- Generate distribution
        rv = np.random.normal(self.nucleus.centroid, cov)
        distribution = rv.pdf(pos)

        # --- Scale to desired intensity range
        distribution = self._normalize(distribution)
        distribution = self._rescale(distribution)
        return distribution

    def _generate_ring_distribution(self, space_shape: tuple[int, ...]) -> NDArray:
        """Generate ring-like fluorophore distribution.
        
        Parameters
        ----------
        space_shape : tuple[int, ...]
            Shape of the space in which the nucleus lives. Coords given as [(Z), Y, X].
            
        Returns
        -------
        NDArray
            Fluorophore distribution according to the ring pattern.
        """
        distances = self.nucleus._compute_ellipsoidal_distances(space_shape)

        # Create ring pattern
        ring_radius = 0.7  # Position of peak intensity #TODO: make this a parameter
        ring_width = 0.2  # Width of the ring #TODO: make this a parameter
        distribution = np.exp(-((distances - ring_radius) / ring_width) ** 2)

        # --- Scale to desired intensity range
        distribution = self._normalize(distribution)
        distribution = self._rescale(distribution)
        return distribution

    def _generate_uniform_distribution(self, space_shape: tuple[int, ...]) -> NDArray:
        """Generate uniform fluorophore distribution with soft edges.
        
        Parameters
        ----------
        space_shape : tuple[int, ...]
            Shape of the space in which the nucleus lives. Coords given as [(Z), Y, X].
        
        Returns
        -------
        NDArray
            Fluorophore distribution according to the uniformly decading pattern.
        """
        distances = self.nucleus._compute_ellipsoidal_distances(space_shape)

        # Create soft mask
        sigma = 0.1  # Controls edge softness #TODO: make this a parameter
        distribution = 1 / (1 + np.exp((distances - 1) / sigma))

        # --- Scale to desired intensity range
        distribution = self._rescale(distribution)
        return distribution

    def render(self, space: Space) -> NDArray:
        """Render the nucleus FP distribution, given its properties and the space object."""
        if not self.nucleus.is_alive:
            # TODO: not clear why we reduce its size and intensity in`die()` method
            # If we don't want to render dead nuclei, we should just discard them
            return np.zeros(space.size)

        # Generate base distribution based on type
        if self.distribution_type == "binary":
            distribution = self.nucleus.render(space)
        elif self.distribution_type == "gaussian":
            distribution = self._generate_gaussian_distribution(space.size)
        elif self.distribution_type == "ring":
            distribution = self._generate_ring_distribution(space.size)
        else:  # uniform
            distribution = self._generate_uniform_distribution(space.size)

        # Add gaussian blur
        noise = np.random.normal(0, self.noise_std, distribution.shape)
        distribution = np.maximum(distribution + noise, 0)

        # Add background
        distribution += self.background_level

        # Scale by raw integrated density # TODO: maybe not necessary
        distribution *= (self.nucleus.raw_int_density / distribution.sum())

        return distribution
