from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.spatial.grid import SpatialGrid
from cellgroup.synthetic.space import Space

# Constants for numerical stability
EPSILON = 1e-10  # Small number for float comparisons
MAX_SINGLE_DISPLACEMENT = 10.0  # Maximum displacement per iteration
MIN_SEPARATION = 0.1  # Minimum separation to maintain between nuclei


@dataclass
class CollisionPair:
    """Represents a collision between two nuclei."""
    nucleus1_id: int
    nucleus2_id: int
    overlap_distance: float
    displacement_vector: np.ndarray
    collision_normal: np.ndarray  # Unit vector along collision direction


class CollisionResolver(BaseModel):
    """Handles detection and resolution of nucleus overlaps.

    This class implements sophisticated collision detection and resolution for
    ellipsoidal nuclei in both 2D and 3D spaces. It uses iterative resolution
    with adaptive step sizes to ensure stable convergence.
    """

    space: Space
    max_iterations: int = Field(default=50, ge=1)
    displacement_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    convergence_threshold: float = Field(default=0.1, gt=0.0)
    collision_buffer: float = Field(default=1.0, ge=0.0)

    # New parameters for enhanced stability
    max_displacement_per_step: float = Field(default=MAX_SINGLE_DISPLACEMENT, gt=0.0)
    min_separation: float = Field(default=MIN_SEPARATION, gt=0.0)

    class Config:
        arbitrary_types_allowed = True

    def _transform_to_unit_sphere(
            self,
            point: np.ndarray,
            center: np.ndarray,
            semi_axes: np.ndarray,
            rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """Transform a point to unit sphere space for an ellipsoid."""
        try:
            # Translate to origin
            translated = point - center

            # Rotate to align with axes
            aligned = rotation_matrix.T @ translated

            # Handle numerical instability in scaling
            scaled = np.zeros_like(aligned)
            for i, (axis, val) in enumerate(zip(semi_axes, aligned)):
                if abs(axis) > EPSILON:  # Avoid division by very small numbers
                    scaled[i] = val / axis

            return scaled

        except Exception as e:
            raise ValueError(f"Failed to transform point: {e}")

    def _get_closest_point_on_ellipsoid(
            self,
            point: np.ndarray,
            nucleus: Nucleus,
            max_iterations: int = 10
    ) -> Tuple[np.ndarray, float]:
        """Find the closest point on an ellipsoid surface to a given point.

        Uses Newton-Raphson iteration for accurate convergence.

        Returns
        -------
        Tuple[np.ndarray, float]
            Closest point and distance to surface
        """
        center = np.array(nucleus.centroid)
        semi_axes = np.array(nucleus.semi_axes)
        rotation = nucleus._get_rotation_matrix()

        # Handle point at center
        if np.allclose(point, center, atol=EPSILON):
            # Choose arbitrary direction based on largest semi-axis
            direction = np.zeros_like(center)
            direction[np.argmax(semi_axes)] = 1.0
            surface_point = center + rotation @ (direction * semi_axes)
            return surface_point, np.linalg.norm(semi_axes[np.argmax(semi_axes)])

        # Transform to unit sphere space
        unit_point = self._transform_to_unit_sphere(
            point, center, semi_axes, rotation
        )

        # Newton-Raphson iteration to find closest surface point
        current_point = unit_point / np.linalg.norm(unit_point)
        for _ in range(max_iterations):
            # Calculate gradient and update
            gradient = 2 * current_point
            hessian = 2 * np.eye(len(current_point))

            # Update position
            delta = np.linalg.solve(hessian, gradient)
            new_point = current_point - delta

            # Normalize to keep on unit sphere
            new_point = new_point / np.linalg.norm(new_point)

            # Check convergence
            if np.allclose(new_point, current_point, atol=EPSILON):
                break

            current_point = new_point

        # Transform back to original space
        surface_aligned = current_point * semi_axes
        surface_point = center + (rotation @ surface_aligned)

        return surface_point, np.linalg.norm(point - surface_point)

    def detect_overlap(
            self,
            nucleus1: Nucleus,
            nucleus2: Nucleus
    ) -> Optional[CollisionPair]:
        """Check if two nuclei overlap and calculate collision parameters."""
        try:
            # Verify dimensionality
            if len(nucleus1.centroid) != len(nucleus2.centroid):
                raise ValueError("Nuclei must have same dimensionality")

            # Quick bounding sphere check
            center1 = np.array(nucleus1.centroid)
            center2 = np.array(nucleus2.centroid)
            max_radius1 = max(nucleus1.semi_axes)
            max_radius2 = max(nucleus2.semi_axes)

            center_distance = np.linalg.norm(center2 - center1)
            if center_distance > (max_radius1 + max_radius2 + self.collision_buffer):
                return None

            # Handle complete overlap
            if center_distance < EPSILON:
                # Generate random direction for separation
                direction = np.random.randn(len(center1))
                direction = direction / np.linalg.norm(direction)

                return CollisionPair(
                    nucleus1_id=nucleus1.idx,
                    nucleus2_id=nucleus2.idx,
                    overlap_distance=max(max_radius1, max_radius2),
                    displacement_vector=direction * max(max_radius1, max_radius2),
                    collision_normal=direction
                )

            # Get closest points on surfaces
            closest1, dist1 = self._get_closest_point_on_ellipsoid(center2, nucleus1)
            closest2, dist2 = self._get_closest_point_on_ellipsoid(center1, nucleus2)

            # Use average of distances for stability
            overlap_distance = (dist1 + dist2) / 2

            if overlap_distance < self.collision_buffer + self.min_separation:
                # Calculate collision normal
                collision_vector = closest2 - closest1
                collision_normal = collision_vector / (np.linalg.norm(collision_vector) + EPSILON)

                return CollisionPair(
                    nucleus1_id=nucleus1.idx,
                    nucleus2_id=nucleus2.idx,
                    overlap_distance=self.collision_buffer + self.min_separation - overlap_distance,
                    displacement_vector=collision_vector,
                    collision_normal=collision_normal
                )

            return None

        except Exception as e:
            raise ValueError(f"Collision detection failed: {e}")

    def _calculate_displacement(
            self,
            collision: CollisionPair,
            nucleus1: Nucleus,
            nucleus2: Nucleus
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate displacement vectors to resolve a collision."""
        try:
            # Handle zero displacement vector
            if np.allclose(collision.displacement_vector, 0, atol=EPSILON):
                direction = collision.collision_normal
            else:
                direction = collision.collision_normal

            # Calculate mass-weighted displacement
            mass1 = np.prod(nucleus1.semi_axes)
            mass2 = np.prod(nucleus2.semi_axes)
            total_mass = mass1 + mass2

            # Avoid division by zero
            if total_mass < EPSILON:
                factor1 = factor2 = 0.5
            else:
                factor1 = mass2 / total_mass
                factor2 = mass1 / total_mass

            # Calculate base displacement
            displacement = min(
                collision.overlap_distance * self.displacement_factor,
                self.max_displacement_per_step
            )

            # Calculate individual displacements
            disp1 = -direction * displacement * factor1
            disp2 = direction * displacement * factor2

            # Limit maximum displacement
            if np.linalg.norm(disp1) > self.max_displacement_per_step:
                disp1 = disp1 * (self.max_displacement_per_step / np.linalg.norm(disp1))
            if np.linalg.norm(disp2) > self.max_displacement_per_step:
                disp2 = disp2 * (self.max_displacement_per_step / np.linalg.norm(disp2))

            return disp1, disp2

        except Exception as e:
            raise ValueError(f"Displacement calculation failed: {e}")

    def _apply_displacement(
            self,
            nucleus: Nucleus,
            displacement: np.ndarray,
            spatial_grid: Optional[SpatialGrid] = None
    ) -> None:
        """Apply displacement to nucleus while respecting space bounds."""
        try:
            # Calculate new position
            new_pos = np.array(nucleus.centroid) + displacement

            # Handle boundary conditions
            if spatial_grid is None or not spatial_grid.periodic:
                # Clip to space bounds
                for i, (pos, size) in enumerate(zip(new_pos, self.space.size)):
                    # Leave small margin from edges
                    new_pos[i] = np.clip(pos, self.min_separation,
                                         size - self.min_separation)
            else:
                # Apply periodic bounds
                new_pos = new_pos % self.space.size

            # Update nucleus position
            nucleus.centroid = tuple(new_pos)

            # Update spatial grid if provided
            if spatial_grid is not None:
                spatial_grid.update_nucleus(nucleus)

        except Exception as e:
            raise ValueError(f"Failed to apply displacement: {e}")

    def resolve_overlaps(
            self,
            nuclei: List[Nucleus],
            spatial_grid: SpatialGrid
    ) -> bool:
        """Resolve all overlaps between nuclei."""
        if not nuclei:
            return True

        try:
            # Create lookup for nuclei by ID
            nucleus_lookup = {n.idx: n for n in nuclei}

            iteration = 0
            max_displacement = float('inf')

            while iteration < self.max_iterations and max_displacement > self.convergence_threshold:
                max_displacement = 0
                displacements = {n.idx: np.zeros_like(n.centroid) for n in nuclei}

                # Detect all collisions
                collisions = []
                for nucleus in nuclei:
                    potential_collisions = spatial_grid.get_potential_collisions(nucleus)

                    for other_id in potential_collisions:
                        other = nucleus_lookup[other_id]

                        # Only process each pair once
                        if nucleus.idx < other_id:
                            collision = self.detect_overlap(nucleus, other)
                            if collision is not None:
                                collisions.append(collision)

                # Early exit if no collisions
                if not collisions:
                    return True

                # Calculate and accumulate displacements
                for collision in collisions:
                    n1 = nucleus_lookup[collision.nucleus1_id]
                    n2 = nucleus_lookup[collision.nucleus2_id]

                    disp1, disp2 = self._calculate_displacement(collision, n1, n2)
                    displacements[n1.idx] += disp1
                    displacements[n2.idx] += disp2

                    max_displacement = max(
                        max_displacement,
                        np.linalg.norm(disp1),
                        np.linalg.norm(disp2)
                    )

                # Apply accumulated displacements
                for nucleus in nuclei:
                    if np.any(displacements[nucleus.idx]):
                        self._apply_displacement(
                            nucleus,
                            displacements[nucleus.idx],
                            spatial_grid
                        )

                iteration += 1

            return max_displacement <= self.convergence_threshold

        except Exception as e:
            raise RuntimeError(f"Overlap resolution failed: {e}")