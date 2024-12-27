from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple, List
import numpy as np


class Space(BaseModel):
    """Defines a space where synthetic data live."""

    space: tuple[int, int, int] = Field(
        description="Size of space in pixels (width, height, depth)"
    )

    scale: tuple[int, int, int] = Field(
        description="Voxel size in each dimension in μm"
    )

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'Space':
        """Validate that dimensions are positive."""
        if any(s <= 0 for s in self.space):
            raise ValueError("Space dimensions must be positive")
        if any(s <= 0 for s in self.scale):
            raise ValueError("Scale dimensions must be positive")
        return self

    @property
    def dimensions(self) -> int:
        """Return number of non-unit dimensions."""
        return sum(1 for s in self.space if s > 1)

    @property
    def volume(self) -> float:
        """Return total volume in μm³."""
        return np.prod([s1 * s2 for s1, s2 in zip(self.space, self.scale)])

    @property
    def bounds(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return bounds for each dimension as (min, max) tuples."""
        return tuple((0, s) for s in self.space)

    def pixel_to_microns(self, coords: tuple[int, ...]) -> tuple[float, ...]:
        """Convert pixel coordinates to microns."""
        return tuple(c * s for c, s in zip(coords, self.scale))

    def microns_to_pixel(self, coords: tuple[float, ...]) -> tuple[int, ...]:
        """Convert micron coordinates to pixels."""
        return tuple(int(c / s) for c, s in zip(coords, self.scale))

    def is_inside(self, coords: tuple[int, ...]) -> bool:
        """Check if coordinates are inside space bounds."""
        return all(0 <= c < s for c, s in zip(coords, self.space))

    def random_position(self, margin: int = 0) -> tuple[int, ...]:
        """Generate random position within space bounds with optional margin."""
        if margin < 0:
            raise ValueError("Margin must be non-negative")
        if any(2 * margin >= s for s in self.space):
            raise ValueError("Margin too large for space dimensions")

        return tuple(
            np.random.randint(margin, s - margin)
            for s in self.space
        )

    def distance(self, coords1: tuple[int, ...], coords2: tuple[int, ...]) -> float:
        """Calculate Euclidean distance between two points in microns."""
        if len(coords1) != len(coords2):
            raise ValueError("Coordinate dimensions must match")

        # Convert to microns before calculating distance
        microns1 = self.pixel_to_microns(coords1)
        microns2 = self.pixel_to_microns(coords2)

        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(microns1, microns2)))

    def get_neighborhood(self,
                         center: tuple[int, ...],
                         radius: int
                         ) -> List[tuple[int, ...]]:
        """Get list of coordinates within radius pixels of center."""
        if radius < 0:
            raise ValueError("Radius must be non-negative")

        # Generate coordinate ranges for each dimension
        ranges = [
            range(max(0, c - radius), min(s, c + radius + 1))
            for c, s in zip(center, self.space)
        ]

        # Generate all combinations within ranges
        coords = [
            coords for coords in np.ndindex(*[len(r) for r in ranges])
            if sum((i - radius) ** 2 for i in coords) <= radius ** 2
        ]

        # Convert indices back to actual coordinates
        return [
            tuple(ranges[dim][i] for dim, i in enumerate(coord))
            for coord in coords
        ]

    def create_grid(self,
                    spacing: tuple[int, ...]
                    ) -> List[tuple[int, ...]]:
        """Create regular grid of points with given spacing."""
        if len(spacing) != len(self.space):
            raise ValueError("Spacing must match space dimensions")
        if any(s <= 0 for s in spacing):
            raise ValueError("Spacing must be positive")

        ranges = [range(0, s1, s2) for s1, s2 in zip(self.space, spacing)]
        return list(np.ndindex(*[len(r) for r in ranges]))

    @classmethod
    def create_2D(cls, width: int, height: int, scale_xy: int = 1) -> 'Space':
        """Create 2D space with given dimensions."""
        return cls(
            space=(width, height, 1),
            scale=(scale_xy, scale_xy, 1)
        )

    @classmethod
    def create_3D(cls,
                  width: int,
                  height: int,
                  depth: int,
                  scale_xyz: int = 1
                  ) -> 'Space':
        """Create 3D space with given dimensions."""
        return cls(
            space=(width, height, depth),
            scale=(scale_xyz, scale_xyz, scale_xyz)
        )
        
    