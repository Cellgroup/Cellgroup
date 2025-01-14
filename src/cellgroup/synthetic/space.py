import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Space(BaseModel):
    """Defines a space where synthetic data live."""

    size: tuple[int, ...] = Field(
        description="Size of space in pixels (Z, Y, X). Can be 2D or 3D."
    )

    scale: tuple[int, ...] = Field(
        description="Voxel size in each dimension in μm (Z, Y, X). Can be 2D or 3D."
    )
    
    #TODO: private numpy attrs for speed-up (?) -> _size, _scale
    
    @field_validator('size')
    def validate_size(self, value: tuple[int, ...]) -> tuple[int, ...]:
        """Validate size field."""
        if len(value) not in [2, 3]:
            raise ValueError("Size must be 2D or 3D.")
        if any(v <= 0 for v in value):
            raise ValueError("Size entries must be positive.")
        return value
    
    @field_validator('scale')
    def validate_scale(self, value: tuple[int, ...]) -> tuple[int, ...]:
        """Validate scale field."""
        if len(value) not in [2, 3]:
            raise ValueError("Scale must be 2D or 3D.")
        if any(v <= 0 for v in value):
            raise ValueError("Scale entries must be positive.")
        return value
    
    @model_validator
    def validate_dimensions(self):
        """Validate that size and scale have the same number of dimensions."""
        if len(self.size) != len(self.scale):
            raise ValueError("Size and scale dimensions must match.")
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions (2D or 3D)."""
        return len(self.size)

    @property
    def volume(self) -> float:
        """Return total volume in μm³."""
        return np.prod([s1 * s2 for s1, s2 in zip(self.size, self.scale)])

    #TODO: not sure it is needed...
    @property
    def bounds(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return bounds for each dimension as (min, max) tuples."""
        return tuple((0, s) for s in self.size)

    def pixel_to_microns(self, coords: tuple[int, ...]) -> tuple[float, ...]:
        """Convert pixel coordinates to microns."""
        return tuple(c * s for c, s in zip(coords, self.scale))

    def microns_to_pixel(self, coords: tuple[float, ...]) -> tuple[int, ...]:
        """Convert micron coordinates to pixels."""
        return tuple(int(c / s) for c, s in zip(coords, self.scale))

    def is_inside(self, coords: tuple[int, ...]) -> bool:
        """Check if coordinates are inside space bounds."""
        return all(0 <= c < s for c, s in zip(coords, self.size))

    def random_position(self, margin: int = 0) -> tuple[int, ...]:
        """Generate random position within space bounds with optional margin."""
        assert margin > 0, "Margin must be non-negative."
        assert all(2 * margin < s for s in self.size), "Margin too large for space dimensions."

        return tuple(
            np.random.randint(margin, s - margin)
            for s in self.size
        )

    #TODO: doesn't need to be a method in the class --> move to synthetic/utils
    def distance(self, coords1: tuple[int, ...], coords2: tuple[int, ...]) -> float:
        """Calculate Euclidean distance between two points in microns."""
        assert len(coords1) == len(coords2), ValueError("Coordinate dimensions must match.")

        # Convert to microns before calculating distance
        #TODO: for our application it is better to use pixel distances
        microns1 = self.pixel_to_microns(coords1)
        microns2 = self.pixel_to_microns(coords2)

        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(microns1, microns2)))

    #TODO: not clear how and where this method should be used
    def get_neighborhood(
        self,
        center: tuple[int, ...],
        radius: int
    ) -> list[tuple[int, ...]]:
        """Get list of coordinates within radius pixels of center."""
        assert radius > 0, ValueError("Radius must be non-negative.")

        # Generate coordinate ranges for each dimension
        #TODO: using numpy arrays instead of for loops would be more efficient
        ranges = [
            range(max(0, c - radius), min(s, c + radius + 1))
            for c, s in zip(center, self.size)
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

    #TODO: not clear how and where this method should be used
    def create_grid(
        self,
        spacing: tuple[int, ...]
    ) -> list[tuple[int, ...]]:
        """Create regular grid of points with given spacing."""
        if len(spacing) != len(self.size):
            raise ValueError("Spacing must match space dimensions")
        if any(s <= 0 for s in spacing):
            raise ValueError("Spacing must be positive")

        ranges = [range(0, s1, s2) for s1, s2 in zip(self.size, spacing)]
        return list(np.ndindex(*[len(r) for r in ranges]))
        
    