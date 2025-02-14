from typing import List, Set, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space


@dataclass
class GridCell:
    """Represents a single cell in the spatial grid.

    Parameters
    ----------
    indices : Tuple[int, ...]
        Grid cell indices (i,j) or (i,j,k)
    bounds : Tuple[Tuple[float, float], ...]
        Physical bounds of cell in each dimension as (min, max)
    nuclei : Set[int]
        Set of nucleus IDs contained in this cell
    """
    indices: Tuple[int, ...]
    bounds: Tuple[Tuple[float, float], ...]
    nuclei: Set[int] = Field(default_factory=set)

    def add_nucleus(self, nucleus_id: int) -> None:
        """Add a nucleus to this cell."""
        self.nuclei.add(nucleus_id)

    def remove_nucleus(self, nucleus_id: int) -> None:
        """Remove a nucleus from this cell."""
        self.nuclei.discard(nucleus_id)

    @property
    def is_empty(self) -> bool:
        """Check if cell contains any nuclei."""
        return len(self.nuclei) == 0

    @property
    def center(self) -> Tuple[float, ...]:
        """Get cell center coordinates."""
        return tuple((b[0] + b[1]) / 2 for b in self.bounds)


class SpatialGrid(BaseModel):
    """Spatial partitioning grid for efficient nucleus interaction detection.

    This grid divides the space into cells and tracks which nuclei are in each cell,
    allowing for efficient collision detection by only checking nuclei in nearby cells.

    Parameters
    ----------
    space : Space
        The space object defining the boundaries
    cell_size : float
        Size of each grid cell. Must be larger than the largest possible nucleus diameter
        but smaller than half the smallest space dimension.
    periodic : bool, default=False
        Whether to use periodic boundary conditions
    """

    space: Space
    cell_size: float = Field(gt=0.0)
    periodic: bool = False
    grid_cells: Dict[Tuple[int, ...], GridCell] = Field(default_factory=dict)
    nucleus_to_cells: Dict[int, Set[Tuple[int, ...]]] = Field(default_factory=dict)

    # Calculated properties
    grid_dimensions: Tuple[int, ...] = Field(init=False)
    max_nucleus_size: float = Field(default=0.0)  # Track largest nucleus for validation

    @validator('cell_size')
    def validate_cell_size(cls, v: float, values: Dict) -> float:
        """Validate that cell size is appropriate for the space."""
        if 'space' not in values:
            return v

        min_space_dim = min(values['space'].size)
        if v > min_space_dim / 2:
            raise ValueError(
                f"Cell size ({v}) is too large for space dimensions {values['space'].size}"
            )
        return v

    def __init__(self, **data):
        """Initialize the spatial grid."""
        super().__init__(**data)
        self.grid_dimensions = self._calculate_grid_dimensions()
        self._initialize_grid()

    def _calculate_grid_dimensions(self) -> Tuple[int, ...]:
        """Calculate number of grid cells needed in each dimension."""
        return tuple(int(np.ceil(s / self.cell_size)) for s in self.space.size)

    def _initialize_grid(self) -> None:
        """Create the initial grid structure with cells."""
        for indices in np.ndindex(*self.grid_dimensions):
            # Calculate physical bounds of this cell
            bounds = tuple(
                (idx * self.cell_size, min((idx + 1) * self.cell_size, size))
                for idx, size in zip(indices, self.space.size)
            )

            self.grid_cells[indices] = GridCell(
                indices=indices,
                bounds=bounds,
                nuclei=set()
            )

    def _validate_position(self, position: Tuple[float, ...]) -> None:
        """Check if a position is within space bounds."""
        if len(position) != len(self.space.size):
            raise ValueError(f"Position {position} has wrong dimensionality")

        for pos, size in zip(position, self.space.size):
            if not self.periodic and (pos < 0 or pos >= size):
                raise ValueError(f"Position {position} is outside space bounds")

    def _apply_periodic_bounds(self, position: Tuple[float, ...]) -> Tuple[float, ...]:
        """Apply periodic boundary conditions to a position if enabled."""
        if not self.periodic:
            return position

        return tuple(pos % size for pos, size in zip(position, self.space.size))

    def _get_cell_indices(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert a position to grid cell indices."""
        position = self._apply_periodic_bounds(position)
        self._validate_position(position)

        indices = tuple(int(p / self.cell_size) for p in position)
        return tuple(
            min(idx, dim - 1)
            for idx, dim in zip(indices, self.grid_dimensions)
        )

    def _get_neighbor_indices(
            self,
            center_indices: Tuple[int, ...],
            radius: int = 1
    ) -> Iterator[Tuple[int, ...]]:
        """Get indices of neighboring cells within given radius."""
        ranges = []
        for c, d in zip(center_indices, self.grid_dimensions):
            if self.periodic:
                # For periodic boundaries, wrap around
                r = range(c - radius, c + radius + 1)
                ranges.append([i % d for i in r])
            else:
                # For non-periodic boundaries, clip to grid dimensions
                ranges.append(range(max(0, c - radius), min(d, c + radius + 1)))

        for indices in np.ndindex(*tuple(len(r) for r in ranges)):
            neighbor_indices = tuple(r[i] for r, i in zip(ranges, indices))
            yield neighbor_indices

    def update_max_nucleus_size(self, nucleus: Nucleus) -> None:
        """Update tracking of maximum nucleus size."""
        size = max(nucleus.semi_axes) * 2  # Diameter
        if size > self.max_nucleus_size:
            self.max_nucleus_size = size
            if size > self.cell_size:
                raise ValueError(
                    f"Nucleus {nucleus.idx} size ({size}) exceeds cell size ({self.cell_size})"
                )

    def _get_cells_for_nucleus(self, nucleus: Nucleus) -> Set[Tuple[int, ...]]:
        """Get all grid cells that a nucleus overlaps with."""
        self.update_max_nucleus_size(nucleus)

        # Get nucleus bounding box
        bbox = nucleus.bounding_box

        # Get min and max cell indices that could contain the nucleus
        try:
            min_pos = self._apply_periodic_bounds(tuple(b[0] for b in bbox))
            max_pos = self._apply_periodic_bounds(tuple(b[1] for b in bbox))
            min_indices = self._get_cell_indices(min_pos)
            max_indices = self._get_cell_indices(max_pos)
        except ValueError as e:
            raise ValueError(f"Nucleus {nucleus.idx} bounding box out of bounds: {e}")

        # Get all cells in the range
        cells = set()
        ranges = []
        for min_idx, max_idx, dim in zip(min_indices, max_indices, self.grid_dimensions):
            if self.periodic:
                if min_idx <= max_idx:
                    ranges.append(range(min_idx, max_idx + 1))
                else:
                    # Wraps around boundary
                    ranges.append(list(range(min_idx, dim)) + list(range(0, max_idx + 1)))
            else:
                ranges.append(range(min_idx, max_idx + 1))

        for indices in np.ndindex(*tuple(len(r) for r in ranges)):
            cell_indices = tuple(r[i] for r, i in zip(ranges, indices))
            if cell_indices in self.grid_cells:
                cells.add(cell_indices)

        return cells

    def register_nucleus(self, nucleus: Nucleus) -> None:
        """Register a nucleus in all relevant grid cells."""
        try:
            cells = self._get_cells_for_nucleus(nucleus)

            for cell_indices in cells:
                self.grid_cells[cell_indices].add_nucleus(nucleus.idx)

            self.nucleus_to_cells[nucleus.idx] = cells

        except ValueError as e:
            raise ValueError(f"Failed to register nucleus {nucleus.idx}: {e}")

    def unregister_nucleus(self, nucleus: Nucleus) -> None:
        """Remove a nucleus from all its registered cells."""
        if nucleus.idx not in self.nucleus_to_cells:
            return

        for cell_indices in self.nucleus_to_cells[nucleus.idx]:
            self.grid_cells[cell_indices].remove_nucleus(nucleus.idx)

        del self.nucleus_to_cells[nucleus.idx]

    def update_nucleus(self, nucleus: Nucleus) -> None:
        """Update the registration of a nucleus (e.g., after it moves)."""
        try:
            new_cells = self._get_cells_for_nucleus(nucleus)

            if nucleus.idx not in self.nucleus_to_cells:
                self.register_nucleus(nucleus)
                return

            old_cells = self.nucleus_to_cells[nucleus.idx]

            for cell_indices in old_cells - new_cells:
                self.grid_cells[cell_indices].remove_nucleus(nucleus.idx)

            for cell_indices in new_cells - old_cells:
                self.grid_cells[cell_indices].add_nucleus(nucleus.idx)

            self.nucleus_to_cells[nucleus.idx] = new_cells

        except ValueError as e:
            raise ValueError(f"Failed to update nucleus {nucleus.idx}: {e}")

    def get_potential_collisions(self, nucleus: Nucleus) -> Set[int]:
        """Get IDs of all nuclei that could potentially collide with given nucleus."""
        try:
            # Calculate search radius based on nucleus size
            radius = max(1, int(np.ceil(2 * max(nucleus.semi_axes) / self.cell_size)))

            # Get center cell for nucleus
            center_cells = self._get_cells_for_nucleus(nucleus)

            potential_collisions = set()
            for cell_indices in center_cells:
                for neighbor_indices in self._get_neighbor_indices(cell_indices, radius):
                    if neighbor_indices in self.grid_cells:
                        potential_collisions.update(
                            self.grid_cells[neighbor_indices].nuclei
                        )

            potential_collisions.discard(nucleus.idx)
            return potential_collisions

        except ValueError as e:
            raise ValueError(
                f"Failed to get potential collisions for nucleus {nucleus.idx}: {e}"
            )

    def get_nuclei_in_radius(
            self,
            center: Tuple[float, ...],
            radius: float
    ) -> Set[int]:
        """Get IDs of all nuclei within given radius of a point."""
        try:
            # Get cell at center point
            center_indices = self._get_cell_indices(center)

            # Calculate number of cells to search based on radius
            cell_radius = int(np.ceil(radius / self.cell_size))

            # Collect nuclei from all cells within radius
            nearby_nuclei = set()
            for indices in self._get_neighbor_indices(center_indices, cell_radius):
                if indices in self.grid_cells:
                    nearby_nuclei.update(self.grid_cells[indices].nuclei)

            return nearby_nuclei

        except ValueError as e:
            raise ValueError(f"Failed to get nuclei in radius: {e}")

    def get_statistics(self) -> Dict:
        """Get statistics about grid occupancy."""
        stats = {
            'total_cells': len(self.grid_cells),
            'occupied_cells': sum(1 for cell in self.grid_cells.values() if not cell.is_empty),
            'total_nuclei': len(self.nucleus_to_cells),
            'max_nucleus_size': self.max_nucleus_size,
            'cell_size': self.cell_size,
            'nuclei_per_cell': {
                indices: len(cell.nuclei)
                for indices, cell in self.grid_cells.items()
            }
        }
        stats['average_nuclei_per_cell'] = (
                sum(stats['nuclei_per_cell'].values()) / stats['total_cells']
        )
        return stats

    def clear(self) -> None:
        """Remove all nuclei from the grid."""
        for cell in self.grid_cells.values():
            cell.nuclei.clear()
        self.nucleus_to_cells.clear()
        self.max_nucleus_size = 0.0