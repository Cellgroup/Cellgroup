from typing import List, Dict, Set, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict

from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.space import Space
from cellgroup.synthetic.spatial.grid import SpatialGrid
from cellgroup.synthetic.physics.collision import CollisionResolver


class UpdatePriority(Enum):
    """Priority levels for different update events."""
    CRITICAL = auto()  # Death events
    HIGH = auto()  # Division events
    MEDIUM = auto()  # Growth events
    LOW = auto()  # Movement events


class UpdateEvent(Enum):
    """Types of events that can occur during nucleus updates."""
    GROWTH = "growth"
    DIVISION = "division"
    DEATH = "death"
    COLLISION = "collision"
    BOUNDARY = "boundary"
    ROTATION = "rotation"


@dataclass
class UpdateResult:
    """Results of a nucleus update."""
    nucleus_id: int
    event_type: UpdateEvent
    priority: UpdatePriority
    new_nuclei: List[Nucleus]
    success: bool
    message: str
    details: Dict = Field(default_factory=dict)  # Additional event-specific information


class DensityState(Enum):
    """Classification of local density states."""
    SPARSE = "sparse"  # Few nuclei, standard updates
    MODERATE = "moderate"  # Normal density, standard updates
    DENSE = "dense"  # High density, careful updates
    CRITICAL = "critical"  # Extremely dense, emergency measures


class UpdateCoordinator(BaseModel):
    """Coordinates the multi-phase update process for nuclei."""

    # Core parameters
    space: Space
    cell_size: Optional[float] = None
    periodic_bounds: bool = False
    max_nuclei: Optional[int] = None
    min_nucleus_separation: float = Field(default=1.0, gt=0.0)

    # Advanced update control
    time_step: float = Field(default=1.0, gt=0.0)
    adaptive_stepping: bool = True
    min_time_step: float = Field(default=0.1, gt=0.0)
    max_time_step: float = Field(default=2.0, gt=0.0)

    # Density control
    density_thresholds: Dict[DensityState, float] = Field(
        default_factory=lambda: {
            DensityState.SPARSE: 0.2,  # Up to 20% local occupancy
            DensityState.MODERATE: 0.5,  # Up to 50% local occupancy
            DensityState.DENSE: 0.8,  # Up to 80% local occupancy
            DensityState.CRITICAL: 1.0  # Above 80% local occupancy
        }
    )

    # Components
    spatial_grid: Optional[SpatialGrid] = None
    collision_resolver: Optional[CollisionResolver] = None

    # Tracking
    update_history: Dict[int, List[UpdateResult]] = Field(default_factory=dict)
    current_time: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize the update coordinator."""
        super().__init__(**data)

        # Initialize cell size if not provided
        if self.cell_size is None:
            max_dimension = max(self.space.size)
            self.cell_size = max_dimension / 10

        # Initialize components
        self.spatial_grid = SpatialGrid(
            space=self.space,
            cell_size=self.cell_size,
            periodic=self.periodic_bounds
        )

        self.collision_resolver = CollisionResolver(
            space=self.space,
            collision_buffer=self.min_nucleus_separation
        )

    def _assess_local_density(self, position: Tuple[float, ...], radius: float) -> DensityState:
        """Assess the local density state around a position."""
        try:
            # Get nuclei in local region
            nearby_nuclei = self.spatial_grid.get_nuclei_in_radius(position, radius * 2)

            # Calculate local density
            local_space = np.pi * radius * radius  # Simplified 2D case
            occupied_space = sum(np.pi * np.prod(n.semi_axes)
                                 for n in nearby_nuclei)

            density_ratio = occupied_space / local_space

            # Determine density state
            for state, threshold in sorted(
                    self.density_thresholds.items(),
                    key=lambda x: x[1]
            ):
                if density_ratio <= threshold:
                    return state

            return DensityState.CRITICAL

        except Exception:
            # Default to DENSE on error to be cautious
            return DensityState.DENSE

    def _adjust_time_step(self,
                          density_state: DensityState,
                          previous_updates: List[UpdateResult]
                          ) -> None:
        """Adjust time step based on local density and update history."""
        if not self.adaptive_stepping:
            return

        # Base adjustment on density state
        adjustment_factors = {
            DensityState.SPARSE: 1.1,  # Increase step size
            DensityState.MODERATE: 1.0,  # Maintain step size
            DensityState.DENSE: 0.8,  # Reduce step size
            DensityState.CRITICAL: 0.5  # Significantly reduce step size
        }

        # Consider recent update success
        recent_success_rate = sum(1 for r in previous_updates[-10:]
                                  if r.success) / max(len(previous_updates[-10:]), 1)

        # Adjust time step
        factor = adjustment_factors[density_state] * (0.5 + 0.5 * recent_success_rate)
        new_step = self.time_step * factor

        # Clamp to limits
        self.time_step = np.clip(new_step, self.min_time_step, self.max_time_step)

    def _validate_and_prepare_nucleus(
            self,
            nucleus: Nucleus,
            density_state: DensityState
    ) -> UpdateResult:
        """Validate and prepare a nucleus for update."""
        try:
            # Basic validation
            if not self._validate_nucleus(nucleus):
                return UpdateResult(
                    nucleus_id=nucleus.idx,
                    event_type=UpdateEvent.GROWTH,
                    priority=UpdatePriority.CRITICAL,
                    new_nuclei=[],
                    success=False,
                    message="Invalid nucleus state",
                    details={"density_state": density_state.value}
                )

            # Adjust nucleus parameters based on density
            if density_state in [DensityState.DENSE, DensityState.CRITICAL]:
                # Reduce movement and growth rates in dense regions
                nucleus.growth_rate *= 0.5
                nucleus.noise_strength *= 0.5

            return UpdateResult(
                nucleus_id=nucleus.idx,
                event_type=UpdateEvent.GROWTH,
                priority=UpdatePriority.MEDIUM,
                new_nuclei=[],
                success=True,
                message="Nucleus prepared for update",
                details={"density_state": density_state.value}
            )

        except Exception as e:
            return UpdateResult(
                nucleus_id=nucleus.idx,
                event_type=UpdateEvent.GROWTH,
                priority=UpdatePriority.CRITICAL,
                new_nuclei=[],
                success=False,
                message=f"Preparation failed: {str(e)}",
                details={"density_state": density_state.value}
            )

    def _handle_rotation(self, nucleus: Nucleus) -> UpdateResult:
        """Handle nucleus rotation updates."""
        try:
            # Store original angles
            original_angles = nucleus.angles

            # Update rotation based on nucleus properties
            # This is a placeholder for your specific rotation logic
            new_angles = tuple(
                angle + np.random.normal(0, nucleus.noise_strength)
                for angle in original_angles
            )

            # Apply new rotation
            nucleus.angle_x = new_angles[0]
            if len(new_angles) > 1:
                nucleus.angle_y = new_angles[1]
            if len(new_angles) > 2:
                nucleus.angle_z = new_angles[2]

            return UpdateResult(
                nucleus_id=nucleus.idx,
                event_type=UpdateEvent.ROTATION,
                priority=UpdatePriority.LOW,
                new_nuclei=[],
                success=True,
                message="Rotation updated successfully",
                details={
                    "original_angles": original_angles,
                    "new_angles": new_angles
                }
            )

        except Exception as e:
            return UpdateResult(
                nucleus_id=nucleus.idx,
                event_type=UpdateEvent.ROTATION,
                priority=UpdatePriority.LOW,
                new_nuclei=[],
                success=False,
                message=f"Rotation update failed: {str(e)}",
                details={"original_angles": original_angles}
            )

    def _handle_dense_division(
            self,
            nucleus: Nucleus,
            daughter_cells: Tuple[Nucleus, Nucleus],
            density_state: DensityState
    ) -> UpdateResult:
        """Handle division in dense regions."""
        try:
            if density_state == DensityState.CRITICAL:
                # Prevent division in critical density
                return UpdateResult(
                    nucleus_id=nucleus.idx,
                    event_type=UpdateEvent.DIVISION,
                    priority=UpdatePriority.HIGH,
                    new_nuclei=[],
                    success=False,
                    message="Division prevented in critical density region",
                    details={"density_state": density_state.value}
                )

            # For dense regions, modify daughter cell positions
            for daughter in daughter_cells:
                # Reduce separation distance
                vector = np.array(daughter.centroid) - np.array(nucleus.centroid)
                vector *= 0.7  # Reduce separation
                daughter.centroid = tuple(np.array(nucleus.centroid) + vector)

            return self._handle_division(nucleus, daughter_cells)

        except Exception as e:
            return UpdateResult(
                nucleus_id=nucleus.idx,
                event_type=UpdateEvent.DIVISION,
                priority=UpdatePriority.HIGH,
                new_nuclei=[],
                success=False,
                message=f"Dense division handling failed: {str(e)}",
                details={"density_state": density_state.value}
            )

    def update_nuclei(
            self,
            nuclei: List[Nucleus]
    ) -> Tuple[List[Nucleus], List[UpdateResult]]:
        """Perform complete update cycle for all nuclei."""
        if not nuclei:
            return [], []

        try:
            results = []
            updated_nuclei = []

            # Group nuclei by local density
            density_groups = defaultdict(list)
            for nucleus in nuclei:
                density_state = self._assess_local_density(
                    nucleus.centroid,
                    max(nucleus.semi_axes)
                )
                density_groups[density_state].append(nucleus)

            # Update nuclei in order of density (sparse to dense)
            for density_state in [
                DensityState.SPARSE,
                DensityState.MODERATE,
                DensityState.DENSE,
                DensityState.CRITICAL
            ]:
                group_nuclei = density_groups[density_state]
                if not group_nuclei:
                    continue

                # Adjust time step for this density group
                self._adjust_time_step(density_state, results)

                # Process nuclei in this group
                for nucleus in group_nuclei:
                    # Prepare nucleus
                    prep_result = self._validate_and_prepare_nucleus(
                        nucleus,
                        density_state
                    )
                    if not prep_result.success:
                        results.append(prep_result)
                        continue

                    # Handle rotation
                    rot_result = self._handle_rotation(nucleus)
                    results.append(rot_result)

                    # Regular update with density-specific handling
                    update_result = self.update_single_nucleus(nucleus)
                    results.append(update_result)

                    if update_result.success:
                        if update_result.event_type == UpdateEvent.DIVISION:
                            if density_state in [DensityState.DENSE, DensityState.CRITICAL]:
                                # Special handling for division in dense regions
                                dense_result = self._handle_dense_division(
                                    nucleus,
                                    tuple(update_result.new_nuclei),
                                    density_state
                                )
                                results.append(dense_result)
                                if dense_result.success:
                                    updated_nuclei.extend(dense_result.new_nuclei)
                            else:
                                updated_nuclei.extend(update_result.new_nuclei)
                        elif update_result.event_type != UpdateEvent.DEATH:
                            updated_nuclei.append(nucleus)

                # Resolve collisions for this density group
                if updated_nuclei:
                    collision_result = self.resolve_collisions(updated_nuclei)
                    results.append(collision_result)

                    if not collision_result.success and density_state != DensityState.CRITICAL:
                        # Try emergency resolution for persistent collisions
                        emergency_result = self._handle_emergency_collision_resolution(
                            updated_nuclei
                        )
                        results.append(emergency_result)

            # Update history
            for result in results:
                if result.nucleus_id not in self.update_history:
                    self.update_history[result.nucleus_id] = []
                self.update_history[result.nucleus_id].append(result)

            # Update time
            self.current_time += self.time_step

            return updated_nuclei, results

        except Exception as e:
            error_result = UpdateResult(
                nucleus_id=-1,
                event_type=UpdateEvent.GROWTH,
                priority=UpdatePriority.CRITICAL,
                new_nuclei=[],
                success=False,
                message=f"Critical update failure: {str(e)}",
                details={"time": self.current_time}
            )
            return [], [error_result]

    def _handle_emergency_collision_resolution(
            self,
            nuclei: List[Nucleus]
    ) -> UpdateResult:
        """Emergency collision resolution for persistent collisions."""
        try:
            # Sort nuclei by size (larger nuclei get priority)
            sorted_nuclei = sorted(
                nuclei,
                key=lambda n: -np.prod(n.semi_axes)
            )

            # Track successful resolutions
            resolved_count = 0
            modified_nuclei = set()

            # Process each nucleus
            for i, nucleus in enumerate(sorted_nuclei):
                collisions_resolved = False
                max_attempts = 3
                attempt = 0

                while not collisions_resolved and attempt < max_attempts:
                    # Get current collisions
                    colliding_nuclei = self.spatial_grid.get_potential_collisions(nucleus)
                    if not colliding_nuclei:
                        collisions_resolved = True
                        continue

                    # Calculate average direction away from colliding nuclei
                    escape_vector = np.zeros_like(nucleus.centroid, dtype=float)
                    for other_id in colliding_nuclei:
                        other = next(n for n in sorted_nuclei if n.idx == other_id)
                        direction = np.array(nucleus.centroid) - np.array(other.centroid)
                        if np.any(direction):  # Avoid zero division
                            escape_vector += direction / np.linalg.norm(direction)

                    if np.any(escape_vector):
                        # Normalize escape vector
                        escape_vector = escape_vector / np.linalg.norm(escape_vector)

                        # Calculate escape distance based on nucleus size
                        escape_distance = max(nucleus.semi_axes) * (attempt + 1)

                        # Move nucleus
                        new_pos = np.array(nucleus.centroid) + escape_vector * escape_distance

                        # Apply boundary conditions
                        if self.periodic_bounds:
                            new_pos = new_pos % self.space.size
                        else:
                            for j, (pos, size) in enumerate(zip(new_pos, self.space.size)):
                                new_pos[j] = np.clip(
                                    pos,
                                    max(nucleus.semi_axes[j], self.min_nucleus_separation),
                                    size - max(nucleus.semi_axes[j], self.min_nucleus_separation)
                                )

                        # Update position
                        nucleus.centroid = tuple(new_pos)
                        self.spatial_grid.update_nucleus(nucleus)
                        modified_nuclei.add(nucleus.idx)

                    attempt += 1

                if collisions_resolved:
                    resolved_count += 1

            # Determine success based on resolution rate
            success_rate = resolved_count / len(sorted_nuclei)
            success_threshold = 0.8  # Consider it successful if 80% resolved

            return UpdateResult(
                nucleus_id=-1,  # Collective update
                event_type=UpdateEvent.COLLISION,
                priority=UpdatePriority.CRITICAL,
                new_nuclei=[],
                success=success_rate >= success_threshold,
                message=(
                    f"Emergency resolution: {resolved_count}/{len(sorted_nuclei)} "
                    f"nuclei resolved ({success_rate:.1%})"
                ),
                details={
                    "resolved_count": resolved_count,
                    "total_nuclei": len(sorted_nuclei),
                    "modified_nuclei": list(modified_nuclei),
                    "success_rate": success_rate
                }
            )

        except Exception as e:
            return UpdateResult(
                nucleus_id=-1,
                event_type=UpdateEvent.COLLISION,
                priority=UpdatePriority.CRITICAL,
                new_nuclei=[],
                success=False,
                message=f"Emergency resolution failed: {str(e)}",
                details={"error": str(e)}
            )

    def get_update_statistics(self) -> Dict:
        """Get comprehensive statistics about updates."""
        stats = {
            'total_updates': sum(len(results) for results in self.update_history.values()),
            'events': {event.value: 0 for event in UpdateEvent},
            'priorities': {priority.name: 0 for priority in UpdatePriority},
            'success_rate': 0.0,
            'active_nuclei': len(self.spatial_grid.nucleus_to_cells),
            'current_time': self.current_time,
            'time_step': self.time_step,
            'density_states': defaultdict(int),
            'grid_stats': self.spatial_grid.get_statistics()
        }

        # Count events and calculate success rates
        successful = 0
        total = 0

        for results in self.update_history.values():
            for result in results:
                stats['events'][result.event_type.value] += 1
                stats['priorities'][result.priority.name] += 1

                if 'density_state' in result.details:
                    stats['density_states'][result.details['density_state']] += 1

                successful += int(result.success)
                total += 1

        if total > 0:
            stats['success_rate'] = successful / total

        # Add time step statistics
        stats['time_step_stats'] = {
            'current': self.time_step,
            'min': self.min_time_step,
            'max': self.max_time_step,
            'adaptive': self.adaptive_stepping
        }

        return stats