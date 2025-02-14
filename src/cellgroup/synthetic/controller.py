from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import time
import psutil
import numpy as np
from pydantic import BaseModel, Field

from cellgroup.configs import SimulationConfig
from cellgroup.synthetic.sample import Sample
from cellgroup.synthetic.space import Space
from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.cluster import Cluster
from cellgroup.synthetic.physics.update import UpdateCoordinator
from cellgroup.synthetic.spatial.grid import SpatialGrid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """Possible states of the simulation."""
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class SimulationEvent:
    """Represents significant events during simulation."""
    timestamp: float
    event_type: str
    details: Dict[str, Any]
    entities: List[int]  # IDs of involved entities

    def __str__(self) -> str:
        return f"[{self.timestamp:.2f}] {self.event_type}: {len(self.entities)} entities"


class SimulationController(BaseModel):
    """Controls and manages the cell simulation."""

    config: SimulationConfig
    state: SimulationState = Field(default=SimulationState.INITIALIZED)
    current_time: float = Field(default=0.0)

    # Core components
    space: Optional[Space] = Field(default=None)
    sample: Optional[Sample] = Field(default=None)
    update_coordinator: Optional[UpdateCoordinator] = Field(default=None)

    # Tracking and analysis
    events: List[SimulationEvent] = Field(default_factory=list)
    snapshots: Dict[int, Dict] = Field(default_factory=dict)
    statistics: Dict[str, List[float]] = Field(default_factory=dict)
    performance_metrics: Dict[str, List[float]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize simulation components."""
        super().__init__(**data)
        self._initialize_simulation()

    @contextmanager
    def _state_transition(self, new_state: SimulationState):
        """Safely manage state transitions."""
        old_state = self.state
        try:
            if self._validate_state_transition(old_state, new_state):
                self.state = new_state
                logger.info(f"State transition: {old_state.name} -> {new_state.name}")
                yield
            else:
                raise ValueError(f"Invalid state transition: {old_state} -> {new_state}")
        except Exception as e:
            self.state = SimulationState.ERROR
            self._log_event('state_transition_error', {'error': str(e)}, [])
            raise

    def _validate_state_transition(self, from_state: SimulationState, to_state: SimulationState) -> bool:
        """Validate state transitions."""
        valid_transitions = {
            SimulationState.INITIALIZED: [SimulationState.RUNNING, SimulationState.ERROR],
            SimulationState.RUNNING: [SimulationState.PAUSED, SimulationState.COMPLETED, SimulationState.ERROR],
            SimulationState.PAUSED: [SimulationState.RUNNING, SimulationState.ERROR],
            SimulationState.COMPLETED: [SimulationState.INITIALIZED],
            SimulationState.ERROR: [SimulationState.INITIALIZED]
        }
        return to_state in valid_transitions.get(from_state, [])

    def _initialize_simulation(self):
        """Set up initial simulation state."""
        try:
            logger.info("Initializing simulation...")

            # Create space
            self.space = Space(
                size=self.config.space_size,
                scale=self.config.space_scale
            )

            # Initialize update coordinator
            self.update_coordinator = UpdateCoordinator(
                space=self.space,
                time_step=self.config.time_step
            )

            # Create initial sample with clusters
            self.sample = Sample.create_random_sample(
                space=self.space,
                n_clusters=self.config.initial_clusters,
                nuclei_per_cluster=self.config.nuclei_per_cluster,
                min_separation=self.config.min_cluster_separation,
                cluster_interaction_range=self.config.repulsion_strength * 2,
                cluster_merge_threshold=self.config.adhesion_strength * 2
            )

            # Initialize tracking
            self._initialize_statistics()
            self._initialize_performance_metrics()

            logger.info("Simulation initialized successfully")

        except Exception as e:
            self.state = SimulationState.ERROR
            logger.error(f"Simulation initialization failed: {str(e)}")
            raise RuntimeError(f"Simulation initialization failed: {str(e)}")

    def _initialize_statistics(self):
        """Initialize statistical tracking."""
        self.statistics = {
            'time': [],
            'total_nuclei': [],
            'active_clusters': [],
            'mean_cluster_size': [],
            'total_deaths': [],
            'total_divisions': [],
            'mean_growth_rate': [],
            'spatial_density': []
        }

    def _initialize_performance_metrics(self):
        """Initialize performance monitoring."""
        if self.config.performance_monitoring:
            self.performance_metrics = {
                'step_time': [],
                'memory_usage': [],
                'collision_checks': [],
                'update_time': []
            }

    def _update_statistics(self):
        """Update simulation statistics."""
        self.statistics['time'].append(self.current_time)
        self.statistics['total_nuclei'].append(self.sample.total_nuclei)
        self.statistics['active_clusters'].append(self.sample.count)

        if self.sample.count > 0:
            self.statistics['mean_cluster_size'].append(
                self.sample.total_nuclei / self.sample.count
            )
            # Calculate spatial density
            total_space = np.prod(self.space.size)
            total_nucleus_area = sum(
                np.pi * np.prod(nucleus.semi_axes)
                for cluster in self.sample.clusters
                for nucleus in cluster.nuclei
            )
            self.statistics['spatial_density'].append(total_nucleus_area / total_space)
        else:
            self.statistics['mean_cluster_size'].append(0)
            self.statistics['spatial_density'].append(0)

    def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.config.performance_monitoring:
            process = psutil.Process()
            self.performance_metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)  # MB

    def _manage_snapshots(self):
        """Manage snapshot storage to prevent memory issues."""
        if len(self.snapshots) > self.config.max_snapshots:
            # Remove oldest snapshots
            times = sorted(self.snapshots.keys())
            for old_time in times[:-self.config.max_snapshots]:
                del self.snapshots[old_time]

    def _save_snapshot(self):
        """Save current simulation state."""
        if self.current_time % self.config.save_frequency == 0:
            self.snapshots[int(self.current_time)] = {
                'time': self.current_time,
                'sample_state': self.sample.model_dump(),
                'statistics': {k: v[-1] for k, v in self.statistics.items()},
                'events': [e for e in self.events if e.timestamp == self.current_time]
            }
            self._manage_snapshots()

    def _log_event(self, event_type: str, details: Dict[str, Any], entities: List[int]):
        """Log a simulation event."""
        event = SimulationEvent(
            timestamp=self.current_time,
            event_type=event_type,
            details=details,
            entities=entities
        )
        self.events.append(event)
        logger.debug(str(event))

    def step(self) -> bool:
        """Perform a single simulation step."""
        try:
            if self.state not in (SimulationState.INITIALIZED, SimulationState.RUNNING):
                return False

            start_time = time.time()

            with self._state_transition(SimulationState.RUNNING):
                # Update sample
                self.sample.update()

                # Update time
                self.current_time += self.config.time_step

                # Update tracking
                self._update_statistics()
                self._update_performance_metrics()
                self._save_snapshot()

                if self.config.performance_monitoring:
                    self.performance_metrics['step_time'].append(time.time() - start_time)

                return True

        except Exception as e:
            self.state = SimulationState.ERROR
            self._log_event(
                'error',
                {'error': str(e), 'step': self.current_time},
                []
            )
            logger.error(f"Step error at time {self.current_time}: {str(e)}")
            return False

    def run(self, duration: Optional[int] = None) -> bool:
        """Run simulation for specified duration."""
        steps = duration or (self.config.duration - int(self.current_time))

        try:
            for step in range(steps):
                if not self.step():
                    return False

                if self.sample.count == 0:
                    with self._state_transition(SimulationState.COMPLETED):
                        self._log_event(
                            'simulation_completed',
                            {'reason': 'no_active_clusters'},
                            []
                        )
                        return True

                if step % 100 == 0:  # Progress logging
                    logger.info(f"Simulation progress: {step}/{steps} steps completed")

            return True

        except Exception as e:
            self.state = SimulationState.ERROR
            self._log_event(
                'simulation_error',
                {'error': str(e)},
                []
            )
            logger.error(f"Run error: {str(e)}")
            return False

    def pause(self):
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            with self._state_transition(SimulationState.PAUSED):
                self._log_event(
                    'simulation_paused',
                    {'time': self.current_time},
                    []
                )

    def resume(self):
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            with self._state_transition(SimulationState.RUNNING):
                self._log_event(
                    'simulation_resumed',
                    {'time': self.current_time},
                    []
                )

    def reset(self):
        """Reset simulation to initial state."""
        with self._state_transition(SimulationState.INITIALIZED):
            self.cleanup()
            self._initialize_simulation()
            self._log_event(
                'simulation_reset',
                {'time': self.current_time},
                []
            )

    def cleanup(self):
        """Clean up resources when simulation ends."""
        try:
            logger.info("Cleaning up simulation resources...")
            # Clean spatial grid
            if self.update_coordinator and self.update_coordinator.spatial_grid:
                self.update_coordinator.spatial_grid.clear()
            # Clear large data structures
            self.snapshots.clear()
            self.events.clear()
            self.statistics.clear()
            self.performance_metrics.clear()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            self._log_event('cleanup_error', {'error': str(e)}, [])

    def get_statistics(self) -> Dict[str, List[float]]:
        """Get current simulation statistics."""
        return self.statistics

    def get_events(self, start_time: Optional[float] = None) -> List[SimulationEvent]:
        """Get events, optionally filtered by start time."""
        if start_time is None:
            return self.events
        return [e for e in self.events if e.timestamp >= start_time]

    def get_snapshot(self, time: Optional[int] = None) -> Dict:
        """Get simulation snapshot at specified time."""
        if time is None:
            time = int(self.current_time)
        return self.snapshots.get(time, {})

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """Get performance metrics if monitoring is enabled."""
        if not self.config.performance_monitoring:
            return {}
        return self.performance_metrics

    def attempt_recovery(self) -> bool:
        """Attempt to recover from error state."""
        if self.state == SimulationState.ERROR:
            try:
                logger.info("Attempting error recovery...")
                # Save current state
                error_snapshot = self.get_snapshot()
                # Reset core components
                self._initialize_simulation()
                # Restore last good state if possible
                if error_snapshot:
                    self._log_event(
                        'recovery_attempted',
                        {'snapshot_time': error_snapshot.get('time', 0)},
                        []
                    )
                return True
            except Exception as e:
                logger.error(f"Recovery failed: {str(e)}")
                self._log_event('recovery_failed', {'error': str(e)}, [])
                return False
        return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current simulation state."""
        return {
            'state': self.state.name,
            'current_time': self.current_time,
            'total_nuclei': self.sample.total_nuclei if self.sample else 0,
            'active_clusters': self.sample.count if self.sample else 0,
            'events_count': len(self.events),
            'snapshots_count': len(self.snapshots),
            'last_event': str(self.events[-1]) if self.events else None,
            'performance': {
                'memory_usage': self.performance_metrics.get('memory_usage', [-1])[-1],
                'last_step_time': self.performance_metrics.get('step_time', [-1])[-1]
            } if self.config.performance_monitoring else {}
        }

    def validate_state(self) -> bool:
        """Validate current simulation state."""
        try:
            if None in (self.space, self.sample, self.update_coordinator):
                logger.error("Core components not properly initialized")
                return False

            # Check for data consistency
            if self.sample.space is not self.space:
                logger.error("Space reference mismatch")
                return False

            # Validate time consistency
            if self.current_time < 0 or (
                    self.statistics['time'] and
                    self.current_time < self.statistics['time'][-1]
            ):
                logger.error("Time inconsistency detected")
                return False

            return True

        except Exception as e:
            logger.error(f"State validation error: {str(e)}")
            return False

    def export_data(self, format: str = 'dict') -> Dict[str, Any]:
        """Export simulation data in specified format."""
        data = {
            'config': self.config.model_dump(),
            'statistics': self.statistics,
            'events': [{'timestamp': e.timestamp,
                        'type': e.event_type,
                        'details': e.details}
                       for e in self.events],
            'final_state': self.get_snapshot(),
            'performance': self.get_performance_metrics()
        }

        if format == 'dict':
            return data
        else:
            raise ValueError(f"Unsupported export format: {format}")