"""Fixtures and configuration for the cellgroup test suite."""

import gc
import numpy as np
import pytest
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from cellgroup.synthetic.space import Space
from cellgroup.synthetic.nucleus import Nucleus
from cellgroup.synthetic.cluster import Cluster
from cellgroup.synthetic.sample import Sample
from cellgroup.synthetic.controller import SimulationController, SimulationConfig
from cellgroup.synthetic.physics.update import UpdateCoordinator


# Test Environment Setup and Cleanup
@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment before each test."""
    # Setup
    np.random.seed(42)  # Ensure reproducibility
    # Create any necessary temporary directories
    temp_dir = Path("./temp_test_data")
    temp_dir.mkdir(exist_ok=True)

    yield

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    gc.collect()


# Configuration Generators
@pytest.fixture
def random_space_config():
    """Generate random but valid space configurations."""

    def _generate(dim: int = 2, min_size: int = 50, max_size: int = 200) -> Dict:
        size = tuple(np.random.randint(min_size, max_size) for _ in range(dim))
        scale = tuple(np.random.uniform(0.5, 2.0) for _ in range(dim))
        return {'size': size, 'scale': scale}

    return _generate


@pytest.fixture
def random_nucleus_params():
    """Generate random but valid nucleus parameters."""

    def _generate(dim: int = 2, space_size: Optional[Tuple] = None) -> Dict:
        if space_size is None:
            space_size = tuple(100 for _ in range(dim))

        return {
            'idx': np.random.randint(1, 1000),
            'time': 0,
            'centroid': tuple(np.random.uniform(10, s - 10) for s in space_size),
            'semi_axes': tuple(np.random.uniform(3, 8) for _ in range(dim)),
            'raw_int_density': np.random.uniform(500, 1500),
            'growth_rate': np.random.uniform(0.05, 0.15),
            'max_size': np.random.uniform(800, 1200),
            'min_division_size': np.random.uniform(400, 600),
            'min_viable_size': np.random.uniform(30, 70)
        }

    return _generate


# Basic Configurations
@pytest.fixture
def space_2d_config() -> Dict[str, Any]:
    """Basic 2D space configuration."""
    return {
        'size': (100, 100),
        'scale': (1.0, 1.0)
    }


@pytest.fixture
def space_3d_config() -> Dict[str, Any]:
    """Basic 3D space configuration."""
    return {
        'size': (50, 100, 100),
        'scale': (1.0, 1.0, 1.0)
    }


# Complex Test Scenarios
@pytest.fixture
def test_scenarios() -> Dict[str, Dict]:
    """Different test scenarios for comprehensive testing."""
    return {
        'basic': {
            'n_clusters': 2,
            'nuclei_per_cluster': 3,
            'space_size': (100, 100),
            'min_separation': 20.0
        },
        'dense': {
            'n_clusters': 10,
            'nuclei_per_cluster': 20,
            'space_size': (150, 150),
            'min_separation': 15.0
        },
        'sparse': {
            'n_clusters': 3,
            'nuclei_per_cluster': 5,
            'space_size': (300, 300),
            'min_separation': 50.0
        },
        'extreme': {
            'n_clusters': 20,
            'nuclei_per_cluster': 30,
            'space_size': (400, 400),
            'min_separation': 10.0
        }
    }


# Space Fixtures
@pytest.fixture
def space_2d(space_2d_config) -> Space:
    """Create a basic 2D space for testing."""
    try:
        space = Space(**space_2d_config)
        if not all(s > 0 for s in space.size):
            raise ValueError("Space dimensions must be positive")
        if not all(s > 0 for s in space.scale):
            raise ValueError("Space scale must be positive")
        return space
    except Exception as e:
        pytest.fail(f"Failed to create 2D space: {str(e)}")


@pytest.fixture
def space_3d(space_3d_config) -> Space:
    """Create a basic 3D space for testing."""
    try:
        space = Space(**space_3d_config)
        if not all(s > 0 for s in space.size):
            raise ValueError("Space dimensions must be positive")
        if not all(s > 0 for s in space.scale):
            raise ValueError("Space scale must be positive")
        return space
    except Exception as e:
        pytest.fail(f"Failed to create 3D space: {str(e)}")


# Nucleus Fixtures
@pytest.fixture
def nucleus_2d(space_2d) -> Nucleus:
    """Create a basic 2D nucleus for testing."""
    try:
        nucleus = Nucleus(
            idx=1,
            time=0,
            centroid=(50, 50),
            semi_axes=(5.0, 4.0),
            angle_x=0.0,
            raw_int_density=1000.0,
            growth_rate=0.1,
            max_size=1000.0,
            min_division_size=500.0,
            min_viable_size=50.0,
            max_age=200
        )
        if not space_2d.is_inside(nucleus.centroid):
            raise ValueError("Nucleus must be inside space bounds")
        return nucleus
    except Exception as e:
        pytest.fail(f"Failed to create 2D nucleus: {str(e)}")


@pytest.fixture
def nucleus_3d(space_3d) -> Nucleus:
    """Create a basic 3D nucleus for testing."""
    try:
        nucleus = Nucleus(
            idx=1,
            time=0,
            centroid=(25, 50, 50),
            semi_axes=(5.0, 4.0, 4.0),
            angle_x=0.0,
            angle_y=0.0,
            angle_z=0.0,
            raw_int_density=1000.0,
            growth_rate=0.1,
            max_size=1000.0,
            min_division_size=500.0,
            min_viable_size=50.0,
            max_age=200
        )
        if not space_3d.is_inside(nucleus.centroid):
            raise ValueError("Nucleus must be inside space bounds")
        return nucleus
    except Exception as e:
        pytest.fail(f"Failed to create 3D nucleus: {str(e)}")


# Cluster Fixtures
@pytest.fixture
def cluster_2d(nucleus_2d) -> Cluster:
    """Create a basic 2D cluster for testing."""
    try:
        return Cluster(
            idx=1,
            nuclei=[nucleus_2d],
            max_radius=(20, 20),
            concentration=0.5,
            repulsion_strength=50.0,
            adhesion_strength=10.0,
            noise_strength=1.0
        )
    except Exception as e:
        pytest.fail(f"Failed to create 2D cluster: {str(e)}")


@pytest.fixture
def cluster_3d(nucleus_3d) -> Cluster:
    """Create a basic 3D cluster for testing."""
    try:
        return Cluster(
            idx=1,
            nuclei=[nucleus_3d],
            max_radius=(20, 20, 20),
            concentration=0.5,
            repulsion_strength=50.0,
            adhesion_strength=10.0,
            noise_strength=1.0
        )
    except Exception as e:
        pytest.fail(f"Failed to create 3D cluster: {str(e)}")


# Sample Fixtures
@pytest.fixture
def sample_2d(space_2d, cluster_2d) -> Sample:
    """Create a basic 2D sample for testing."""
    try:
        return Sample(
            clusters=[cluster_2d],
            space=space_2d,
            cluster_interaction_range=100.0,
            cluster_merge_threshold=20.0
        )
    except Exception as e:
        pytest.fail(f"Failed to create 2D sample: {str(e)}")


@pytest.fixture
def sample_3d(space_3d, cluster_3d) -> Sample:
    """Create a basic 3D sample for testing."""
    try:
        return Sample(
            clusters=[cluster_3d],
            space=space_3d,
            cluster_interaction_range=100.0,
            cluster_merge_threshold=20.0
        )
    except Exception as e:
        pytest.fail(f"Failed to create 3D sample: {str(e)}")


# Simulation Configuration Fixtures
@pytest.fixture
def basic_sim_config() -> SimulationConfig:
    """Create a basic simulation configuration."""
    try:
        return SimulationConfig(
            duration=100,
            time_step=1.0,
            save_frequency=10,
            space_size=(100, 100),
            space_scale=(1.0, 1.0),
            initial_clusters=2,
            nuclei_per_cluster=3,
            min_cluster_separation=20.0,
            noise_strength=0.1,
            repulsion_strength=10.0,
            adhesion_strength=5.0,
            growth_rate=0.1,
            division_threshold=100.0,
            death_probability=0.01,
            max_snapshots=10,
            performance_monitoring=True
        )
    except Exception as e:
        pytest.fail(f"Failed to create simulation config: {str(e)}")


@pytest.fixture
def controller(basic_sim_config) -> SimulationController:
    """Create a basic simulation controller for testing."""
    try:
        return SimulationController(config=basic_sim_config)
    except Exception as e:
        pytest.fail(f"Failed to create controller: {str(e)}")


# Mock Objects
@pytest.fixture
def mock_update_coordinator(mocker):
    """Create mock UpdateCoordinator for testing."""
    coordinator = mocker.MagicMock(spec=UpdateCoordinator)
    coordinator.time_step = 1.0
    coordinator.update.return_value = True
    return coordinator


# Helper Functions
@pytest.fixture
def assert_position_in_bounds():
    """Helper function to check if a position is within space bounds."""

    def _check(position: Tuple[float, ...], space: Space) -> bool:
        return all(0 <= p <= s for p, s in zip(position, space.size))

    return _check


@pytest.fixture
def assert_valid_nucleus():
    """Helper function to validate nucleus properties."""

    def _check(nucleus: Nucleus) -> bool:
        return (
                nucleus.idx >= 0 and
                all(a > 0 for a in nucleus.semi_axes) and
                (nucleus.raw_int_density is None or nucleus.raw_int_density >= 0)
        )

    return _check


@pytest.fixture
def create_test_data():
    """Helper function to create test data files."""

    def _create(filename: str, content: str) -> Path:
        path = Path("./temp_test_data") / filename
        path.write_text(content)
        return path

    return _create


# Test Data
@pytest.fixture
def sample_trajectory_data() -> List[Dict]:
    """Generate sample trajectory data for testing."""
    return [
        {'time': t, 'x': np.sin(t / 10), 'y': np.cos(t / 10)}
        for t in range(100)
    ]


@pytest.fixture
def sample_growth_data() -> List[Dict]:
    """Generate sample growth data for testing."""
    return [
        {'time': t, 'size': 100 * (1 + 0.1 * t)}
        for t in range(50)
    ]