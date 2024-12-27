# Cellgroup

Cellgroup is a Python library for synthetic cell imaging data generation and analysis. It provides tools for modeling, simulating, and analyzing cell nuclei, clusters, and fluorescence distributions.

## Core Components

### Space (`space.py`)

The `Space` class defines the coordinate system and physical properties of the simulation environment:

- Handles 2D and 3D spaces with configurable dimensions
- Manages coordinate conversions between pixels and microns
- Provides utility functions for:
  - Position validation
  - Distance calculations
  - Neighborhood computations
  - Grid generation
  - Random position generation

### Nucleus (`nucleus.py`)

The `Nucleus` class represents individual cell nuclei with biological properties and behaviors:

#### Base Nucleus Features:
- Core geometric properties (position, size, orientation)
- Growth dynamics and cell cycle modeling
- Division and death mechanics
- Lineage tracking
- Intensity measurements

#### NucleusFluorophoreDistribution Features:
- Extends `Nucleus` with fluorophore density distributions
- Supports multiple distribution types:
  - Gaussian
  - Ring-like
  - Uniform
- Configurable intensity patterns and noise
- Realistic fluorescence rendering

### Cluster (`cluster.py`)

The `Cluster` class manages groups of interacting nuclei:

- Maintains collections of nuclei
- Handles inter-nuclear forces:
  - Repulsion
  - Adhesion
  - Random motion
- Cluster-level properties:
  - Geometric measurements
  - Density calculations
  - Evolution tracking
- Visualization capabilities

### Sample (`sample.py`)

The `Sample` class orchestrates multiple clusters in a shared environment:

- Multi-cluster management
- Cluster interaction handling:
  - Proximity detection
  - Merging mechanics
- Global statistics and metrics
- Time evolution tracking
- Integrated visualization

## Key Features

1. **Realistic Cell Modeling**
   - Biologically-inspired growth dynamics
   - Cell division and death mechanics
   - Fluorescence distribution patterns

2. **Spatial Organization**
   - Multi-scale coordinate system
   - Physical force modeling
   - Cluster formation and evolution

3. **Customizable Parameters**
   - Growth rates
   - Division thresholds
   - Interaction strengths
   - Distribution patterns

4. **Analysis Tools**
   - Geometric measurements
   - Intensity analysis
   - Population statistics
   - Cluster metrics

## Example Usage

```python
# Create a 2D simulation space
space = Space.create_2D(width=1024, height=1024, scale_xy=0.5)

# Create a sample with multiple clusters
sample = Sample.create_random_sample(
    space=space,
    n_clusters=3,
    nuclei_per_cluster=50,
    min_separation=100.0
)

# Run simulation for multiple timesteps
for t in range(100):
    sample.update()
    image = sample.render()
    metrics = sample.get_cluster_metrics()