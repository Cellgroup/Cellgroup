"""
Cellgroup synthetic module for generating synthetic microscopy data.

This module provides tools for modeling, simulating, and analyzing cell nuclei,
clusters, and fluorescence distributions in both 2D and 3D spaces.
"""

# Import core classes
from .cluster import Cluster
from .nucleus import Nucleus
from .space import Space
from .sample import Sample
from .nucleus_fp_distribution import NucleusFluorophoreDistribution
from .utils import Status


__all__ = [
    # Core classes
    "Cluster",
    "Nucleus",
    "Space",
    "Sample",
    "NucleusFluorophoreDistribution",
    "Status",

]