"""
Cellgroup Denoising Module
=========================

This module provides tools for analyzing and processing fluorescence microscopy images,
with a focus on denoising and quality assessment.

Main Functions
-------------
analyze_gradients : Analyze intensity gradients in microscopy images
analyze_light_intensity : Analyze light intensity distributions and variations
analyze_frequencies : Analyze frequency components and patterns
analyze_spatial_variations : Analyze spatial correlations and patterns

Examples
--------
>>> from cellgroup.denoising import analyze_gradients
>>> results = analyze_gradients('microscopy_image.tif')
>>> magnitude_map = results['gradient_magnitude']
"""

from .gradients import analyze_gradients
from .intensity import analyze_light_intensity
from .frequencies import analyze_frequencies
from .spatial import analyze_spatial_variations

# Define the public API
__all__ = [
    'analyze_gradients',
    'analyze_light_intensity',
    'analyze_frequencies',
    'analyze_spatial_variations'
]

# Version of the denoising module
__version__ = '0.1.0'

# Module level logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def get_version():
    """Return the version of the denoising module."""
    return __version__

# Optional: Add any module initialization code here
def _initialize():
    """Initialize the denoising module (if needed)."""
    pass

_initialize()