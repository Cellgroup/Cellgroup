"""
Module for analyzing gradient patterns in fluorescence microscopy images.
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from .utils import load_and_normalize_image


def analyze_gradients(image_path):
    """
    Analyze gradient patterns in microscopy images with enhanced contrast.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image

    Returns
    -------
    dict
        Analysis results containing gradient statistics and components
    matplotlib.figure.Figure
        Figure containing visualization plots
    """
    # Load and normalize image using utility function
    img_rescaled, _ = load_and_normalize_image(image_path)

    # Calculate gradients using Sobel operators
    gradient_x = ndimage.sobel(img_rescaled, axis=0)
    gradient_y = ndimage.sobel(img_rescaled, axis=1)

    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    magnitude_normalized = magnitude / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    # Calculate gradient statistics
    mean_magnitude = np.mean(magnitude_normalized)
    significant_mask = magnitude_normalized > 0.1
    median_direction = np.median(direction[significant_mask])
    gradient_strength = np.percentile(magnitude_normalized, 95)

    # Create visualization
    fig = create_gradient_plots(
        img_rescaled,
        magnitude_normalized,
        direction,
        gradient_x,
        gradient_y,
        mean_magnitude,
        median_direction
    )

    # Prepare results
    results = {
        'gradient_stats': {
            'mean_magnitude': float(mean_magnitude),
            'median_direction': float(median_direction),
            'gradient_strength': float(gradient_strength)
        },
        'gradient_components': {
            'magnitude': magnitude_normalized,
            'direction': direction,
            'gradient_x': gradient_x,
            'gradient_y': gradient_y
        }
    }

    return results, fig


def create_gradient_plots(img_rescaled, magnitude_normalized, direction,
                         gradient_x, gradient_y, mean_magnitude, median_direction):
    """Create focused gradient analysis plots showing only key visualizations."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax1.imshow(img_rescaled, cmap='gray')
    ax1.set_title('Contrast Enhanced Image')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')

    # Gradient magnitude
    im2 = ax2.imshow(magnitude_normalized, cmap='viridis')
    ax2.set_title(f'Gradient Magnitude\nMean: {mean_magnitude:.3f}')
    plt.colorbar(im2, ax=ax2)

    # Direction histogram
    significant_mask = magnitude_normalized > 0.1
    significant_directions = direction[significant_mask]
    if len(significant_directions) > 0:
        ax3.hist(significant_directions.ravel(), bins=180, range=(-180, 180))
    ax3.set_title('Gradient Direction Distribution\n(Significant Gradients Only)')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    return fig