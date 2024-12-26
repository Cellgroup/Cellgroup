"""
Module for analyzing spatial patterns in fluorescence microscopy images.
"""

import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import tifffile


def load_and_normalize_image(image_path):
    original_image = tifffile.imread(image_path)
    image_float = original_image.astype(float)
    p2, p98 = np.percentile(image_float, (2, 98))
    normalized_image = exposure.rescale_intensity(image_float, in_range=(p2, p98))
    return normalized_image, image_float


def analyze_spatial_variations_metrics(normalized_image, row_step=10, col_step=10):
    row_profile = np.mean(normalized_image[::row_step], axis=1)
    col_profile = np.mean(normalized_image[:, ::col_step], axis=0)

    return {
        'spatial_stats': {
            'row_variation': float(np.std(row_profile)),
            'col_variation': float(np.std(col_profile))
        },
        'profiles': {
            'row_profile': row_profile,
            'col_profile': col_profile
        }
    }


def plot_spatial_analysis(image_path):
    normalized_image, _ = load_and_normalize_image(image_path)
    results = analyze_spatial_variations_metrics(normalized_image)

    # Create figure with 3 horizontally arranged subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    im1 = ax1.imshow(normalized_image, cmap='gray')
    ax1.set_title('Normalized Image')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    plt.colorbar(im1, ax=ax1, label='Intensity')

    # Row profile
    ax2.plot(results['profiles']['row_profile'], 'b-')
    ax2.set_title(f"Row Intensity Profile\nVariation: {results['spatial_stats']['row_variation']:.3f}")
    ax2.set_xlabel('Row Number')
    ax2.set_ylabel('Mean Intensity')
    ax2.grid(True)

    # Column profile
    ax3.plot(results['profiles']['col_profile'], 'r-')
    ax3.set_title(f"Column Intensity Profile\nVariation: {results['spatial_stats']['col_variation']:.3f}")
    ax3.set_xlabel('Column Number')
    ax3.set_ylabel('Mean Intensity')
    ax3.grid(True)

    plt.tight_layout()
    return fig, results


def analyze_spatial_variations(image_path):
    fig, results = plot_spatial_analysis(image_path)
    print("\nSpatial Analysis Results:")
    print(f"Row Variation: {results['spatial_stats']['row_variation']:.3f}")
    print(f"Column Variation: {results['spatial_stats']['col_variation']:.3f}")
    return results, fig