"""
Module for analyzing frequency patterns in fluorescence microscopy images.
"""

import numpy as np
from .utils import load_and_normalize_image, perform_fft_analysis


def analyze_frequencies(image_path, row_step=10, col_step=10, patch_size=50):
    """
    Analyze both spatial frequencies and noise patterns.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image
    row_step : int, optional
        Step size for row sampling
    col_step : int, optional
        Step size for column sampling
    patch_size : int, optional
        Size of patches for noise analysis

    Returns
    -------
    dict
        Analysis results containing frequency and noise measurements
    ndarray
        Normalized image
    """
    # Load and normalize image using utility function
    normalized_image, _ = load_and_normalize_image(image_path)

    # Get FFT analysis results using utility function
    fft_results = perform_fft_analysis(normalized_image)

    # Noise Analysis (specific to this module)
    row_profile = np.mean(normalized_image[::row_step], axis=1)
    col_profile = np.mean(normalized_image[:, ::col_step], axis=0)
    patches = normalized_image[::patch_size, ::patch_size]
    noise_level = np.std(patches)

    return {
        'frequency_analysis': {
            'horizontal_peaks': fft_results['horizontal_profile'][fft_results['horizontal_peaks']].tolist(),
            'vertical_peaks': fft_results['vertical_profile'][fft_results['vertical_peaks']].tolist(),
            'magnitude_spectrum': fft_results['magnitude_spectrum'],
            'horizontal_profile': fft_results['horizontal_profile'],
            'vertical_profile': fft_results['vertical_profile']
        },
        'noise_analysis': {
            'row_variation': float(np.std(row_profile)),
            'col_variation': float(np.std(col_profile)),
            'noise_level': float(noise_level),
            'profiles': (row_profile, col_profile)
        }
    }, normalized_image


# Visualization function can stay the same as it's specific to this module
def plot_frequency_analysis(image_path):
    """Create visualization of both frequency and noise analysis."""
    results, normalized_image = analyze_frequencies(image_path)

    # Rest of your plotting code stays the same...
    # (The plotting code is correctly separated from the analysis)


def analyze_microscopy_frequencies(image_path):
    """Wrapper function for easy analysis of a single image"""
    results, _ = analyze_frequencies(image_path)
    fig = plot_frequency_analysis(image_path)

    # Rest of your print statements stay the same...

    return results, fig