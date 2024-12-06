"""
Utility functions for the Cellgroup denoising module.
Provides common image loading and processing functions used across different analysis modules.
"""

import numpy as np
from skimage import io, exposure
import tifffile
from scipy import fftpack
from scipy.signal import find_peaks
from functools import lru_cache


@lru_cache(maxsize=32)
def load_and_normalize_image(image_path):
    """
    Load and normalize microscopy image with contrast enhancement.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image file

    Returns
    -------
    tuple
        (normalized_image, original_image_float)
    """
    try:
        # Try tifffile first for microscopy images
        img = tifffile.imread(image_path)
    except:
        # Fall back to skimage if tifffile fails
        img = io.imread(image_path)

    img = img.astype(float)
    p2, p98 = np.percentile(img, (2, 98))
    img_rescaled = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescaled, img


def perform_fft_analysis(image):
    """
    Perform FFT analysis of the image.

    Parameters
    ----------
    image : ndarray
        Input image

    Returns
    -------
    dict
        Dictionary containing magnitude spectrum and profiles
    """
    # Calculate FFT
    fft2 = fftpack.fft2(image)
    fft2_shifted = fftpack.fftshift(fft2)
    magnitude_spectrum = np.abs(fft2_shifted)

    # Get central profiles
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    horizontal_profile = magnitude_spectrum[center_y, :]
    vertical_profile = magnitude_spectrum[:, center_x]

    # Find peaks in profiles
    h_peaks = find_peaks(horizontal_profile, distance=20)[0]
    v_peaks = find_peaks(vertical_profile, distance=20)[0]

    return {
        'magnitude_spectrum': magnitude_spectrum,
        'horizontal_profile': horizontal_profile,
        'vertical_profile': vertical_profile,
        'horizontal_peaks': h_peaks,
        'vertical_peaks': v_peaks
    }


def calculate_local_stats(image, window_size=16):
    """
    Calculate local mean and standard deviation.

    Parameters
    ----------
    image : ndarray
        Input image
    window_size : int
        Size of the local window

    Returns
    -------
    tuple
        (local_mean, local_std)
    """
    height, width = image.shape
    local_mean = np.zeros_like(image, dtype=np.float32)
    local_std = np.zeros_like(image, dtype=np.float32)

    for i in range(0, height - window_size + 1, window_size // 2):
        for j in range(0, width - window_size + 1, window_size // 2):
            window = image[i:i + window_size, j:j + window_size]
            local_mean[i:i + window_size, j:j + window_size] = np.mean(window)
            local_std[i:i + window_size, j:j + window_size] = np.std(window)

    return local_mean, local_std