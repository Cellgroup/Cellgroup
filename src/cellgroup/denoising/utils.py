"""
Utility functions for the Cellgroup denoising module.
Provides common image loading and processing functions used across different analysis modules.
"""

from scipy import fftpack
from scipy.signal import find_peaks
from pathlib import Path
import numpy as np
from skimage import io, exposure
import tifffile
from functools import lru_cache
import logging
import numpy as np
from skimage import io, exposure
import tifffile
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def load_and_normalize_image(image_path):
    """
    Load and normalize microscopy image with contrast enhancement.
    Handles both TIFF and common image formats with robust error checking
    and logging.

    Parameters
    ----------
    image_path : str or Path
        Path to the microscopy image file

    Returns
    -------
    tuple
        (normalized_image, original_image_float), where:
        - normalized_image has enhanced contrast using percentile-based rescaling
        - original_image_float is the original image converted to float

    Raises
    ------
    FileNotFoundError
        If the image file doesn't exist
    ValueError
        If the image is empty or contains invalid data
    RuntimeError
        If both tifffile and skimage fail to load the image

    Examples
    --------
    >>> img_normalized, img_original = load_and_normalize_image("path/to/image.tif")
    >>> print(f"Normalized range: [{img_normalized.min()}, {img_normalized.max()}]")
    """
    # Convert string path to Path object for robust handling
    image_path = Path(image_path)

    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Try loading with tifffile first
    try:
        logger.debug(f"Attempting to load {image_path} with tifffile")
        img = tifffile.imread(str(image_path))
        logger.debug("Successfully loaded image with tifffile")
    except Exception as e:
        logger.debug(f"tifffile failed: {e}, trying skimage.io")
        try:
            # Fall back to skimage if tifffile fails
            img = io.imread(str(image_path))
            logger.debug("Successfully loaded image with skimage.io")
        except Exception as e:
            raise RuntimeError(f"Failed to load image with both tifffile and skimage: {e}")

    # Validate image data
    if img is None or img.size == 0:
        raise ValueError("Loaded image is empty or invalid")

    # Convert to float, handling potential overflow
    img = img.astype(np.float32)

    # Check for invalid values
    if np.any(np.isnan(img)) or np.any(np.isinf(img)):
        logger.warning("Image contains NaN or Inf values - replacing with 0")
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate robust percentiles for contrast enhancement
    try:
        p2, p98 = np.percentile(img, (2, 98))
        img_rescaled = exposure.rescale_intensity(img, in_range=(p2, p98))
    except Exception as e:
        logger.error(f"Error during contrast enhancement: {e}")
        # Fall back to simple normalization if percentile-based fails
        img_min, img_max = img.min(), img.max()
        if img_min != img_max:  # Avoid division by zero
            img_rescaled = (img - img_min) / (img_max - img_min)
        else:
            img_rescaled = img.copy()

    logger.debug(f"Original range: [{img.min():.2f}, {img.max():.2f}]")
    logger.debug(f"Normalized range: [{img_rescaled.min():.2f}, {img_rescaled.max():.2f}]")

    return img_rescaled, img

# Replace the cached_imread function with this
@lru_cache(maxsize=32)
def cached_imread(image_path):
    """
    Load and normalize microscopy image with contrast enhancement.
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


# Then modify analyze_light_intensity to use both returned values
def analyze_light_intensity(image_path):
    """
    Main analysis function with parallel processing.
    """
    img_rescaled, img_float = cached_imread(image_path)

    if img_float.ndim > 2:
        img_float = img_float.mean(axis=2)
        img_rescaled = img_rescaled.mean(axis=2)

    # Use img_rescaled for visualization and img_float for analysis

def perform_fft_analysis(img):
    """
    Optimized FFT analysis.
    """
    fft2 = fftpack.fft2(img)
    fft2_shifted = fftpack.fftshift(fft2)
    magnitude_spectrum = np.abs(fft2_shifted)

    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    horizontal_profile = magnitude_spectrum[center_y, :].copy()
    vertical_profile = magnitude_spectrum[:, center_x].copy()

    h_peaks = find_peaks(horizontal_profile, distance=20)[0]
    v_peaks = find_peaks(vertical_profile, distance=20)[0]

    return magnitude_spectrum, horizontal_profile, vertical_profile, h_peaks, v_peaks



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