"""
Module for analyzing light intensity patterns in fluorescence microscopy images.
"""
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)  # Only show WARNING and higher messages
import numpy as np
from scipy import fftpack
from skimage import io, exposure
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from functools import lru_cache
import concurrent.futures


# Cache for image reading
@lru_cache(maxsize=32)
def cached_imread(image_path):
    return io.imread(image_path)


def fast_local_statistics(img, window_size):
    """
    Optimized calculation of local statistics using numpy operations.
    """
    height, width = img.shape
    local_std = np.zeros_like(img, dtype=np.float32)
    local_mean = np.zeros_like(img, dtype=np.float32)

    for i in range(0, height - window_size + 1, window_size // 2):
        for j in range(0, width - window_size + 1, window_size // 2):
            window = img[i:i + window_size, j:j + window_size]
            mean_val = np.mean(window)
            std_val = np.std(window)

            local_mean[i:i + window_size, j:j + window_size] = mean_val
            local_std[i:i + window_size, j:j + window_size] = std_val

    return local_mean, local_std


def fast_coefficient_variation(patches):
    """
    Optimized calculation of coefficient of variation.
    """
    local_means = np.mean(patches, axis=(2, 3))
    local_stds = np.std(patches, axis=(2, 3))

    local_cv = np.zeros_like(local_means)
    mask = local_means != 0
    local_cv[mask] = local_stds[mask] / local_means[mask]

    return local_cv


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


def analyze_noise_characteristics(img):
    """
    Optimized noise analysis.
    """
    window_size = 16
    smoothed = gaussian_filter(img, sigma=window_size / 4)

    local_mean, local_std = fast_local_statistics(img, window_size)

    patch_shape = (64, 64)
    patches = np.lib.stride_tricks.as_strided(
        img,
        shape=(img.shape[0] // patch_shape[0],
               img.shape[1] // patch_shape[1],
               patch_shape[0], patch_shape[1]),
        strides=(img.strides[0] * patch_shape[0],
                 img.strides[1] * patch_shape[1],
                 img.strides[0],
                 img.strides[1])
    )

    local_cv = fast_coefficient_variation(patches)

    return local_mean, local_std, local_cv


def analyze_light_intensity(image_path):
    """
    Main analysis function with parallel processing.
    """
    img = cached_imread(image_path)
    if img.ndim > 2:
        img = img.mean(axis=2)
    img_float = img.astype(np.float32)

    height, width = img_float.shape
    magnitude_spectrum = np.zeros((height, width), dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fft_future = executor.submit(perform_fft_analysis, img_float)
        noise_future = executor.submit(analyze_noise_characteristics, img_float)

        fft_results = fft_future.result()
        noise_results = noise_future.result()

    magnitude_spectrum, horizontal_profile, vertical_profile, h_peaks, v_peaks = fft_results
    local_mean, local_std, local_cv = noise_results

    fig = create_visualization(img, magnitude_spectrum, horizontal_profile,
                               vertical_profile, local_mean, local_std, local_cv)

    metrics = calculate_metrics(img_float, horizontal_profile,
                                vertical_profile, h_peaks, v_peaks, local_cv)

    return metrics, fig


def create_visualization(img, magnitude_spectrum, horizontal_profile,
                         vertical_profile, local_mean, local_std, local_cv):
    """
    Create visualization with optimized memory usage.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    ax1.imshow(exposure.equalize_hist(img), cmap='gray')
    ax1.set_title('Original Image (Enhanced Contrast)')

    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    log_spectrum = np.log1p(magnitude_spectrum)
    ax2.imshow(log_spectrum, cmap='gray')
    ax2.axhline(y=center_y, color='r', alpha=0.3)
    ax2.axvline(x=center_x, color='r', alpha=0.3)
    ax2.set_title('FFT Magnitude Spectrum with Frequency Profiles')

    valid_pixels = (local_mean != 0) & (local_std != 0)
    if np.sum(valid_pixels) > 10000:
        sample_indices = np.random.choice(np.sum(valid_pixels), 10000, replace=False)
        mean_sample = local_mean[valid_pixels][sample_indices]
        std_sample = local_std[valid_pixels][sample_indices]
        ax3.scatter(mean_sample, std_sample, alpha=0.1, s=1, label='Measured')
    else:
        ax3.scatter(local_mean[valid_pixels], local_std[valid_pixels],
                    alpha=0.1, s=1, label='Measured')

    x_range = np.linspace(0, np.max(local_mean[valid_pixels]), 100)
    ax3.plot(x_range, np.sqrt(x_range), 'r-', label='Theoretical Poisson')
    ax3.set_xlabel('Local Mean Intensity')
    ax3.set_ylabel('Local Standard Deviation')
    ax3.set_title('Noise Characteristics vs Theoretical Poisson')
    ax3.legend()

    im4 = ax4.imshow(local_cv, cmap='viridis')
    plt.colorbar(im4, ax=ax4)
    ax4.set_title('Local Coefficient of Variation')

    plt.tight_layout()
    return fig


def calculate_metrics(img_float, horizontal_profile, vertical_profile,
                      h_peaks, v_peaks, local_cv):
    """
    Calculate metrics using vectorized operations.
    """
    return {
        'horizontal_band_strengths': horizontal_profile[h_peaks].tolist(),
        'vertical_band_strengths': vertical_profile[v_peaks].tolist(),
        'mean_intensity': float(np.mean(img_float)),
        'std_intensity': float(np.std(img_float)),
        'mean_cv': float(np.mean(local_cv)),
        'max_cv': float(np.max(local_cv)),
        'intensity_range': (float(np.min(img_float)), float(np.max(img_float))),
        'signal_to_noise': float(np.mean(img_float) / np.std(img_float)) if np.std(img_float) != 0 else float('inf')
    }
