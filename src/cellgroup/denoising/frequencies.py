"""
Module for analyzing frequency patterns in fluorescence microscopy images.
"""


from .utils import load_and_normalize_image, perform_fft_analysis

import numpy as np
from scipy import fftpack
from skimage import io, exposure
import matplotlib.pyplot as plt
import tifffile


def load_and_enhance_image(image_path):
    """Load and enhance image contrast."""
    img = tifffile.imread(image_path)
    img = img.astype(float)
    p2, p98 = np.percentile(img, (2, 98))
    img_enhanced = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_enhanced


def analyze_frequencies_metrics(image_path):
    """
    Focused frequency analysis showing key aspects of the frequency domain.
    """
    # Load and enhance image
    img_enhanced = load_and_enhance_image(image_path)

    # FFT Analysis
    fft2 = fftpack.fft2(img_enhanced)
    fft2_shifted = fftpack.fftshift(fft2)
    magnitude_spectrum = np.abs(fft2_shifted)

    # Calculate mean before any enhancement
    mean_freq = np.mean(magnitude_spectrum)

    # Enhance frequency magnitude visualization
    log_magnitude = np.log1p(magnitude_spectrum)
    p2, p98 = np.percentile(log_magnitude, (2, 98))
    magnitude_enhanced = exposure.rescale_intensity(log_magnitude, in_range=(p2, p98))

    # Compute directional analysis
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    angles = np.arctan2(y, x)

    # Compute directional profile with more bins for smoother visualization
    angle_bins = np.linspace(-np.pi, np.pi, 181)[:-1]  # 180 bins for degrees
    directional_profile = np.zeros_like(angle_bins)
    for i in range(len(angle_bins)):
        if i < len(angle_bins) - 1:
            mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        else:
            mask = (angles >= angle_bins[i]) | (angles < angle_bins[0])
        directional_profile[i] = np.mean(magnitude_spectrum[mask])

    return {
        'image': img_enhanced,
        'magnitude_spectrum': magnitude_enhanced,
        'directional_profile': directional_profile,
        'angle_bins': np.degrees(angle_bins),  # Convert to degrees
        'freq_distribution': log_magnitude.ravel(),
        'mean_freq': mean_freq
    }


def plot_frequency_analysis(image_path):
    """
    Create focused visualization with four key plots.
    """
    results = analyze_frequencies_metrics(image_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    # Enhanced contrast image
    ax1.imshow(results['image'], cmap='gray')
    ax1.set_title('Contrast Enhanced Image')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')

    # Enhanced frequency magnitude
    im2 = ax2.imshow(results['magnitude_spectrum'], cmap='viridis')
    ax2.set_title(f'Frequency Magnitude\nMean: {results["mean_freq"]:.3f}')
    plt.colorbar(im2, ax=ax2)

    # Frequency distribution histogram
    ax3.hist(results['freq_distribution'], bins=100, color='blue', alpha=0.7)
    ax3.set_title('Frequency Distribution')
    ax3.set_xlabel('Log Magnitude')
    ax3.set_ylabel('Count')

    # Directional strength
    ax4.plot(results['angle_bins'], results['directional_profile'])
    ax4.set_title('Directional Frequency Strength')
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True)

    plt.tight_layout()
    return fig, results


def analyze_frequencies(image_path):
    """Wrapper function for frequency analysis"""
    fig, results = plot_frequency_analysis(image_path)
    return results, fig


