import numpy as np
from scipy import fftpack
from skimage import io, exposure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tifffile
from scipy.ndimage import gaussian_filter


def load_and_normalize_image(image_path):
    """Load and normalize image."""
    original_image = tifffile.imread(image_path)
    image_float = original_image.astype(float)
    # Enhance contrast using percentile-based scaling
    p2, p98 = np.percentile(image_float, (2, 98))
    normalized_image = exposure.rescale_intensity(image_float, in_range=(p2, p98))
    return normalized_image, image_float


def calculate_spatial_autocorrelation(image, max_distance=50):
    """
    Calculate spatial autocorrelation up to a maximum distance.

    Args:
        image: 2D numpy array of the image
        max_distance: Maximum distance to calculate autocorrelation

    Returns:
        distances: Array of distances
        autocorr: Array of autocorrelation values
    """
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Calculate mean and variance
    image_mean = np.mean(image)
    image_var = np.var(image)

    # Create distance matrix
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    distances = np.sqrt(x * x + y * y)

    # Calculate autocorrelation for each distance
    distance_values = np.arange(0, max_distance)
    autocorr_values = []

    for d in distance_values:
        mask = (distances >= d - 0.5) & (distances < d + 0.5)
        if mask.any():
            # Calculate autocorrelation for this distance
            pixels = image[mask]
            shifted_pixels = image[mask]
            covariance = np.mean((pixels - image_mean) * (shifted_pixels - image_mean))
            autocorr = covariance / image_var if image_var != 0 else 0
            autocorr_values.append(autocorr)
        else:
            autocorr_values.append(0)

    return distance_values, np.array(autocorr_values)


def analyze_spatial_variations(normalized_image, row_step=10, col_step=10, patch_size=50):
    """
    Enhanced spatial analysis with better error handling.
    """
    # Previous implementation remains the same
    row_profile = np.mean(normalized_image[::row_step], axis=1)
    col_profile = np.mean(normalized_image[:, ::col_step], axis=0)

    patches = normalized_image[::patch_size, ::patch_size]
    noise_level = np.std(patches) if patches.size > 0 else 0

    if normalized_image.size > 0:
        signal_threshold = np.percentile(normalized_image, 90)
        background_threshold = np.percentile(normalized_image, 10)
        signal_regions = normalized_image > signal_threshold
        background_regions = normalized_image < background_threshold

        if np.any(signal_regions) and np.any(background_regions):
            signal_mean = np.mean(normalized_image[signal_regions])
            background_mean = np.mean(normalized_image[background_regions])
            signal_range = signal_mean - background_mean
            snr = signal_range / noise_level if noise_level > 0 else 0
        else:
            signal_mean = background_mean = signal_range = snr = 0
    else:
        signal_mean = background_mean = signal_range = snr = 0

    # Calculate spatial autocorrelation
    distances, autocorr = calculate_spatial_autocorrelation(normalized_image)

    # Calculate correlation length (distance where autocorrelation drops to 1/e)
    correlation_length = 0
    if len(autocorr) > 0:
        threshold = 1 / np.e
        for i, ac in enumerate(autocorr):
            if ac < threshold:
                correlation_length = distances[i]
                break

    return {
        'spatial_stats': {
            'row_variation': float(np.std(row_profile)),
            'col_variation': float(np.std(col_profile)),
            'noise_level': float(noise_level),
            'snr': float(snr),
            'correlation_length': float(correlation_length)
        },
        'profiles': {
            'row_profile': row_profile,
            'col_profile': col_profile
        },
        'intensity_stats': {
            'signal_mean': float(signal_mean),
            'background_mean': float(background_mean),
            'signal_range': float(signal_range)
        },
        'autocorrelation': {
            'distances': distances,
            'values': autocorr
        }
    }


def plot_spatial_analysis(image_path):
    """
    Create comprehensive visualization of spatial analysis.
    """
    normalized_image, _ = load_and_normalize_image(image_path)
    results = analyze_spatial_variations(normalized_image)

    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(4, 2, figure=fig)

    # Original image with colorbar
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(normalized_image, cmap='gray')
    ax1.set_title('Normalized Image')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    plt.colorbar(im1, ax=ax1, label='Intensity')

    # Row profile
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results['profiles']['row_profile'], 'b-', label='Row Profile')
    ax2.set_title(f"Row Intensity Profile\nVariation: {results['spatial_stats']['row_variation']:.3f}")
    ax2.set_xlabel('Row Number')
    ax2.set_ylabel('Mean Intensity')
    ax2.grid(True)

    # Column profile
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results['profiles']['col_profile'], 'r-', label='Column Profile')
    ax3.set_title(f"Column Intensity Profile\nVariation: {results['spatial_stats']['col_variation']:.3f}")
    ax3.set_xlabel('Column Number')
    ax3.set_ylabel('Mean Intensity')
    ax3.grid(True)

    # Intensity histogram
    ax4 = fig.add_subplot(gs[2, :])
    ax4.hist(normalized_image.ravel(), bins=100, density=True)
    ax4.set_title(f"Intensity Distribution\nSNR: {results['spatial_stats']['snr']:.3f}")
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Frequency')
    ax4.grid(True)

    # Spatial autocorrelation
    ax5 = fig.add_subplot(gs[3, :])
    ax5.plot(results['autocorrelation']['distances'],
             results['autocorrelation']['values'],
             'g-', label='Spatial Autocorrelation')
    ax5.axhline(y=1 / np.e, color='r', linestyle='--',
                label='1/e threshold')
    ax5.set_title(
        f"Spatial Autocorrelation\nCorrelation Length: {results['spatial_stats']['correlation_length']:.1f} pixels")
    ax5.set_xlabel('Distance (pixels)')
    ax5.set_ylabel('Autocorrelation')
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout()
    return fig, results


def analyze_microscopy_spatial(image_path):
    """Wrapper function for spatial analysis"""
    fig, results = plot_spatial_analysis(image_path)

    print("\nSpatial Analysis Results:")
    print(f"Row Variation: {results['spatial_stats']['row_variation']:.3f}")
    print(f"Column Variation: {results['spatial_stats']['col_variation']:.3f}")
    print(f"Noise Level: {results['spatial_stats']['noise_level']:.3f}")
    print(f"Signal-to-Noise Ratio: {results['spatial_stats']['snr']:.3f}")
    print(f"Signal Range: {results['intensity_stats']['signal_range']:.3f}")
    print(f"Correlation Length: {results['spatial_stats']['correlation_length']:.1f} pixels")

    return results, fig


if __name__ == "__main__":
    try:
        image_path = '/EXP2111_A06_D#07_T0003_C01.tif'  # Replace with your image path
        results, fig = analyze_microscopy_spatial(image_path)
        plt.show()

    except Exception as e:
        print(f"Error analyzing image: {str(e)}")