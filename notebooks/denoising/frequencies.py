import numpy as np
from scipy import fftpack
from skimage import io, exposure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tifffile


def load_and_normalize_image(image_path):
    """Load and normalize the image."""
    original_image = tifffile.imread(image_path)
    image_float = original_image.astype(float)
    img_min = image_float.min()
    img_max = image_float.max()
    normalized_image = (image_float - img_min) / (img_max - img_min)
    return normalized_image, image_float


def analyze_frequencies(image_path, row_step=10, col_step=10, patch_size=50):
    """
    Analyze both spatial frequencies and noise patterns.
    """
    # Load and normalize image
    normalized_image, _ = load_and_normalize_image(image_path)

    # FFT Analysis
    fft2 = fftpack.fft2(normalized_image)
    fft2_shifted = fftpack.fftshift(fft2)
    magnitude_spectrum = np.abs(fft2_shifted)

    # Get frequency profiles
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    horizontal_profile = magnitude_spectrum[center_y, :]
    vertical_profile = magnitude_spectrum[:, center_x]

    # Noise Analysis
    row_profile = np.mean(normalized_image[::row_step], axis=1)
    col_profile = np.mean(normalized_image[:, ::col_step], axis=0)
    patches = normalized_image[::patch_size, ::patch_size]
    noise_level = np.std(patches)

    # Find peaks in frequency profiles
    h_peaks = find_peaks(horizontal_profile, distance=20)[0]
    v_peaks = find_peaks(vertical_profile, distance=20)[0]

    return {
        'frequency_analysis': {
            'horizontal_peaks': horizontal_profile[h_peaks].tolist(),
            'vertical_peaks': vertical_profile[v_peaks].tolist(),
            'magnitude_spectrum': magnitude_spectrum,
            'horizontal_profile': horizontal_profile,
            'vertical_profile': vertical_profile
        },
        'noise_analysis': {
            'row_variation': float(np.std(row_profile)),
            'col_variation': float(np.std(col_profile)),
            'noise_level': float(noise_level),
            'profiles': (row_profile, col_profile)
        }
    }, normalized_image


def plot_frequency_analysis(image_path):
    """
    Create visualization of both frequency and noise analysis.
    """
    results, normalized_image = analyze_frequencies(image_path)

    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 2, figure=fig)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(normalized_image, cmap='gray')
    ax1.set_title('Normalized Image')

    # FFT magnitude spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    magnitude_spectrum = results['frequency_analysis']['magnitude_spectrum']
    log_spectrum = np.log1p(magnitude_spectrum)
    ax2.imshow(log_spectrum, cmap='gray')
    ax2.set_title('FFT Magnitude Spectrum')

    # Horizontal frequency profile
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(results['frequency_analysis']['horizontal_profile'])
    ax3.set_title(f"Horizontal Frequency Profile\nRow Variation: {results['noise_analysis']['row_variation']:.3f}")

    # Vertical frequency profile
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(results['frequency_analysis']['vertical_profile'])
    ax4.set_title(f"Vertical Frequency Profile\nColumn Variation: {results['noise_analysis']['col_variation']:.3f}")

    # Noise distribution
    ax5 = fig.add_subplot(gs[2, 1])
    noise_profile = results['noise_analysis']['profiles'][0]
    ax5.hist(noise_profile, bins=50)
    ax5.set_title(f"Noise Distribution\nNoise Level: {results['noise_analysis']['noise_level']:.3f}")

    plt.tight_layout()
    return fig, results


def analyze_microscopy_frequencies(image_path):
    """Wrapper function for easy analysis of a single image"""
    results, _ = analyze_frequencies(image_path)
    fig = plot_frequency_analysis(image_path)

    print("\nAnalysis Results:")
    print(f"Row Variation: {results['noise_analysis']['row_variation']:.3f}")
    print(f"Column Variation: {results['noise_analysis']['col_variation']:.3f}")
    print(f"Noise Level: {results['noise_analysis']['noise_level']:.3f}")
    print(f"Number of significant horizontal peaks: {len(results['frequency_analysis']['horizontal_peaks'])}")
    print(f"Number of significant vertical peaks: {len(results['frequency_analysis']['vertical_peaks'])}")

    return results, fig


if __name__ == "__main__":
    try:
        image_path = '/EXP2111_A06_D#07_T0003_C01.tif'
        results, fig = analyze_microscopy_frequencies(image_path)
        plt.show()

    except Exception as e:
        print(f"Error analyzing image: {str(e)}")