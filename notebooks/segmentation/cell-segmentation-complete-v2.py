"""
Cell Nuclei Segmentation Pipeline
--------------------------------
A comprehensive solution for analyzing cell imaging data, combining H-Watershed
and StarDist segmentation methods for robust cell nuclei detection.

This pipeline is designed to handle challenging cases where:
- Light intensity varies significantly between samples
- Multiple fluorescent colors/dyes are present
- Cells undergo dramatic morphological changes

Key Features:
- Dual segmentation approach using H-Watershed and StarDist
- Sophisticated mask combination strategy
- Quality control metrics
- Visualization tools
- Comprehensive error handling and logging
- Parameter customization via JSON configuration

"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
from scipy import ndimage
from scipy.ndimage import peak_local_max
from skimage import io, morphology, filters, measure
from skimage.segmentation import watershed
import stardist
from stardist.models import StarDist2D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('segmentation.log')
    ]
)
logger = logging.getLogger(__name__)

class SegmentationError(Exception):
    """Custom exception for segmentation-related errors."""
    pass


"""
Modifications to support TIFF file processing in the cell segmentation pipeline.
"""


class CellSegmentation:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path],
                 file_pattern: str = "*.tif*",  # Now accepts a file pattern
                 h_watershed_params: Optional[Dict] = None,
                 stardist_params: Optional[Dict] = None,
                 filter_params: Optional[Dict] = None):
        """
        Initialize the segmentation pipeline with support for multiple file formats.

        Args:
            input_path: Directory containing input images
            output_path: Directory for saving results
            file_pattern: Glob pattern for finding image files (e.g., "*.tif" or "*.tiff")
            h_watershed_params: Parameters for H-Watershed segmentation
            stardist_params: Parameters for StarDist segmentation
            filter_params: Parameters for region filtering
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.file_pattern = file_pattern  # Store the file pattern

        # Rest of initialization remains the same...

    def process_directory(self) -> Dict:
        """
        Process all matching images in the input directory.

        Now supports multiple file extensions including TIFF variants.
        """
        self.processing_stats['start_time'] = datetime.now()

        try:
            # Look for both .tif and .tiff files (case insensitive)
            image_files = []
            for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                image_files.extend(self.input_path.glob(pattern))

            total_files = len(image_files)
            logger.info(f"Found {total_files} TIFF images to process")

            for image_file in image_files:
                if not (self.output_path / image_file.name).exists():
                    logger.info(f"Processing {image_file.name}")
                    try:
                        self.process_image(image_file.name)
                        self.processing_stats['processed_images'] += 1
                    except Exception as e:
                        self.processing_stats['failed_images'] += 1
                        logger.error(f"Failed to process {image_file.name}: {str(e)}")
                else:
                    logger.info(f"Skipping {image_file.name} - already processed")

        finally:
            self.processing_stats['end_time'] = datetime.now()

        return self.processing_stats

    def process_image(self, image_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image with improved TIFF handling.

        Args:
            image_name: Name of the image file

        Returns:
            Tuple of (final mask array, processing statistics)
        """
        image_output_dir = self.output_path / image_name
        image_output_dir.mkdir(exist_ok=True)

        stats = {
            'start_time': datetime.now(),
            'end_time': None,
            'success': False
        }

        try:
            # Read TIFF image with proper handling of multi-page TIFFs
            image_path = self.input_path / image_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = io.imread(str(image_path))

            # Handle multi-page TIFFs
            if image.ndim > 2:
                logger.info(f"Detected multi-page TIFF, using first channel/frame")
                image = image[0] if image.ndim == 3 else image[0, 0]

            # Normalize image to 16-bit range if needed
            if image.dtype == np.uint16:
                image = (image / 65535.0 * 255).astype(np.uint8)

            # Rest of the processing remains the same...

        except Exception as e:
            logger.error(f"Error processing {image_name}: {str(e)}")
            stats['error'] = str(e)
            raise

        finally:
            stats['end_time'] = datetime.now()
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity values using percentile-based scaling.

        Args:
            image: Input image array

        Returns:
            Normalized image array with values between 0 and 1
        """
        p_low = np.percentile(image, self.stardist_params['percentile_low'])
        p_high = np.percentile(image, self.stardist_params['percentile_high'])
        return np.clip((image - p_low) / (p_high - p_low), 0, 1)

    def run_h_watershed(self, image: np.ndarray) -> np.ndarray:
        """
        Perform H-Watershed segmentation with peak flooding.

        This implementation includes:
        - H-minima transform for marker selection
        - Peak flooding for handling intensity variations
        - Optional region splitting

        Args:
            image: Input image array

        Returns:
            Binary mask after H-Watershed segmentation

        Raises:
            SegmentationError: If segmentation fails
        """
        try:
            # Calculate h-minima transform for marker selection
            h_minima = morphology.h_minima(
                image,
                self.h_watershed_params['h_min']
            )

            # Apply threshold
            thresh = filters.threshold_otsu(image)
            binary = image > thresh * (self.h_watershed_params['threshold'] / 1000.0)

            # Apply peak flooding if enabled
            if self.h_watershed_params['peak_flooding'] < 100:
                flood_level = self.h_watershed_params['peak_flooding'] / 100.0
                local_maxi = peak_local_max(
                    image,
                    indices=False,
                    footprint=np.ones((3, 3)),
                    labels=binary
                )
                markers = measure.label(local_maxi)

                # Adjust threshold for each region based on local maxima
                for region in measure.regionprops(markers, intensity_image=image):
                    max_intensity = region.max_intensity
                    region_thresh = thresh + flood_level * (max_intensity - thresh)
                    mask = markers == region.label
                    binary[mask] = image[mask] > region_thresh

            # Handle splitting based on parameter
            if self.h_watershed_params['allow_splitting']:
                distance = ndimage.distance_transform_edt(binary)
                markers = measure.label(h_minima)
            else:
                distance = ndimage.distance_transform_edt(binary)
                markers = measure.label(
                    peak_local_max(distance, indices=False)
                )

            # Apply watershed
            mask = watershed(-distance, markers, mask=binary)

            return mask > 0

        except Exception as e:
            raise SegmentationError(f"H-Watershed segmentation failed: {str(e)}")

    def run_stardist(self, image: np.ndarray) -> np.ndarray:
        """
        Perform StarDist segmentation using a pre-trained model.

        Args:
            image: Input image array

        Returns:
            Binary mask after StarDist segmentation

        Raises:
            SegmentationError: If segmentation fails
        """
        try:
            # Normalize image
            img_norm = self.normalize_image(image)

            # Predict instances using StarDist
            labels, _ = self.model.predict_instances(
                img_norm,
                prob_thresh=self.stardist_params['prob_thresh'],
                nms_thresh=self.stardist_params['nms_thresh'],
                n_tiles=self.stardist_params['n_tiles'],
                exclude_border=self.stardist_params['exclude_border']
            )

            return labels > 0

        except Exception as e:
            raise SegmentationError(f"StarDist segmentation failed: {str(e)}")

    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply post-processing operations to improve segmentation quality.

        Operations include:
        - Morphological opening to remove small artifacts
        - Watershed to separate touching cells
        - Hole filling

        Args:
            mask: Input binary mask

        Returns:
            Processed binary mask
        """
        # Morphological opening
        mask = morphology.binary_opening(mask, morphology.disk(1))

        # Watershed to separate touching cells
        distance = ndimage.distance_transform_edt(mask)
        markers = measure.label(morphology.h_maxima(distance, 1))
        mask = watershed(-distance, markers, mask=mask)

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)

        return mask

    def combine_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Combine two segmentation masks using logical operations.

        The combination strategy uses:
        1. XOR to identify regions of disagreement
        2. OR to include all potential cell regions
        3. Post-processing to clean up the result

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            Combined binary mask
        """
        # XOR operation to find regions of disagreement
        xor_mask = np.logical_xor(mask1, mask2)
        xor_mask = self.post_process_mask(xor_mask)

        # OR operation to combine regions
        combined = np.logical_or(mask1, xor_mask)

        # Final post-processing
        combined = self.post_process_mask(combined)

        return combined

    def filter_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Filter segmented regions based on size and shape criteria.

        This method implements similar filtering to ImageJ's Analyze Particles:
        - Size filtering removes too small or large regions
        - Circularity filtering removes non-cell-like shapes
        - Matches ImageJ's circularity formula: 4π*(area/perimeter^2)

        Args:
            mask: Input labeled mask

        Returns:
            Filtered binary mask where only regions meeting criteria remain
        """
        props = measure.regionprops(mask)
        filtered_mask = np.zeros_like(mask)

        for prop in props:
            area = prop.area
            perimeter = prop.perimeter

            # Calculate circularity using ImageJ's formula
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # Apply size and circularity filters
            if (self.filter_params['size_min'] <= area <= self.filter_params['size_max'] and
                    self.filter_params['circularity_min'] <= circularity <= self.filter_params['circularity_max']):
                filtered_mask[mask == prop.label] = 1

        return filtered_mask

    def save_mask(self, mask: np.ndarray, output_path: Path, filename: str) -> None:
        """
        Save segmentation mask to file with proper formatting.

        Args:
            mask: Binary mask to save
            output_path: Directory to save the mask
            filename: Name of the output file

        Raises:
            IOError: If saving fails
        """
        try:
            io.imsave(
                str(output_path / filename),
                mask.astype(np.uint8) * 255,
                check_contrast=False
            )
        except Exception as e:
            logger.error(f"Failed to save mask {filename}: {str(e)}")
            raise IOError(f"Failed to save mask: {str(e)}")

    def process_image(self, image_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image through the complete segmentation pipeline.

        Processing steps:
        1. H-Watershed segmentation
        2. StarDist segmentation
        3. Mask combination
        4. Region filtering
        5. Result saving

        Args:
            image_name: Name of the image file

        Returns:
            Tuple of (final mask array, processing statistics)

        Raises:
            SegmentationError: If processing fails
        """
        # Create output directory for this image
        image_output_dir = self.output_path / image_name
        image_output_dir.mkdir(exist_ok=True)

        # Initialize statistics
        stats = {
            'start_time': datetime.now(),
            'end_time': None,
            'success': False
        }

        try:
            # Read image
            image_path = self.input_path / image_name
            image = io.imread(str(image_path))

            # Run H-Watershed
            logger.info(f"Running H-Watershed on {image_name}...")
            h_watershed_mask = self.run_h_watershed(image)
            self.save_mask(
                h_watershed_mask,
                image_output_dir,
                f"{image_name} - watershed (h={self.h_watershed_params['h_min']}, "
                f"T={self.h_watershed_params['threshold']}, %=100).tif"
            )

            # Run StarDist
            logger.info(f"Running StarDist on {image_name}...")
            stardist_mask = self.run_stardist(image)
            self.save_mask(
                stardist_mask,
                image_output_dir,
                f"{image_name}_stardist_mask.tif"
            )

            # Combine masks
            logger.info("Combining masks...")
            combined_mask = self.combine_masks(h_watershed_mask, stardist_mask)
            self.save_mask(
                combined_mask,
                image_output_dir,
                f"{image_name}XOR_Mask.tif"
            )

            # Filter regions
            logger.info("Filtering regions...")
            final_mask = self.filter_regions(measure.label(combined_mask))
            self.save_mask(
                final_mask,
                image_output_dir,
                f"{image_name}_final_mask.tif"
            )

            stats['success'] = True
            logger.info(f"Successfully processed {image_name}")

            return final_mask, stats

        except Exception as e:
            logger.error(f"Error processing {image_name}: {str(e)}")
            stats['error'] = str(e)
            raise

        finally:
            stats['end_time'] = datetime.now()

    def process_directory(self) -> Dict:
        """
        Process all images in the input directory.

        Returns:
            Dictionary containing processing statistics:
            - Number of processed images
            - Number of failed images
            - Total processing time
        """
        self.processing_stats['start_time'] = datetime.now()

        try:
            # Process each PNG image in the input directory
            for image_file in self.input_path.glob('*.png'):
                if not (self.output_path / image_file.name).exists():
                    logger.info(f"Processing {image_file.name}")
                    try:
                        self.process_image(image_file.name)
                        self.processing_stats['processed_images'] += 1
                    except Exception as e:
                        self.processing_stats['failed_images'] += 1
                        logger.error(f"Failed to process {image_file.name}: {str(e)}")
                else:
                    logger.info(f"Skipping {image_file.name} - already processed")

        finally:
            self.processing_stats['end_time'] = datetime.now()

        return self.processing_stats


class SegmentationQC:
    """
    Quality Control class for evaluating segmentation results.

    This class provides methods to:
    - Calculate quality metrics
    - Check segmentation quality against thresholds
    - Generate QC reports
    """

    def __init__(self):
        """Initialize the QC class."""
        self.metrics = {}

    def calculate_metrics(self, mask: np.ndarray) -> Dict:
        """
        Calculate comprehensive quality metrics for a segmentation mask.

        Args:
            mask: Binary segmentation mask

        Returns:
            Dictionary containing QC metrics including:
            - Total cell count
            - Average cell size
            - Size variability
            - Average circularity
            - Coverage percentage
        """
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)

        metrics = {
            'total_cells': len(props),
            'average_size': np.mean([p.area for p in props]) if props else 0,
            'size_std': np.std([p.area for p in props]) if props else 0,
            'average_circularity': np.mean([
                (4 * np.pi * p.area) / (p.perimeter ** 2)
                for p in props if p.perimeter > 0
            ]) if props else 0,
            'coverage': np.sum(mask) / mask.size
        }

        return metrics


class SegmentationVisualization:
    """
    Visualization tools for segmentation results.

    This class provides methods to create various visualizations that help assess
    segmentation quality and prepare results for presentation. It includes options
    for overlays, boundary highlighting, and comparative views.
    """

    @staticmethod
    def create_overlay(
            image: np.ndarray,
            mask: np.ndarray,
            alpha: float = 0.3,
            color: Tuple[float, float, float] = (1, 0, 0)
    ) -> np.ndarray:
        """
        Create a transparent overlay of the segmentation mask on the original image.

        This visualization helps assess segmentation accuracy by showing how well
        the detected regions align with the visible cells in the original image.

        Args:
            image: Original grayscale image
            mask: Binary segmentation mask
            alpha: Transparency level (0-1, where 1 is fully opaque)
            color: RGB color tuple for the overlay

        Returns:
            RGB image with colored segmentation overlay
        """
        # Normalize image to 0-1 range
        img_norm = (image - image.min()) / (image.max() - image.min())

        # Create RGB image
        rgb = np.stack([img_norm] * 3, axis=-1)

        # Create colored overlay
        overlay = np.zeros_like(rgb)
        overlay[mask > 0] = color

        # Combine image and overlay
        result = (1 - alpha) * rgb + alpha * overlay

        return np.clip(result, 0, 1)

    @staticmethod
    def create_boundary_overlay(
            image: np.ndarray,
            mask: np.ndarray,
            boundary_width: int = 2,
            boundary_color: Tuple[float, float, float] = (1, 0, 0)
    ) -> np.ndarray:
        """
        Create an overlay showing cell boundaries.

        This visualization emphasizes the detected cell boundaries, which is
        particularly useful for assessing cell separation and edge detection
        accuracy.

        Args:
            image: Original grayscale image
            mask: Binary segmentation mask
            boundary_width: Width of boundary lines in pixels
            boundary_color: RGB color tuple for boundaries

        Returns:
            RGB image with highlighted cell boundaries
        """
        # Extract boundaries using morphological operations
        boundaries = mask ^ morphology.binary_erosion(
            mask,
            morphology.disk(boundary_width)
        )

        return SegmentationVisualization.create_overlay(
            image,
            boundaries,
            alpha=0.8,
            color=boundary_color
        )

    @staticmethod
    def create_comparison_view(
            image: np.ndarray,
            h_watershed_mask: np.ndarray,
            stardist_mask: np.ndarray,
            final_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create a side-by-side comparison of different segmentation stages.

        This visualization helps understand how the segmentation evolves through
        the pipeline and how the different methods contribute to the final result.

        Args:
            image: Original grayscale image
            h_watershed_mask: Mask from H-Watershed
            stardist_mask: Mask from StarDist
            final_mask: Final combined and filtered mask

        Returns:
            Combined RGB image showing all stages
        """
        # Create overlays for each mask
        h_overlay = SegmentationVisualization.create_overlay(
            image, h_watershed_mask, color=(1, 0, 0)
        )
        s_overlay = SegmentationVisualization.create_overlay(
            image, stardist_mask, color=(0, 1, 0)
        )
        f_overlay = SegmentationVisualization.create_overlay(
            image, final_mask, color=(0, 0, 1)
        )

        # Normalize original image to RGB
        img_norm = (image - image.min()) / (image.max() - image.min())
        orig_rgb = np.stack([img_norm] * 3, axis=-1)

        # Combine into 2x2 grid
        top = np.hstack([orig_rgb, h_overlay])
        bottom = np.hstack([s_overlay, f_overlay])

        return np.vstack([top, bottom])

    @staticmethod
    def save_visualization(
            image: np.ndarray,
            output_path: Path,
            filename: str,
            dpi: int = 300
    ) -> None:
        """
        Save visualization output to file.

        Supports multiple output formats and ensures proper scaling and color
        reproduction.

        Args:
            image: RGB image array to save
            output_path: Directory to save the image
            filename: Output filename
            dpi: Dots per inch for output resolution
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(
                output_path / filename,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()
        except Exception as e:
            logger.error(f"Failed to save visualization {filename}: {str(e)}")
            raise


def setup_logging(log_file: str = 'segmentation.log') -> None:
    """
    Configure logging for the segmentation pipeline.

    Sets up both console and file logging with appropriate formatting and log
    levels. This helps track pipeline progress and diagnose issues.

    Args:
        log_file: Path to the log file
    """
    # Create log directory if needed
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Set up handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )

    # Add custom logging levels for pipeline stages
    logging.addLevelName(15, 'PIPELINE')
    logging.addLevelName(25, 'VALIDATION')


def parse_arguments():
    """
    Parse command-line arguments for the segmentation pipeline.

    Defines and processes all command-line options, providing help text and
    argument validation.

    Returns:
        Parsed argument namespace
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Cell Nuclei Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help='Path to input images directory'
    )
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='Path to output directory'
    )

    # Optional arguments
    parser.add_argument(
        '--params',
        type=str,
        help='Path to JSON parameter file'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='segmentation.log',
        help='Path to log file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations for results'
    )

    return parser.parse_args()


def load_parameters(params_file: Optional[str] = None) -> Dict:
    """
    Load and validate segmentation parameters from a JSON file.

    This function handles parameter loading with several important features:
    - Provides sensible defaults for all parameters
    - Validates parameter types and ranges
    - Merges custom parameters with defaults
    - Ensures backward compatibility

    Args:
        params_file: Path to JSON parameter file

    Returns:
        Dictionary containing validated parameters

    Raises:
        ValueError: If parameter validation fails
    """
    # Define default parameters with explanatory comments
    default_params = {
        'h_watershed_params': {
            'h_min': 700.0,  # Controls region merging sensitivity
            'threshold': 450.0,  # Base intensity threshold
            'peak_flooding': 98,  # Percentage for peak flooding
            'allow_splitting': True  # Allow splitting of merged regions
        },
        'stardist_params': {
            'prob_thresh': 0.6,  # Probability threshold for detection
            'nms_thresh': 0.8,  # Non-maximum suppression threshold
            'n_tiles': 5,  # Number of tiles for processing large images
            'exclude_border': 2,  # Pixels to exclude at image border
            'normalize': True,  # Apply intensity normalization
            'percentile_low': 1.0,  # Lower percentile for normalization
            'percentile_high': 99.8  # Upper percentile for normalization
        },
        'filter_params': {
            'size_min': 25,  # Minimum region size in pixels
            'size_max': 800,  # Maximum region size in pixels
            'circularity_min': 0.0,  # Minimum circularity (0-1)
            'circularity_max': 1.0  # Maximum circularity (0-1)
        },
        'visualization_params': {
            'generate_overlays': True,  # Create segmentation overlays
            'save_comparisons': True,  # Save comparison visualizations
            'dpi': 300,  # Output resolution for saved images
            'overlay_alpha': 0.3  # Transparency for overlays
        }
    }

    def validate_parameters(params: Dict) -> None:
        """Validate parameter values and types."""
        # Validate H-Watershed parameters
        hw_params = params.get('h_watershed_params', {})
        if not isinstance(hw_params.get('h_min', 0), (int, float)) or hw_params.get('h_min', 0) <= 0:
            raise ValueError("h_min must be a positive number")
        if not isinstance(hw_params.get('peak_flooding', 0), (int, float)) or not 0 <= hw_params.get('peak_flooding',
                                                                                                     0) <= 100:
            raise ValueError("peak_flooding must be between 0 and 100")

        # Validate StarDist parameters
        sd_params = params.get('stardist_params', {})
        if not 0 <= sd_params.get('prob_thresh', 0) <= 1:
            raise ValueError("prob_thresh must be between 0 and 1")
        if not 0 <= sd_params.get('nms_thresh', 0) <= 1:
            raise ValueError("nms_thresh must be between 0 and 1")

        # Validate filter parameters
        f_params = params.get('filter_params', {})
        if f_params.get('size_min', 0) >= f_params.get('size_max', float('inf')):
            raise ValueError("size_min must be less than size_max")
        if not 0 <= f_params.get('circularity_min', 0) <= 1:
            raise ValueError("circularity_min must be between 0 and 1")

    if params_file:
        try:
            with open(params_file, 'r') as f:
                custom_params = json.load(f)

                # Validate custom parameters
                validate_parameters(custom_params)

                # Update default parameters with custom ones
                for category, params in custom_params.items():
                    if category in default_params:
                        default_params[category].update(params)

                logger.info(f"Successfully loaded parameters from {params_file}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in parameter file: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to load parameters from {params_file}: {str(e)}")
            logger.warning("Using default parameters")

    # Validate final parameter set
    validate_parameters(default_params)

    return default_params


def save_results(
        output_path: Path,
        image_name: str,
        original_image: np.ndarray,
        final_mask: np.ndarray,
        metrics: Dict,
        params: Dict
) -> None:
    """
    Save all results and metadata for a processed image.

    This function creates a comprehensive output package including:
    - Segmentation masks
    - Quality control metrics
    - Processing parameters
    - Visualizations
    - Analysis report

    Args:
        output_path: Directory to save results
        image_name: Name of the processed image
        original_image: Original input image
        final_mask: Final segmentation mask
        metrics: Dictionary of QC metrics
        params: Processing parameters used
    """
    # Create results directory
    results_dir = output_path / image_name
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save segmentation mask
        io.imsave(
            str(results_dir / f"{image_name}_mask.tif"),
            final_mask.astype(np.uint8) * 255
        )

        # Save visualization if enabled
        if params['visualization_params']['generate_overlays']:
            viz = SegmentationVisualization()
            overlay = viz.create_overlay(
                original_image,
                final_mask,
                alpha=params['visualization_params']['overlay_alpha']
            )
            viz.save_visualization(
                overlay,
                results_dir,
                f"{image_name}_overlay.png",
                dpi=params['visualization_params']['dpi']
            )

        # Save metrics and parameters
        results = {
            'metrics': metrics,
            'parameters': params,
            'timestamp': datetime.now().isoformat()
        }

        with open(results_dir / f"{image_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Generate analysis report
        generate_report(results_dir, image_name, results)

    except Exception as e:
        logger.error(f"Failed to save results for {image_name}: {str(e)}")
        raise


def generate_report(output_dir: Path, image_name: str, results: Dict) -> None:
    """
    Generate a detailed analysis report in Markdown format.

    The report includes:
    - Processing parameters used
    - Quality metrics and their interpretation
    - Potential issues or warnings
    - Suggestions for parameter adjustments

    Args:
        output_dir: Directory to save the report
        image_name: Name of the processed image
        results: Dictionary containing metrics and parameters
    """
    report = f"""# Cell Segmentation Analysis Report

## Image Information
- File: {image_name}
- Processing Date: {results['timestamp']}

## Quality Metrics
- Total Cells Detected: {results['metrics']['total_cells']}
- Average Cell Size: {results['metrics']['average_size']:.2f} pixels
- Size Variation: {results['metrics']['size_std']:.2f}
- Average Circularity: {results['metrics']['average_circularity']:.3f}
- Image Coverage: {results['metrics']['coverage'] * 100:.1f}%

## Processing Parameters
### H-Watershed
- h_min: {results['parameters']['h_watershed_params']['h_min']}
- Threshold: {results['parameters']['h_watershed_params']['threshold']}
- Peak Flooding: {results['parameters']['h_watershed_params']['peak_flooding']}%

### StarDist
- Probability Threshold: {results['parameters']['stardist_params']['prob_thresh']}
- NMS Threshold: {results['parameters']['stardist_params']['nms_thresh']}
- Tiles: {results['parameters']['stardist_params']['n_tiles']}

### Filtering
- Size Range: {results['parameters']['filter_params']['size_min']}-{results['parameters']['filter_params']['size_max']} pixels
- Circularity Range: {results['parameters']['filter_params']['circularity_min']}-{results['parameters']['filter_params']['circularity_max']}

## Analysis Notes
"""
    # Add quality assessment notes
    if results['metrics']['total_cells'] < 10:
        report += "- Warning: Low cell count detected\n"
    if results['metrics']['size_std'] > 200:
        report += "- Warning: High variation in cell sizes\n"
    if results['metrics']['coverage'] < 0.05:
        report += "- Warning: Low image coverage, possible under-segmentation\n"

    # Save report
    with open(output_dir / f"{image_name}_report.md", 'w') as f:
        f.write(report)


def run_pipeline(args: argparse.Namespace) -> int:
    """
    Main pipeline execution function that orchestrates the entire segmentation process.

    This function coordinates all pipeline components, handling the flow from input
    to output while providing progress updates and error handling. It ensures all
    resources are properly managed and cleaned up.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Set up logging first thing
    setup_logging(args.log_file)
    logger.info("Starting cell segmentation pipeline")

    try:
        # Load and validate parameters
        logger.info("Loading parameters...")
        params = load_parameters(args.params)

        # Initialize pipeline components
        segmentation = CellSegmentation(
            input_path=args.input,
            output_path=args.output,
            h_watershed_params=params['h_watershed_params'],
            stardist_params=params['stardist_params'],
            filter_params=params['filter_params']
        )

        qc = SegmentationQC()

        # Process all images
        total_images = len(list(Path(args.input).glob('*.png')))
        logger.info(f"Found {total_images} images to process")

        results = []
        for i, image_file in enumerate(Path(args.input).glob('*.png'), 1):
            logger.info(f"Processing image {i}/{total_images}: {image_file.name}")

            try:
                # Process single image
                final_mask, stats = segmentation.process_image(image_file.name)

                # Run quality control
                passed_qc, metrics = qc.check_segmentation_quality(final_mask)

                # Load original image for visualization
                original_image = io.imread(str(image_file))

                # Save comprehensive results
                save_results(
                    Path(args.output),
                    image_file.name,
                    original_image,
                    final_mask,
                    metrics,
                    params
                )

                results.append({
                    'image': image_file.name,
                    'success': True,
                    'passed_qc': passed_qc,
                    'metrics': metrics
                })

            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {str(e)}")
                results.append({
                    'image': image_file.name,
                    'success': False,
                    'error': str(e)
                })

        # Generate summary report
        generate_summary_report(args.output, results)

        # Log final statistics
        successful = sum(1 for r in results if r['success'])
        passed_qc = sum(1 for r in results if r.get('passed_qc', False))

        logger.info("Pipeline completed:")
        logger.info(f"- Successfully processed: {successful}/{total_images}")
        logger.info(f"- Passed quality control: {passed_qc}/{successful}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


def generate_summary_report(output_path: str, results: List[Dict]) -> None:
    """
    Generate a comprehensive summary report of all processed images.

    Creates a detailed report including:
    - Overall success rates
    - Quality metrics distribution
    - Common failure patterns
    - Recommendations for improvement

    Args:
        output_path: Directory to save the report
        results: List of processing results for all images
    """
    successful_results = [r for r in results if r['success']]

    report = """# Cell Segmentation Pipeline Summary Report

## Processing Statistics
"""
    # Add overall statistics
    report += f"""
- Total images processed: {len(results)}
- Successfully processed: {len(successful_results)}
- Success rate: {len(successful_results) / len(results) * 100:.1f}%
"""

    if successful_results:
        # Calculate metric statistics
        metrics_stats = {
            'total_cells': [],
            'average_size': [],
            'coverage': []
        }

        for result in successful_results:
            metrics = result['metrics']
            metrics_stats['total_cells'].append(metrics['total_cells'])
            metrics_stats['average_size'].append(metrics['average_size'])
            metrics_stats['coverage'].append(metrics['coverage'])

        report += """
## Quality Metrics Summary
"""
        # Add metric statistics
        for metric, values in metrics_stats.items():
            report += f"""
### {metric.replace('_', ' ').title()}
- Mean: {np.mean(values):.2f}
- Median: {np.median(values):.2f}
- Standard deviation: {np.std(values):.2f}
- Range: {min(values):.2f} - {max(values):.2f}
"""

    # Add failure analysis if there were any failures
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        report += """
## Failure Analysis
"""
        error_counts = {}
        for result in failed_results:
            error = result['error']
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in error_counts.items():
            report += f"- {error}: {count} occurrences\n"

    # Save report
    report_path = Path(output_path) / 'pipeline_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)


def main():
    """
    Main entry point for the cell segmentation pipeline.

    This function:
    1. Parses command-line arguments
    2. Sets up the execution environment
    3. Runs the pipeline
    4. Handles exit codes
    """
    # Parse arguments
    args = parse_arguments()

    try:
        # Run pipeline and get exit code
        exit_code = run_pipeline(args)

        # Exit with appropriate code
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


# Example configurations showing common usage patterns
example_params = {
    # Configuration for bright, well-separated nuclei
    'bright_nuclei': {
        'h_watershed_params': {
            'h_min': 500.0,
            'threshold': 400.0,
            'peak_flooding': 95
        }
    },
    # Configuration for dim, clustered nuclei
    'dim_clustered': {
        'h_watershed_params': {
            'h_min': 800.0,
            'threshold': 300.0,
            'peak_flooding': 98
        },
        'stardist_params': {
            'prob_thresh': 0.5,
            'nms_thresh': 0.7
        }
    }
}

if __name__ == "__main__":
    main()

