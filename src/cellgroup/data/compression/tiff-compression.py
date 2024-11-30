from PIL import Image
import os

def compress_tiff(input_path, output_path):
    """
    Compress a TIFF file using lossless compression and optimize black areas.
    
    Parameters:
    input_path (str): Path to input TIFF file
    output_path (str): Path to save compressed TIFF file
    
    Returns:
    tuple: Original size and compressed size in bytes
    """
    # Open the image
    with Image.open(input_path) as img:
        # Convert to RGB if in RGBA mode
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Get original size
        original_size = os.path.getsize(input_path)
        
        # Save with optimal settings for TIFF compression
        img.save(
            output_path,
            'TIFF',
            compression='tiff_lzw',  # LZW compression (lossless)
            optimize=True
        )
        
        # Get compressed size
        compressed_size = os.path.getsize(output_path)
        
        return original_size, compressed_size

def main():
    # Example usage
    input_file = "EXP2111_A09_D#14_T0008_C13.tif"
    output_file = "compressed2.tif"
    
    original_size, compressed_size = compress_tiff(input_file, output_file)
    
    # Print results
    print(f"Original size: {original_size/1024:.2f} KB")
    print(f"Compressed size: {compressed_size/1024:.2f} KB")
    print(f"Compression ratio: {original_size/compressed_size:.2f}x")

if __name__ == "__main__":
    main()
