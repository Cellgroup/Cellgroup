from PIL import Image
import os
import sys
import shutil

def compress_tiff(input_path, output_path, delete_original=False):
    """
    Compress a TIFF file using lossless compression and optionally delete original.
    
    Parameters:
    input_path (str): Path to input TIFF file
    output_path (str): Path to save compressed TIFF file
    delete_original (bool): Whether to delete the original file after compression
    
    Returns:
    tuple: Original size and compressed size in bytes
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Get original size
            original_size = os.path.getsize(input_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with optimal settings for TIFF compression
            img.save(
                output_path,
                'TIFF',
                compression='tiff_lzw',  # LZW compression (lossless)
                optimize=True
            )
            
            # Get compressed size
            compressed_size = os.path.getsize(output_path)
            
            # Verify the compressed file exists and is valid before deleting original
            if delete_original and os.path.exists(output_path):
                try:
                    # Verify the compressed file can be opened
                    with Image.open(output_path) as _:
                        os.remove(input_path)
                        print(f"Deleted original file: {input_path}")
                except Exception as e:
                    print(f"Warning: Could not verify compressed file, original kept: {str(e)}")
            
            return original_size, compressed_size
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None, None

def process_folder(input_folder, delete_originals=False):
    """
    Process all TIFF files in the input folder and save compressed versions.
    
    Parameters:
    input_folder (str): Path to input folder containing TIFF files
    delete_originals (bool): Whether to delete original files after compression
    """
    # Create output folder name
    output_folder = input_folder.rstrip('/') + '_compressed'
    
    # Get all TIFF files from input folder
    tiff_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.tiff', '.tif'))]
    
    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    total_original = 0
    total_compressed = 0
    processed_files = 0
    
    # Process each TIFF file
    for file in tiff_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        
        print(f"\nProcessing: {file}")
        original_size, compressed_size = compress_tiff(input_path, output_path, delete_originals)
        
        if original_size and compressed_size:
            print(f"Original size: {original_size/1024:.2f} KB")
            print(f"Compressed size: {compressed_size/1024:.2f} KB")
            print(f"Compression ratio: {original_size/compressed_size:.2f}x")
            
            total_original += original_size
            total_compressed += compressed_size
            processed_files += 1
    
    # Print summary
    if processed_files > 0:
        print(f"\nSummary:")
        print(f"Processed {processed_files} files")
        print(f"Total original size: {total_original/1024/1024:.2f} MB")
        print(f"Total compressed size: {total_compressed/1024/1024:.2f} MB")
        print(f"Overall compression ratio: {total_original/total_compressed:.2f}x")
        print(f"Compressed files saved to: {output_folder}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)
    
    # Add confirmation for deletion
    delete_originals = input("Delete original files after compression? (yes/no): ").lower().startswith('y')
    process_folder(input_folder, delete_originals)

if __name__ == "__main__":
    main()
