"""
Utility functions for image captioning
"""

import os
import csv
from typing import List, Tuple
from PIL import Image
import config


def validate_image_folder(folder_path: str) -> bool:
    """
    Validate if the image folder exists and contains supported image files.
    
    Args:
        folder_path (str): Path to the image folder
        
    Returns:
        bool: True if folder exists and contains images, False otherwise
    """
    if not os.path.exists(folder_path):
        print(f"Error: Image folder '{folder_path}' does not exist.")
        return False
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return False
    
    image_files = get_image_files(folder_path)
    if not image_files:
        print(f"Error: No supported image files found in '{folder_path}'.")
        print(f"Supported formats: {config.SUPPORTED_IMAGE_FORMATS}")
        return False
    
    return True


def get_image_files(folder_path: str) -> List[str]:
    """
    Get list of supported image files from a folder.
    
    Args:
        folder_path (str): Path to the image folder
        
    Returns:
        List[str]: List of image filenames
    """
    try:
        files = os.listdir(folder_path)
        image_files = [
            f for f in files 
            if f.lower().endswith(config.SUPPORTED_IMAGE_FORMATS)
        ]
        return sorted(image_files)
    except Exception as e:
        print(f"Error reading folder '{folder_path}': {e}")
        return []


def validate_image_file(image_path: str) -> bool:
    """
    Validate if an image file can be opened and processed.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Invalid image file '{image_path}': {e}")
        return False


def create_output_directory(output_path: str) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path (str): Path to the output file
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False


def save_results_to_csv(results: List[Tuple[str, str]], output_csv: str) -> bool:
    """
    Save results to CSV file.
    
    Args:
        results (List[Tuple[str, str]]): List of (filename, caption) tuples
        output_csv (str): Path to output CSV file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(config.CSV_HEADERS)
            writer.writerows(results)
        return True
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return False


def format_caption(caption: str) -> str:
    """
    Clean and format the generated caption.
    
    Args:
        caption (str): Raw caption text
        
    Returns:
        str: Cleaned caption
    """
    if not caption:
        return ""
    
    # Remove extra whitespace and newlines
    caption = caption.strip()
    caption = ' '.join(caption.split())
    
    return caption


def print_progress_summary(processed: int, total: int, successful: int):
    """
    Print processing progress summary.
    
    Args:
        processed (int): Number of files processed
        total (int): Total number of files
        successful (int): Number of successfully processed files
    """
    print(f"\nProcessing Summary:")
    print(f"Total files: {total}")
    print(f"Processed: {processed}")
    print(f"Successful: {successful}")
    print(f"Failed: {processed - successful}")
    if total > 0:
        print(f"Success rate: {(successful/total)*100:.1f}%")