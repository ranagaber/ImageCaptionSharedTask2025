#!/usr/bin/env python3
"""
Simplified script to run image captioning with minimal setup
"""

import sys
import os
from image_captioning import ArabicImageCaptioner


def main():
    """Simple interface for running image captioning."""
    
    if len(sys.argv) != 3:
        print("Usage: python run_captioning.py <image_folder> <output_csv>")
        print("Example: python run_captioning.py ./images ./results/captions.csv")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    output_csv = sys.argv[2]
    
    # Validate inputs
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        sys.exit(1)
    
    # Initialize and run captioner
    print("Initializing Arabic Image Captioner...")
    captioner = ArabicImageCaptioner()
    captioner.load_model()
    
    print(f"Processing images from: {image_folder}")
    print(f"Output will be saved to: {output_csv}")
    
    captioner.process_folder(image_folder, output_csv)
    print("Processing completed!")


if __name__ == "__main__":
    main()