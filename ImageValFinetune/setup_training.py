#!/usr/bin/env python3
"""
Setup script for Arabic image captioning fine-tuning
This script handles the initial setup and data preparation
"""

import os
import sys
import argparse
from finetune_trainer import ArabicImageCaptionTrainer


def setup_for_colab(base_dir: str = "/content/drive/MyDrive/ImageVal"):
    """Setup specifically for Google Colab environment."""
    print("=== Google Colab Setup ===")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("⚠️ Not running in Colab, skipping drive mount")
    
    # Initialize trainer
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # Setup environment
    if not trainer.setup_environment():
        print("❌ Setup failed")
        return False
    
    # Prepare dataset with default paths
    excel_file = f"{base_dir}/Train/TrainSubtask2.xlsx"
    images_dir = f"{base_dir}/Train/images"
    
    print(f"\nExpected file locations:")
    print(f"Excel file: {excel_file}")
    print(f"Images directory: {images_dir}")
    print("\nMake sure these files exist before running training!")
    
    if os.path.exists(excel_file) and os.path.exists(images_dir):
        if trainer.prepare_dataset(excel_file=excel_file, images_dir=images_dir):
            print("✅ Dataset preparation complete")
            
            # Create both standard and conservative configs
            standard_config = trainer.create_training_config(conservative=False)
            conservative_config = trainer.create_training_config(conservative=True)
            
            print(f"\n🎉 Setup complete!")
            print(f"Standard config: {standard_config}")
            print(f"Conservative config: {conservative_config}")
            print(f"\nTo start training, run:")
            print(f"trainer.start_training('{standard_config}')")
            print(f"or")
            print(f"trainer.start_training('{conservative_config}')  # for limited VRAM")
            
            return True
        else:
            print("❌ Dataset preparation failed")
            return False
    else:
        print("⚠️ Required files not found. Please upload your data first.")
        return False


def setup_for_local(base_dir: str, excel_file: str, images_dir: str):
    """Setup for local environment."""
    print("=== Local Environment Setup ===")
    
    # Validate inputs
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Initialize trainer
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # Setup environment
    if not trainer.setup_environment():
        return False
    
    # Prepare dataset
    if not trainer.prepare_dataset(excel_file=excel_file, images_dir=images_dir):
        return False
    
    # Create configs
    standard_config = trainer.create_training_config(conservative=False)
    conservative_config = trainer.create_training_config(conservative=True)
    
    print(f"\n🎉 Setup complete!")
    print(f"Standard config: {standard_config}")
    print(f"Conservative config: {conservative_config}")
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Arabic image captioning fine-tuning")
    parser.add_argument("--colab", action="store_true", help="Setup for Google Colab")
    parser.add_argument("--base_dir", type=str, help="Base directory for data and outputs")
    parser.add_argument("--excel_file", type=str, help="Path to Excel training file")
    parser.add_argument("--images_dir", type=str, help="Directory containing training images")
    
    args = parser.parse_args()
    
    if args.colab:
        # Colab setup with default paths
        base_dir = args.base_dir or "/content/drive/MyDrive/ImageVal"
        success = setup_for_colab(base_dir)
    else:
        # Local setup with required arguments
        if not all([args.base_dir, args.excel_file, args.images_dir]):
            print("❌ For local setup, please provide: --base_dir, --excel_file, --images_dir")
            sys.exit(1)
        
        success = setup_for_local(args.base_dir, args.excel_file, args.images_dir)
    
    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()