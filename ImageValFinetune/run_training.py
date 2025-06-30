#!/usr/bin/env python3
"""
Simple script to run training with pre-configured settings
"""

import sys
import os
from finetune_trainer import ArabicImageCaptionTrainer


def run_training_colab():
    """Run training in Google Colab with default settings."""
    base_dir = "/content/drive/MyDrive/ImageVal"
    
    # Mount drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass
    
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # Check if configs exist
    standard_config = f"{base_dir}/qwen_arabic_standard.yaml"
    conservative_config = f"{base_dir}/qwen_arabic_conservative.yaml"
    
    if os.path.exists(conservative_config):
        print("Using conservative configuration (recommended for Colab)")
        config_path = conservative_config
    elif os.path.exists(standard_config):
        print("Using standard configuration")
        config_path = standard_config
    else:
        print("❌ No configuration files found. Run setup_training.py first!")
        return False
    
    # Start training
    print(f"Starting training with config: {config_path}")
    success = trainer.start_training(config_path)
    
    if success:
        print("\n✅ Training completed! Running evaluation...")
        trainer.evaluate_model(max_images=50)  # Evaluate on first 50 test images
    
    return success


def run_training_local(base_dir: str, conservative: bool = False):
    """Run training in local environment."""
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # Determine config file
    config_suffix = "conservative" if conservative else "standard"
    config_path = f"{base_dir}/qwen_arabic_{config_suffix}.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Run setup_training.py first!")
        return False
    
    # Start training
    print(f"Starting training with config: {config_path}")
    success = trainer.start_training(config_path)
    
    if success:
        print("\n✅ Training completed! Running evaluation...")
        trainer.evaluate_model()
    
    return success


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Arabic image captioning training")
    parser.add_argument("--colab", action="store_true", help="Run in Google Colab mode")
    parser.add_argument("--base_dir", type=str, help="Base directory for local training")
    parser.add_argument("--conservative", action="store_true", help="Use conservative settings")
    
    args = parser.parse_args()
    
    if args.colab:
        success = run_training_colab()
    elif args.base_dir:
        success = run_training_local(args.base_dir, args.conservative)
    else:
        print("Please specify either --colab or --base_dir")
        sys.exit(1)
    
    if success:
        print("\n🎉 Training and evaluation completed successfully!")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()