#!/usr/bin/env python3
"""
Script to evaluate trained models and generate captions
"""

import os
import sys
import argparse
from finetune_trainer import ArabicImageCaptionTrainer


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Arabic image captioning model")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing model and data")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to evaluate (e.g., checkpoint-50)")
    parser.add_argument("--test_images", type=str, help="Directory containing test images")
    parser.add_argument("--max_images", type=int, help="Maximum number of images to process")
    parser.add_argument("--list_checkpoints", action="store_true", help="List available checkpoints and exit")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ArabicImageCaptionTrainer(base_dir=args.base_dir)
    
    # List checkpoints if requested
    if args.list_checkpoints:
        checkpoints = trainer.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found")
            sys.exit(1)
        return
    
    # Determine checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        if os.path.isabs(args.checkpoint):
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(trainer.paths["output_dir"], args.checkpoint)
    
    # Determine test images directory
    test_images_dir = args.test_images or trainer.paths["test_images_dir"]
    
    # Run evaluation
    print("Starting model evaluation...")
    results = trainer.evaluate_model(
        checkpoint_path=checkpoint_path,
        test_images_dir=test_images_dir,
        max_images=args.max_images
    )
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Processed {len(results)} images")
        
        # Show some sample results
        print("\n=== Sample Results ===")
        for i, result in enumerate(results[:3]):
            print(f"\n{i+1}. {result['image_file']}")
            print(f"   Caption: {result['arabic_caption']}")
    else:
        print("❌ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()