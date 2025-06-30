"""
Google Colab Notebook equivalent - Run these cells in sequence
This replaces the original Jupyter notebook with modular code
"""

# ============================================================================
# CELL 1: System Setup and Requirements Check
# ============================================================================

def cell_1_system_setup():
    """Check GPU and install basic requirements"""
    
    # Check GPU
    import subprocess
    import torch
    
    print("=== GPU Information ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("nvidia-smi not available")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("Not running in Colab")

# Run this cell first
# cell_1_system_setup()


# ============================================================================
# CELL 2: Install Dependencies and Setup
# ============================================================================

def cell_2_install_dependencies():
    """Install all required dependencies"""
    
    import subprocess
    import sys
    
    # Install requirements
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.41.0",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "Pillow>=8.0.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "datasets>=2.0.0"
    ]
    
    for req in requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", req])
    
    print("✅ Dependencies installed")

# Uncomment to run:
# cell_2_install_dependencies()


# ============================================================================
# CELL 3: Download and Setup Code
# ============================================================================

def cell_3_setup_code():
    """Download the modular code files"""
    
    import os
    
    # Create code files (in real usage, you'd upload these files)
    print("Setting up modular code structure...")
    
    # In Colab, you would upload the .py files or clone from a repository
    # For this example, we'll assume the files are available
    
    files_needed = [
        'finetune_config.py',
        'finetune_utils.py', 
        'finetune_trainer.py',
        'setup_training.py'
    ]
    
    for file in files_needed:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing - please upload this file")
    
    print("Code setup complete!")

# cell_3_setup_code()


# ============================================================================
# CELL 4: Setup Training Environment
# ============================================================================

def cell_4_setup_training():
    """Setup the training environment"""
    
    # Import our modules
    from setup_training import setup_for_colab
    
    # Setup for Colab with default paths
    base_dir = "/content/drive/MyDrive/ImageVal"
    
    print("Setting up training environment...")
    success = setup_for_colab(base_dir)
    
    if success:
        print("\n🎉 Training setup complete!")
        print("\nNext steps:")
        print("1. Make sure your TrainSubtask2.xlsx is in /content/drive/MyDrive/ImageVal/Train/")
        print("2. Make sure your training images are in /content/drive/MyDrive/ImageVal/Train/images/")
        print("3. Run the training cell")
    else:
        print("❌ Setup failed. Check your data files.")
    
    return success

# Uncomment to run:
# success = cell_4_setup_training()


# ============================================================================
# CELL 5: Start Training
# ============================================================================

def cell_5_start_training():
    """Start the training process"""
    
    from run_training import run_training_colab
    
    print("Starting training process...")
    print("This may take several hours depending on your data size and GPU.")
    
    success = run_training_colab()
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("❌ Training failed. Check the logs above.")
    
    return success

# Uncomment to run training:
# training_success = cell_5_start_training()


# ============================================================================
# CELL 6: Evaluate Model (Alternative approach)
# ============================================================================

def cell_6_evaluate_model():
    """Evaluate the trained model"""
    
    from finetune_trainer import ArabicImageCaptionTrainer
    
    # Initialize trainer
    base_dir = "/content/drive/MyDrive/ImageVal"
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # List available checkpoints
    print("Available checkpoints:")
    checkpoints = trainer.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. Train the model first.")
        return
    
    # Evaluate the latest checkpoint
    print(f"Evaluating latest checkpoint: {checkpoints[-1]}")
    
    results = trainer.evaluate_model(
        max_images=50  # Evaluate first 50 test images
    )
    
    if results:
        print(f"\n✅ Evaluation completed!")
        print(f"Generated captions for {len(results)} images")
        
        # Show sample results
        print("\n=== Sample Results ===")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['image_file']}")
            print(f"   Caption: {result['arabic_caption']}")
    
    return results

# Uncomment to run evaluation:
# results = cell_6_evaluate_model()


# ============================================================================
# CELL 7: Manual Training Configuration (Advanced)
# ============================================================================

def cell_7_advanced_training():
    """Advanced training with custom configuration"""
    
    from finetune_trainer import ArabicImageCaptionTrainer
    
    # Initialize trainer
    base_dir = "/content/drive/MyDrive/ImageVal"
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    # Custom configuration for specific needs
    custom_config = {
        "learning_rate": 1e-5,        # Lower learning rate
        "num_train_epochs": 20,       # More epochs
        "lora_rank": 16,              # Higher rank (if you have memory)
        "save_steps": 50,             # Save more frequently
        "eval_steps": 20,             # Evaluate more frequently
    }
    
    print("Creating custom training configuration...")
    
    # Create custom config
    config_path = trainer.create_training_config(
        conservative=True,  # Use conservative base settings
        custom_config=custom_config
    )
    
    print(f"Custom config created: {config_path}")
    
    # Optionally start training with custom config
    start_training = input("Start training with custom config? (y/n): ")
    if start_training.lower() == 'y':
        success = trainer.start_training(config_path)
        if success:
            print("Custom training completed!")
            trainer.evaluate_model()
    
    return config_path

# Uncomment for advanced training:
# custom_config_path = cell_7_advanced_training()


# ============================================================================
# CELL 8: Inference on New Images
# ============================================================================

def cell_8_inference_on_new_images():
    """Run inference on new images"""
    
    import os
    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    
    # Load the fine-tuned model
    base_dir = "/content/drive/MyDrive/ImageVal"
    model_dir = f"{base_dir}/qwen2_5vl_arabic_model"
    
    # Find latest checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    def generate_caption_for_image(image_path):
        """Generate caption for a single image"""
        try:
            image = Image.open(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in Arabic."}
                    ]
                }
            ]
            
            # Process
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract caption
            if "assistant\n" in response:
                caption = response.split("assistant\n")[-1].strip()
            else:
                caption = response.split("Describe this image in Arabic.")[-1].strip()
            
            return caption
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Example usage
    test_image_path = f"{base_dir}/Test/images"
    if os.path.exists(test_image_path):
        test_images = [f for f in os.listdir(test_image_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            print(f"\nTesting on {min(3, len(test_images))} images:")
            for i, img_file in enumerate(test_images[:3]):
                img_path = os.path.join(test_image_path, img_file)
                caption = generate_caption_for_image(img_path)
                print(f"\n{i+1}. {img_file}")
                print(f"   Caption: {caption}")
    
    print("\n✅ Inference testing complete!")
    return generate_caption_for_image

# Uncomment to test inference:
# caption_generator = cell_8_inference_on_new_images()


# ============================================================================
# CELL 9: Save and Export Model
# ============================================================================

def cell_9_export_model():
    """Export the trained model for later use"""
    
    import os
    import shutil
    
    base_dir = "/content/drive/MyDrive/ImageVal"
    model_dir = f"{base_dir}/qwen2_5vl_arabic_model"
    
    # Find best checkpoint (you may want to choose based on validation loss)
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # For simplicity, use the latest checkpoint
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    # Create export directory
    export_dir = f"{base_dir}/final_model"
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    
    # Copy the checkpoint to export directory
    shutil.copytree(checkpoint_path, export_dir)
    
    # Create a simple inference script
    inference_script = f"""
# Simple inference script for the fine-tuned model
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "{export_dir}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def generate_arabic_caption(image_path):
    image = Image.open(image_path)
    
    messages = [{{
        "role": "user",
        "content": [
            {{"type": "image", "image": image}},
            {{"type": "text", "text": "Describe this image in Arabic."}}
        ]
    }}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    caption = response.split("assistant\\n")[-1].strip() if "assistant\\n" in response else response.split("Describe this image in Arabic.")[-1].strip()
    
    return caption

# Usage example:
# caption = generate_arabic_caption("path/to/your/image.jpg")
# print(caption)
"""
    
    with open(f"{export_dir}/inference.py", 'w') as f:
        f.write(inference_script)
    
    print(f"✅ Model exported to: {export_dir}")
    print(f"✅ Inference script created: {export_dir}/inference.py")
    print(f"\nTo use the model later:")
    print(f"1. Load from: {export_dir}")
    print(f"2. Use the inference.py script")
    
    return export_dir

# Uncomment to export model:
# export_path = cell_9_export_model()


# ============================================================================
# CELL 10: Complete Training Pipeline (All-in-One)
# ============================================================================

def cell_10_complete_pipeline():
    """Run the complete training pipeline from start to finish"""
    
    print("🚀 Starting Complete Training Pipeline")
    print("=" * 50)
    
    # Step 1: Setup
    print("\n📋 Step 1: Environment Setup")
    from setup_training import setup_for_colab
    base_dir = "/content/drive/MyDrive/ImageVal"
    
    if not setup_for_colab(base_dir):
        print("❌ Setup failed!")
        return False
    
    # Step 2: Training
    print("\n🎯 Step 2: Training")
    from run_training import run_training_colab
    
    if not run_training_colab():
        print("❌ Training failed!")
        return False
    
    # Step 3: Evaluation
    print("\n📊 Step 3: Evaluation")
    from finetune_trainer import ArabicImageCaptionTrainer
    trainer = ArabicImageCaptionTrainer(base_dir=base_dir)
    
    results = trainer.evaluate_model(max_images=20)
    if not results:
        print("❌ Evaluation failed!")
        return False
    
    # Step 4: Export
    print("\n💾 Step 4: Model Export")
    export_path = cell_9_export_model()
    
    print("\n🎉 Complete Pipeline Finished Successfully!")
    print(f"✅ Model trained and saved to: {export_path}")
    print(f"✅ Evaluation results: {len(results)} images processed")
    
    return True

# Uncomment to run complete pipeline:
# pipeline_success = cell_10_complete_pipeline()


# ============================================================================
# Usage Instructions for Colab
# ============================================================================

"""
INSTRUCTIONS FOR GOOGLE COLAB:

1. Upload all the .py files to your Colab session:
   - finetune_config.py
   - finetune_utils.py
   - finetune_trainer.py
   - setup_training.py
   - run_training.py

2. Run cells in sequence:
   
   # Basic setup
   cell_1_system_setup()
   cell_2_install_dependencies()
   cell_3_setup_code()
   
   # Quick training
   cell_4_setup_training()
   cell_5_start_training()
   cell_6_evaluate_model()
   
   # OR run everything at once
   cell_10_complete_pipeline()

3. Make sure your data is in the correct Google Drive location:
   /content/drive/MyDrive/ImageVal/Train/TrainSubtask2.xlsx
   /content/drive/MyDrive/ImageVal/Train/images/[your images]

4. For testing:
   /content/drive/MyDrive/ImageVal/Test/images/[test images]

5. Monitor GPU usage and adjust settings if you get OOM errors.
"""