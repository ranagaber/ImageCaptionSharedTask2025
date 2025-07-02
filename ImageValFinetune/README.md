# Arabic Image Captioning Fine-tuning with Qwen2.5-VL

This project provides a complete pipeline for fine-tuning the Qwen2.5-VL-7B-Instruct model on Arabic image captions using LlamaFactory.

## Features

- Complete fine-tuning pipeline using LlamaFactory
- LoRA (Low-Rank Adaptation) for efficient training
- Support for both standard and conservative configurations
- Automated dataset preparation from Excel files
- Model evaluation and caption generation
- Google Colab and local environment support

## Requirements

### Hardware Requirements
- **Minimum**: 12GB VRAM (RTX 3080/4080, Tesla T4)
- **Recommended**: 16GB+ VRAM (RTX 4090, A100)
- 32GB+ system RAM
- ~50GB free disk space

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.0+
- Git

## Installation

1. Install dependencies:
```bash
pip install -r requirements_finetune.txt
```

2. The setup script will automatically install LlamaFactory

## Project Structure

```
arabic-image-captioning-finetune/
├── finetune_trainer.py       # Main trainer class
├── finetune_config.py        # Configuration settings
├── finetune_utils.py         # Utility functions
├── setup_training.py         # Setup script
├── run_training.py           # Simple training runner
├── evaluate_model.py         # Model evaluation script
├── requirements_finetune.txt # Dependencies
└── README_FINETUNE.md       # This file
```

## Quick Start

### Google Colab Setup

1. **Setup and prepare data:**
```python
# In Colab cell
!python setup_training.py --colab
```

2. **Start training:**
```python
!python run_training.py --colab
```

### Local Setup

1. **Prepare your data structure:**
```
your_base_dir/
├── Train/
│   ├── TrainSubtask2.xlsx    # Excel file with image names and Arabic descriptions
│   └── images/               # Training images
└── Test/
    └── images/               # Test images (optional)
```

2. **Setup:**
```bash
python setup_training.py \
    --base_dir /path/to/your/data \
    --excel_file /path/to/TrainSubtask2.xlsx \
    --images_dir /path/to/images
```

3. **Start training:**
```bash
python run_training.py --base_dir /path/to/your/data
```

## Configuration Options

### Standard vs Conservative Settings

**Standard Configuration** (for 16GB+ VRAM):
- LoRA rank: 8
- Batch size: 1
- Gradient accumulation: 16 steps

**Conservative Configuration** (for 12GB VRAM):
- LoRA rank: 4
- Batch size: 1
- Gradient accumulation: 32 steps
- Reduced workers

### Training Parameters

Key parameters you can adjust in `finetune_config.py`:

```python
TRAINING_CONFIG = {
    "lora_rank": 8,              # LoRA rank (4-16)
    "lora_alpha": 16,            # LoRA alpha
    "learning_rate": 2.0e-5,     # Learning rate
    "num_train_epochs": 15.0,    # Number of epochs
    "warmup_ratio": 0.1,         # Warmup ratio
    # ... more options
}
```

## Data Format

### Excel File Structure
Your `TrainSubtask2.xlsx` should have columns:
- `File Name`: Image filename (without extension)
- `Description`: Arabic caption for the image

### Example:
| File Name | Description |
|-----------|-------------|
| IMG001 | صورة تاريخية تظهر مدينة القدس القديمة |
| IMG002 | مشهد من الحياة اليومية في فلسطين |

## Training Process

1. **Environment Setup**: Installs LlamaFactory and dependencies
2. **Dataset Preparation**: Converts Excel data to LlamaFactory format
3. **Dataset Registration**: Registers dataset in LlamaFactory
4. **Configuration Creation**: Generates YAML training config
5. **Training**: Runs LoRA fine-tuning
6. **Evaluation**: Tests model on validation/test images

## Monitoring Training

Training outputs are saved to:
- Model checkpoints: `{output_dir}/checkpoint-{step}/`
- Training logs: Console output with loss curves
- Configuration: `{base_dir}/qwen_arabic_*.yaml`

## Evaluation and Inference

### Evaluate Trained Model

```bash
# Evaluate latest checkpoint
python evaluate_model.py --base_dir /path/to/data

# Evaluate specific checkpoint
python evaluate_model.py \
    --base_dir /path/to/data \
    --checkpoint checkpoint-50 \
    --max_images 100

# List available checkpoints
python evaluate_model.py \
    --base_dir /path/to/data \
    --list_checkpoints
```

### Results

Evaluation generates:
- `generated_arabic_captions.json`: Detailed results
- `fine_tune_generated_arabic_captions.csv`: CSV format results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use conservative configuration: `--conservative`
   - Reduce batch size in config
   - Enable gradient checkpointing

2. **Dataset Loading Errors**:
   - Verify image paths are correct
   - Check Excel file format
   - Ensure images are not corrupted

3. **LlamaFactory Installation Issues**:
   - Install from source: `pip install -e ".[torch,metrics]"`
   - Check PyTorch compatibility

### Memory Optimization

For limited VRAM:
```python
# Use these settings in custom_config
custom_config = {
    "lora_rank": 4,
    "gradient_accumulation_steps": 64,
    "dataloader_num_workers": 0,
    "preprocessing_num_workers": 1
}
```

## Performance Tips

1. **Use FP16**: Enabled by default, reduces memory usage
2. **Gradient Checkpointing**: Trades compute for memory
3. **LoRA Settings**: Lower rank = less memory, potentially less quality
4. **Batch Size**: Increase gradient accumulation instead of batch size

## Model Output

The fine-tuned model will generate Arabic captions in the style of your training data. Example output:

```
Input: Image of historical building
Output: صورة تاريخية تظهر مبنى قديم في القدس
```


## License

This project uses the Qwen2.5-VL model and LlamaFactory. Please refer to their respective licenses for usage terms.