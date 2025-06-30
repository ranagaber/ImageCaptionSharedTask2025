# Arabic Image Captioning with Qwen2.5-VL

This project generates Arabic captions for images using the Qwen2.5-VL-7B-Instruct model, specifically designed for historical content related to Palestinian Nakba and Israeli occupation.

## Features

- Generates concise Arabic captions (15-50 words) for images
- Supports multiple image formats (PNG, JPG, JPEG, BMP, TIFF, WEBP)
- Batch processing of entire image folders
- CSV output with image filenames and corresponding captions
- Progress tracking and error handling
- Modular code structure for easy customization

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- ~15GB disk space for model weights

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Process a folder of images with the simple interface:

```bash
python run_captioning.py /path/to/images /path/to/output.csv
```

### Advanced Usage

Use the main script with full options:

```bash
python image_captioning.py \
    --image_folder /path/to/images \
    --output_csv /path/to/output.csv \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --max_tokens 128
```

### Command Line Arguments

- `--image_folder`: Path to folder containing images (required)
- `--output_csv`: Path to output CSV file (required)
- `--model_name`: Model name to use (default: Qwen/Qwen2.5-VL-7B-Instruct)
- `--max_tokens`: Maximum number of tokens to generate (default: 128)

## Examples

### Basic Usage
```bash
# Process images in ./test_images and save to results.csv
python run_captioning.py ./test_images ./results.csv
```

### With Custom Parameters
```bash
python image_captioning.py \
    --image_folder ./historical_photos \
    --output_csv ./captions/historical_captions.csv \
    --max_tokens 100
```

### Google Colab/Drive Usage
```bash
python image_captioning.py \
    --image_folder "/content/drive/MyDrive/ImageVal/Test/images" \
    --output_csv "/content/drive/MyDrive/ImageVal/Test/captions.csv"
```

## Output Format

The script generates a CSV file with two columns:
- `image_file`: Filename of the processed image
- `arabic_caption`: Generated Arabic caption

Example output:
```csv
image_file,arabic_caption
ISH.PH01.12.004.jpg,"صورة تاريخية تظهر جنودا يمارسون التدريبات العسكرية في ظل الظروف الصعبة"
ISH.PH01.12.010.jpg,"صورة تاريخية تظهر جماعة من الناس يحملون أسلحة في ميدان"
```

## File Structure

```
arabic-image-captioning/
├── image_captioning.py    # Main captioning class and CLI
├── run_captioning.py      # Simplified runner script
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Customization

### Modify Caption Prompt

Edit the prompt in `config.py`:

```python
CAPTION_PROMPT = (
    "Your custom prompt here..."
)
```

### Change Supported Formats

Modify `SUPPORTED_IMAGE_FORMATS` in `config.py`:

```python
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.your_format')
```

### Use Different Model

```bash
python image_captioning.py \
    --model_name "your/custom-model" \
    --image_folder ./images \
    --output_csv ./output.csv
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use a smaller model
2. **Module Import Error**: Ensure all dependencies are installed correctly
3. **Image Loading Error**: Check that image files are not corrupted

### Performance Tips

- Use GPU acceleration for faster processing
- Process images in smaller batches for memory efficiency
- Ensure sufficient disk space for model downloads

## Hardware Requirements

### Minimum
- 8GB RAM
- 4GB GPU memory
- CPU processing (slower)

### Recommended
- 16GB+ RAM
- 8GB+ GPU memory (RTX 3080/4080, A100, etc.)
- SSD storage for faster I/O

## License

This project uses the Qwen2.5-VL model which has its own license terms. Please refer to the official Qwen documentation for licensing details.