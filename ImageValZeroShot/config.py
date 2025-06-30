"""
Configuration file for Arabic Image Captioning
"""

# Model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 128

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

# Caption generation prompt template
CAPTION_PROMPT = (
    "You are an expert in visual scene understanding and multilingual caption generation."
    "Analyze the content of this image, which is potentially related to the palestnian Nakba"
    "and Israeli occupation of Palestine, and provide a concise and meaningful caption in Arabic - about 15 to 50 words."
    "The caption should reflect the scene's content, emotional context, and should be natural and culturally appropriate."
    " Do not include any English or metadata — The caption must be in Arabic."
)

# CSV output configuration
CSV_HEADERS = ["image_file", "arabic_caption"]