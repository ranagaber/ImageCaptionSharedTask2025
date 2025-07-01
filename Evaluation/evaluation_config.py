"""
Configuration file for Arabic Caption Evaluation
"""

import re

# Default column names
DEFAULT_COLUMNS = {
    "reference": "Description",
    "candidate": "arabic_caption",
    "image_file": "image_file"
}

# Arabic text normalization patterns
ARABIC_PATTERNS = {
    # Arabic diacritics (Tashkeel) for removal
    "diacritics": re.compile(r'[\u064B-\u065F\u0670\u0640]'),
    
    # Arabic punctuation marks
    "arabic_punctuation": re.compile(r'[؟،؛٪٫٬‰؍]'),
    
    # General punctuation
    "general_punctuation": re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]'),
    
    # Multiple spaces
    "multiple_spaces": re.compile(r'\s+'),
    
    # Arabic letter normalization patterns
    "alef_patterns": re.compile(r'[أإآا]'),
    "yeh_patterns": re.compile(r'[يى]'),
    "teh_patterns": re.compile(r'[ةه]'),
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    "bleu_weights": {
        "bleu1": (1, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (0.33, 0.33, 0.33, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25)
    },
    "rouge_types": ['rouge1', 'rouge2', 'rougeL'],
    "similarity_metrics": [
        'char_cosine_similarity',
        'word_cosine_similarity', 
        'jaccard_similarity',
        'lin_similarity',
        'semantic_similarity'
    ]
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    "char_ngram_range": (1, 4),
    "word_ngram_range": (1, 2),
    "lowercase": False,
    "strip_accents": None,
    "min_token_length": 2
}

# Output file configurations
OUTPUT_CONFIG = {
    "detailed_results": "detailed_arabic_caption_evaluation_results.csv",
    "metrics_only": "metrics_only_evaluation_results.csv",
    "summary_json": "enhanced_evaluation_summary.json",
    "comprehensive_excel": "comprehensive_evaluation_results.xlsx",
    "enhanced_csv": "detailed_evaluation_results.csv"
}

# Performance categories for BLEU-4 scores
PERFORMANCE_CATEGORIES = {
    "excellent": 0.5,  # > 0.5
    "good": 0.3,       # 0.3-0.5
    "fair": 0.1,       # 0.1-0.3
    "poor": 0.0        # <= 0.1
}

# LLM Judge configuration
LLM_JUDGE_CONFIG = {
    "system_prompt": """
You are an expert AI evaluator specializing in Arabic language and semantics. Your task is to act as an impartial judge and evaluate the quality of a "model-generated caption" of a given image by comparing it to a "ground truth caption" for the same image.
You will not see the image itself. Your entire evaluation must be based on the textual comparison of the two provided Arabic captions. Assume the "ground truth caption" is the accurate and correct description of the image.

Evaluation Criteria:
Please evaluate the "model-generated caption" based on the following criteria, using a scale of 1 to 10, where 1 is Very Poor and 10 is Excellent.

Semantic Similarity:
How closely does the model's caption convey the same core meaning as the ground truth?
Does the caption mention the same key objects, attributes, and actions as the ground truth?
Score 10: The meaning is identical or nearly identical.
Score 1: The meaning is completely different or irrelevant.

REPLY WITH THE SCORE ONLY. NO EXPLANATION

Caption to Evaluate:
""",
    "temperature": 0,
    "max_retries": 3,
    "structured_output": True,
    "model_id": "gpt-4o",
    "base_url": "https://api.openai.com/v1"
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "whitegrid",
    "color_palette": "Set2",
    "save_format": "png"
}

# Metrics display names for reports
METRICS_DISPLAY_NAMES = {
    'bleu1': 'BLEU-1',
    'bleu2': 'BLEU-2', 
    'bleu3': 'BLEU-3',
    'bleu4': 'BLEU-4',
    'rouge1': 'ROUGE-1',
    'rouge2': 'ROUGE-2',
    'rougeL': 'ROUGE-L',
    'char_cosine_similarity': 'Char Cosine Sim',
    'word_cosine_similarity': 'Word Cosine Sim',
    'jaccard_similarity': 'Jaccard Sim',
    'lin_similarity': 'Lin Similarity',
    'semantic_similarity': 'Semantic Sim'
}

# Progress reporting intervals
PROGRESS_CONFIG = {
    "report_interval": 50,  # Report progress every N items
    "verbose": True
}