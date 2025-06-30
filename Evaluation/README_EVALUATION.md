# Arabic Caption Evaluation Suite

A comprehensive evaluation toolkit for Arabic image captioning models with advanced text preprocessing, multiple metrics, and LLM-as-a-Judge capabilities.

## Features

- **Advanced Arabic Text Processing**: Diacritics removal, letter normalization, and intelligent tokenization
- **Comprehensive Metrics**: BLEU, ROUGE, cosine similarity, Jaccard, Lin similarity, and semantic similarity
- **LLM-as-a-Judge**: Optional evaluation using large language models (GPT-4, etc.)
- **Rich Visualizations**: Distribution plots, correlation matrices, performance analysis
- **Multiple Output Formats**: CSV, JSON, Excel with detailed results
- **Performance Analysis**: Best/worst examples, category analysis, length correlation

## Installation

```bash
pip install -r requirements_evaluation.txt
```

## Quick Start

### Basic Evaluation

```bash
# Simple evaluation with auto-detected columns
python run_evaluation.py results.csv

# Specify column names
python run_evaluation.py results.csv Description arabic_caption
```

### Python Usage

```python
from evaluation_main import evaluate_arabic_captions
import pandas as pd

# Load your data
df = pd.read_csv('your_results.csv')

# Run evaluation
results = evaluate_arabic_captions(
    df=df,
    ref_col='Description',        # Ground truth column
    cand_col='arabic_caption',    # Generated caption column
    save_results=True,
    create_visualizations=True
)

# Access results
metrics_df = results['results_df']
summary = results['summary']
print(f"Average BLEU-4: {summary['bleu4_mean']:.4f}")
```

## Advanced Usage

### With LLM Judge

```python
from evaluation_main import evaluate_arabic_captions

# Configure LLM judge
llm_config = {
    'api_key': 'your-openai-api-key',
    'base_url': 'https://api.openai.com/v1',  # or OpenRouter, etc.
    'model_id': 'gpt-4o',
    'max_samples': 100,
    'run_evaluation': True
}

results = evaluate_arabic_captions(
    df=df,
    ref_col='Description',
    cand_col='arabic_caption',
    llm_judge_config=llm_config
)
```

### Command Line with All Options

```bash
python evaluation_main.py results.csv \
    --ref_col Description \
    --cand_col arabic_caption \
    --output_dir ./my_evaluation/ \
    --llm_judge \
    --api_key your-api-key \
    --model_id gpt-4o \
    --max_samples 50
```

### Using Individual Components

```python
from arabic_text_processor import ArabicTextProcessor
from evaluation_metrics import EvaluationMetrics
from llm_judge import LLMJudge

# Text processing
processor = ArabicTextProcessor()
normalized_text = processor.normalize_arabic_text("النص العربي")
tokens = processor.advanced_tokenize_arabic("النص العربي")

# Metrics calculation
calculator = EvaluationMetrics()
scores = calculator.evaluate_single_pair("المرجع", "المُولد")

# LLM judge
judge = LLMJudge(api_key="your-key", base_url="url", model_id="gpt-4")
score = judge.judge_single_pair("المرجع", "المُولد")
```

## Data Format

Your CSV file should contain at least two columns:

| Description (Reference) | arabic_caption (Generated) |
|------------------------|----------------------------|
| صورة تاريخية تظهر مدينة القدس | صورة للمدينة القديمة |
| مشهد من الحياة اليومية | الناس في الشارع |

### Supported Column Names

The evaluation automatically detects common column names:

**Reference columns**: `Description`, `reference`, `ground_truth`, `caption`  
**Candidate columns**: `arabic_caption`, `generated_caption`, `candidate`, `prediction`

## Metrics Explained

### BLEU Scores
- **BLEU-1 to BLEU-4**: Precision-based metrics measuring n-gram overlap
- Higher values indicate better lexical similarity

### ROUGE Scores  
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

### Similarity Metrics
- **Character Cosine**: Character-level TF-IDF similarity
- **Word Cosine**: Word-level TF-IDF similarity  
- **Jaccard**: Set intersection over union
- **Lin Similarity**: Dice coefficient (2×intersection/total)
- **Semantic**: Weighted word overlap with frequency consideration

### LLM Judge
- **Score Range**: 1-10 scale
- **Criteria**: Semantic similarity, meaning preservation
- **Models**: Supports any OpenAI-compatible API

## Output Files

When `save_results=True`, the following files are created:

```
evaluation_output/
├── detailed_arabic_caption_evaluation_results.csv    # Full results + metrics
├── metrics_only_evaluation_results.csv               # Just the metrics
├── enhanced_evaluation_summary.json                  # Summary statistics
├── comprehensive_evaluation_results.xlsx             # Multi-sheet Excel
└── plots/                                            # Visualizations
    ├── metrics_distribution.png
    ├── correlation_matrix.png
    ├── performance_categories.png
    ├── metric_comparison.png
    └── length_analysis.png
```

## Arabic Text Processing

### Normalization Features
- **Diacritics Removal**: Removes Tashkeel marks (ً ٌ ٍ َ ُ ِ ّ ْ)
- **Letter Normalization**: 
  - أإآا → ا (Alef variants)
  - يى → ي (Yeh variants)  
  - ةه → ة (Teh Marbuta)
- **Punctuation Cleaning**: Removes Arabic and Latin punctuation
- **Tokenization**: Advanced Arabic-aware word splitting

### Example
```python
processor = ArabicTextProcessor()

original = "هٰذِهِ صُورَةٌ جَمِيلَةٌ لِلْمَدِينَةِ القَدِيمَةِ"
normalized = processor.normalize_arabic_text(original)
# Result: "هذه صورة جميلة للمدينة القديمة"

tokens = processor.advanced_tokenize_arabic(original)
# Result: ["هذه", "صورة", "جميلة", "للمدينة", "القديمة"]
```

## Performance Categories

Results are automatically categorized based on BLEU-4 scores:

- **Excellent** (>0.5): Very high quality matches
- **Good** (0.3-0.5): Good quality with minor differences  
- **Fair** (0.1-0.3): Moderate quality, some similarity
- **Poor** (≤0.1): Low quality, little similarity

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
```bash
pip install nltk rouge-score scikit-learn matplotlib seaborn
```

2. **NLTK Data**:
```python
import nltk
nltk.download('punkt')
```

3. **OpenAI API Issues**:
   - Verify API key is correct
   - Check rate limits
   - Ensure model is available

4. **Memory Issues**:
   - Process data in smaller batches
   - Disable visualizations for large datasets
   - Use `max_samples` parameter

### Validation Errors

The system validates Arabic text and reports issues:
- **No Arabic characters**: Text contains no Arabic script
- **No meaningful content**: Text is empty after normalization
- **Insufficient tokens**: Text too short for meaningful evaluation

## Configuration

Modify `evaluation_config.py` to customize:

```python
# Change performance thresholds
PERFORMANCE_CATEGORIES = {
    "excellent": 0.6,  # Stricter threshold
    "good": 0.4,
    "fair": 0.2,
    "poor": 0.0
}

# Adjust text processing
TFIDF_CONFIG = {
    "char_ngram_range": (1, 3),  # Shorter n-grams
    "min_token_length": 3        # Longer minimum tokens
}
```

## API Reference

### Main Functions

```python
evaluate_arabic_captions(df, ref_col, cand_col, save_results=True, 
                         create_visualizations=True, output_dir="./", 
                         llm_judge_config=None)
```

### Classes

- **`ArabicTextProcessor`**: Text normalization and tokenization
- **`EvaluationMetrics`**: Core metric calculations  
- **`LLMJudge`**: LLM-based evaluation
- **`EvaluationVisualizer`**: Plot generation
- **`ArabicCaptionEvaluator`**: Main evaluation orchestrator

## Examples

### Batch Processing

```python
import glob
import pandas as pd

# Process multiple files
results_files = glob.glob("results_*.csv")
all_results = []

for file in results_files:
    df = pd.read_csv(file)
    result = evaluate_arabic_captions(df, save_results=False)
    all_results.append({
        'file': file,
        'bleu4': result['summary']['bleu4_mean'],
        'rouge_l': result['summary']['rougeL_mean']
    })

comparison_df = pd.DataFrame(all_results)
print(comparison_df)
```

### Custom Metrics

```python
from evaluation_metrics import EvaluationMetrics

calculator = EvaluationMetrics()

# Evaluate single pair
scores = calculator.evaluate_single_pair(
    reference="صورة جميلة للمدينة القديمة",
    candidate="صورة للمدينة التاريخية"
)

print(f"BLEU-4: {scores['bleu4']}")
print(f"Semantic: {scores['semantic_similarity']}")
```

## Contributing

To extend the evaluation suite:

1. **Add new metrics**: Extend `EvaluationMetrics` class
2. **Custom preprocessing**: Modify `ArabicTextProcessor`
3. **New visualizations**: Add methods to `EvaluationVisualizer`
4. **Additional LLM providers**: Extend `LLMJudge` class

## License

This evaluation suite is designed for research and educational purposes. Please ensure compliance with API provider terms when using LLM judge features.