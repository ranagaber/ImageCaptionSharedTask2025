"""
Main evaluation class and functions
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# Local imports
import evaluation_config as config
from arabic_text_processor import ArabicTextProcessor
from evaluation_metrics import EvaluationMetrics
from evaluation_visualizer import EvaluationVisualizer
from llm_judge import LLMJudge


class ArabicCaptionEvaluator:
    """
    Comprehensive evaluation suite for Arabic image captioning with enhanced preprocessing.
    """
    
    def __init__(self):
        """Initialize evaluator components."""
        self.text_processor = ArabicTextProcessor()
        self.metrics_calculator = EvaluationMetrics()
        self.visualizer = EvaluationVisualizer()
        self.llm_judge = None
    
    def setup_llm_judge(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                       model_id: str = "gpt-4"):
        """
        Setup LLM judge for evaluation.
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            model_id: Model identifier
        """
        self.llm_judge = LLMJudge(api_key, base_url, model_id)
    
    def evaluate_dataset(self, df: pd.DataFrame, 
                        ref_col: str = config.DEFAULT_COLUMNS["reference"],
                        cand_col: str = config.DEFAULT_COLUMNS["candidate"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate entire dataset and return both metrics and detailed results.
        
        Args:
            df: DataFrame with reference and candidate columns
            ref_col: Name of reference column
            cand_col: Name of candidate column
            
        Returns:
            Tuple of (metrics_df, detailed_results_df)
        """
        print(f"Evaluating {len(df)} caption pairs with enhanced preprocessing...")
        
        # Validate input data
        missing_ref = df[ref_col].isna().sum()
        missing_cand = df[cand_col].isna().sum()
        
        if missing_ref > 0 or missing_cand > 0:
            print(f"Warning: Found {missing_ref} missing references and {missing_cand} missing candidates")
            print("Removing rows with missing values...")
            df = df.dropna(subset=[ref_col, cand_col])
            print(f"Dataset size after cleaning: {len(df)}")
        
        results = []
        
        # Reset index to ensure sequential numbering
        df_reset = df.reset_index(drop=True)
        
        # Process with progress bar
        for idx, row in tqdm(df_reset.iterrows(), total=len(df_reset), desc="Evaluating captions"):
            reference = row[ref_col]
            candidate = row[cand_col]
            
            # Validate texts
            ref_validation = self.text_processor.validate_arabic_text(reference)
            cand_validation = self.text_processor.validate_arabic_text(candidate)
            
            if not ref_validation['is_valid'] or not cand_validation['is_valid']:
                # Still compute metrics but flag as invalid
                scores = {metric: 0.0 for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 
                                                   'rouge1', 'rouge2', 'rougeL',
                                                   'char_cosine_similarity', 'word_cosine_similarity',
                                                   'jaccard_similarity', 'lin_similarity', 'semantic_similarity']}
                scores['validation_error'] = f"Ref: {ref_validation.get('error', 'OK')}, Cand: {cand_validation.get('error', 'OK')}"
            else:
                scores = self.metrics_calculator.evaluate_single_pair(reference, candidate)
                scores['validation_error'] = None
            
            scores['original_index'] = idx
            results.append(scores)
            
            if (idx + 1) % config.PROGRESS_CONFIG["report_interval"] == 0:
                print(f"Processed {idx + 1}/{len(df_reset)} pairs...")
        
        # Create metrics DataFrame
        results_df = pd.DataFrame(results)
        
        # Create detailed results DataFrame by merging original data with metrics
        detailed_results_df = df_reset.copy()
        
        # Add all evaluation metrics to the detailed DataFrame
        for metric in results_df.columns:
            if metric != 'original_index':
                detailed_results_df[metric] = results_df[metric].values
        
        # Add text statistics
        detailed_results_df['reference_length'] = detailed_results_df[ref_col].apply(
            lambda x: len(self.text_processor.advanced_tokenize_arabic(x)) if pd.notna(x) else 0
        )
        detailed_results_df['candidate_length'] = detailed_results_df[cand_col].apply(
            lambda x: len(self.text_processor.advanced_tokenize_arabic(x)) if pd.notna(x) else 0
        )
        detailed_results_df['length_ratio'] = detailed_results_df.apply(
            lambda row: row['candidate_length'] / row['reference_length']
            if row['reference_length'] > 0 else 0, axis=1
        )
        
        # Add normalized texts for inspection
        detailed_results_df['reference_normalized'] = detailed_results_df[ref_col].apply(
            self.text_processor.normalize_arabic_text
        )
        detailed_results_df['candidate_normalized'] = detailed_results_df[cand_col].apply(
            self.text_processor.normalize_arabic_text
        )
        
        return results_df, detailed_results_df
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate comprehensive summary statistics.
        
        Args:
            results_df: DataFrame with evaluation metrics
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL',
                  'char_cosine_similarity', 'word_cosine_similarity', 'jaccard_similarity',
                  'lin_similarity', 'semantic_similarity']
        
        summary = {}
        for metric in metrics:
            if metric in results_df.columns:
                summary[f'{metric}_mean'] = float(round(results_df[metric].mean(), 4))
                summary[f'{metric}_std'] = float(round(results_df[metric].std(), 4))
                summary[f'{metric}_median'] = float(round(results_df[metric].median(), 4))
                summary[f'{metric}_min'] = float(round(results_df[metric].min(), 4))
                summary[f'{metric}_max'] = float(round(results_df[metric].max(), 4))
                summary[f'{metric}_q25'] = float(round(results_df[metric].quantile(0.25), 4))
                summary[f'{metric}_q75'] = float(round(results_df[metric].quantile(0.75), 4))
        
        return summary
    
    def analyze_performance_categories(self, results_df: pd.DataFrame, 
                                     metric: str = 'bleu4') -> Dict[str, int]:
        """
        Analyze performance categories based on metric thresholds.
        
        Args:
            results_df: DataFrame with evaluation metrics
            metric: Metric to use for categorization
            
        Returns:
            Dictionary with category counts
        """
        if metric not in results_df.columns:
            return {}
        
        scores = results_df[metric]
        categories = config.PERFORMANCE_CATEGORIES
        
        category_counts = {
            'excellent': int(len(scores[scores > categories['excellent']])),
            'good': int(len(scores[(scores > categories['good']) & (scores <= categories['excellent'])])),
            'fair': int(len(scores[(scores > categories['fair']) & (scores <= categories['good'])])),
            'poor': int(len(scores[scores <= categories['fair']]))
        }
        
        return category_counts
    
    def get_best_worst_examples(self, detailed_results_df: pd.DataFrame,
                               metric: str = 'bleu4', n: int = 10,
                               ref_col: str = config.DEFAULT_COLUMNS["reference"],
                               cand_col: str = config.DEFAULT_COLUMNS["candidate"]) -> Dict[str, pd.DataFrame]:
        """
        Get best and worst performing examples.
        
        Args:
            detailed_results_df: DataFrame with detailed results
            metric: Metric to use for ranking
            n: Number of examples to return
            ref_col: Reference column name
            cand_col: Candidate column name
            
        Returns:
            Dictionary with best and worst examples
        """
        if metric not in detailed_results_df.columns:
            return {}
        
        # Get best examples
        best_examples = detailed_results_df.nlargest(n, metric)[[
            ref_col, cand_col, metric, 'reference_length', 'candidate_length'
        ]].copy()
        
        # Get worst examples
        worst_examples = detailed_results_df.nsmallest(n, metric)[[
            ref_col, cand_col, metric, 'reference_length', 'candidate_length'
        ]].copy()
        
        return {
            'best': best_examples,
            'worst': worst_examples
        }
    
    def save_results(self, results_df: pd.DataFrame, detailed_results_df: pd.DataFrame,
                    summary: Dict, output_dir: str = "./evaluation_output/") -> Dict[str, str]:
        """
        Save evaluation results to various formats.
        
        Args:
            results_df: Metrics DataFrame
            detailed_results_df: Detailed results DataFrame
            summary: Summary statistics
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save detailed results CSV
        detailed_path = os.path.join(output_dir, config.OUTPUT_CONFIG["detailed_results"])
        detailed_results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        saved_files['detailed_csv'] = detailed_path
        
        # Save metrics only CSV
        metrics_path = os.path.join(output_dir, config.OUTPUT_CONFIG["metrics_only"])
        results_df.to_csv(metrics_path, index=False)
        saved_files['metrics_csv'] = metrics_path
        
        # Save summary JSON
        summary_path = os.path.join(output_dir, config.OUTPUT_CONFIG["summary_json"])
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        saved_files['summary_json'] = summary_path
        
        # Save comprehensive Excel file
        try:
            excel_path = os.path.join(output_dir, config.OUTPUT_CONFIG["comprehensive_excel"])
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Detailed results
                detailed_results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Metrics only
                results_df.to_excel(writer, sheet_name='Metrics_Only', index=False)
                
                # Summary statistics
                summary_items = [(k.replace('_mean', ''), v) for k, v in summary.items() if k.endswith('_mean')]
                summary_df = pd.DataFrame(summary_items, columns=['Metric', 'Mean'])
                
                for stat in ['std', 'median', 'min', 'max', 'q25', 'q75']:
                    summary_df[stat.title()] = [summary.get(f"{row['Metric']}_{stat}", 0) for _, row in summary_df.iterrows()]
                
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            saved_files['excel'] = excel_path
            
        except ImportError:
            print("⚠️ openpyxl not available. Excel file not created.")
        
        return saved_files


def evaluate_arabic_captions(df: pd.DataFrame, 
                           ref_col: str = config.DEFAULT_COLUMNS["reference"],
                           cand_col: str = config.DEFAULT_COLUMNS["candidate"],
                           save_results: bool = True,
                           create_visualizations: bool = True,
                           output_dir: str = "./evaluation_output/",
                           llm_judge_config: Optional[Dict] = None) -> Dict:
    """
    Main function to evaluate Arabic captions with enhanced preprocessing.
    
    Args:
        df: DataFrame with reference and candidate columns
        ref_col: Name of reference (ground truth) column
        cand_col: Name of candidate (generated) column
        save_results: Whether to save results to files
        create_visualizations: Whether to create visualizations
        output_dir: Output directory for results
        llm_judge_config: Optional LLM judge configuration
        
    Returns:
        Dictionary with results dataframe, detailed results, and summary
    """
    print("🚀 Starting Enhanced Arabic Caption Evaluation...")
    print(f"Dataset size: {len(df)} samples")
    print(f"Reference column: {ref_col}")
    print(f"Candidate column: {cand_col}")
    
    # Initialize evaluator
    evaluator = ArabicCaptionEvaluator()
    
    # Setup LLM judge if requested
    if llm_judge_config:
        evaluator.setup_llm_judge(
            llm_judge_config['api_key'],
            llm_judge_config.get('base_url', 'https://api.openai.com/v1'),
            llm_judge_config.get('model_id', 'gpt-4')
        )
    
    # Run evaluation
    results_df, detailed_results_df = evaluator.evaluate_dataset(df, ref_col, cand_col)
    
    # Generate summary
    summary = evaluator.generate_summary_report(results_df)
    
    # Performance analysis
    performance_categories = evaluator.analyze_performance_categories(results_df)
    best_worst_examples = evaluator.get_best_worst_examples(detailed_results_df)
    
    # Print comprehensive summary
    print_evaluation_summary(summary, performance_categories, detailed_results_df, ref_col, cand_col)
    
    # Print best/worst examples
    print_best_worst_examples(best_worst_examples, ref_col, cand_col)
    
    # Run LLM judge evaluation if available
    if evaluator.llm_judge and llm_judge_config.get('run_evaluation', False):
        max_samples = llm_judge_config.get('max_samples', 50)
        print(f"\n🤖 Running LLM Judge Evaluation on {max_samples} samples...")
        detailed_results_df = evaluator.llm_judge.evaluate_dataset(
            detailed_results_df, ref_col, cand_col, max_samples
        )
    
    # Save results if requested
    saved_files = {}
    if save_results:
        saved_files = evaluator.save_results(results_df, detailed_results_df, summary, output_dir)
        print("\n💾 Results saved to:")
        for file_type, file_path in saved_files.items():
            print(f"  - {file_type}: {file_path}")
    
    # Create visualizations if requested
    if create_visualizations:
        print("\n📊 Creating visualizations...")
        plot_dir = os.path.join(output_dir, "plots")
        figures = evaluator.visualizer.create_comprehensive_report(
            results_df, detailed_results_df, plot_dir
        )
    
    return {
        'results_df': results_df,
        'detailed_results_df': detailed_results_df,
        'summary': summary,
        'performance_categories': performance_categories,
        'best_worst_examples': best_worst_examples,
        'evaluator': evaluator,
        'saved_files': saved_files if save_results else None,
        'figures': figures if create_visualizations else None
    }


def print_evaluation_summary(summary: Dict, performance_categories: Dict, 
                           detailed_results_df: pd.DataFrame, ref_col: str, cand_col: str):
    """Print comprehensive evaluation summary."""
    print("\n📊 ENHANCED EVALUATION SUMMARY")
    print("=" * 60)
    
    metrics_display = config.METRICS_DISPLAY_NAMES
    
    for metric_key, display_name in metrics_display.items():
        if f'{metric_key}_mean' in summary:
            mean_val = summary[f'{metric_key}_mean']
            std_val = summary[f'{metric_key}_std']
            max_val = summary[f'{metric_key}_max']
            median_val = summary[f'{metric_key}_median']
            print(f"{display_name:18}: {mean_val:.4f} (±{std_val:.4f}) [Median: {median_val:.4f}, Max: {max_val:.4f}]")
    
    # Print additional statistics
    print(f"\nDataset Statistics:")
    print(f"Average reference length: {detailed_results_df['reference_length'].mean():.2f} words")
    print(f"Average candidate length: {detailed_results_df['candidate_length'].mean():.2f} words")
    print(f"Average length ratio: {detailed_results_df['length_ratio'].mean():.2f}")
    
    # Print performance categories
    if performance_categories:
        print(f"\n🎯 PERFORMANCE CATEGORIES (BLEU-4):")
        total_samples = sum(performance_categories.values())
        for category, count in performance_categories.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{category.title():12}: {count:4d} captions ({percentage:5.1f}%)")


def print_best_worst_examples(best_worst_examples: Dict, ref_col: str, cand_col: str, n: int = 5):
    """Print best and worst performing examples."""
    if not best_worst_examples:
        return
    
    print(f"\n🏆 TOP {n} BEST CAPTIONS:")
    print("=" * 60)
    best_examples = best_worst_examples['best'].head(n)
    for rank, (_, row) in enumerate(best_examples.iterrows(), 1):
        bleu4_score = row.get('bleu4', 'N/A')
        print(f"\nRank {rank} (BLEU-4: {bleu4_score:.4f}):")
        print(f"Ground Truth: {row[ref_col]}")
        print(f"Generated:    {row[cand_col]}")
    
    print(f"\n💔 TOP {n} WORST CAPTIONS:")
    print("=" * 60)
    worst_examples = best_worst_examples['worst'].head(n)
    for rank, (_, row) in enumerate(worst_examples.iterrows(), 1):
        bleu4_score = row.get('bleu4', 'N/A')
        print(f"\nRank {rank} (BLEU-4: {bleu4_score:.4f}):")
        print(f"Ground Truth: {row[ref_col]}")
        print(f"Generated:    {row[cand_col]}")


def print_correlation_analysis(results_df: pd.DataFrame):
    """Print correlation analysis between metrics."""
    metrics = ['bleu4', 'rouge1', 'rougeL', 'char_cosine_similarity', 
              'word_cosine_similarity', 'jaccard_similarity', 'lin_similarity']
    
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if len(available_metrics) < 2:
        return
    
    print(f"\n🔗 METRIC CORRELATIONS:")
    correlation_matrix = results_df[available_metrics].corr()
    print(correlation_matrix.round(3))
    
    # Find highest correlations
    print(f"\nHighest correlations:")
    correlations = []
    for i in range(len(available_metrics)):
        for j in range(i+1, len(available_metrics)):
            metric1, metric2 = available_metrics[i], available_metrics[j]
            corr = correlation_matrix.loc[metric1, metric2]
            correlations.append((metric1, metric2, corr))
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for metric1, metric2, corr in correlations[:5]:
        print(f"  {metric1} ↔ {metric2}: {corr:.3f}")


def quick_evaluate(csv_path: str, ref_col: str = None, cand_col: str = None,
                  save_results: bool = True, create_plots: bool = True) -> Dict:
    """
    Quick evaluation function for command-line usage.
    
    Args:
        csv_path: Path to CSV file with evaluation data
        ref_col: Reference column name (auto-detect if None)
        cand_col: Candidate column name (auto-detect if None)
        save_results: Whether to save results
        create_plots: Whether to create visualizations
        
    Returns:
        Evaluation results dictionary
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns if not provided
    if ref_col is None:
        possible_ref_cols = ['Description', 'reference', 'ground_truth', 'caption']
        for col in possible_ref_cols:
            if col in df.columns:
                ref_col = col
                break
        if ref_col is None:
            raise ValueError(f"Could not auto-detect reference column. Available columns: {df.columns.tolist()}")
    
    if cand_col is None:
        possible_cand_cols = ['arabic_caption', 'generated_caption', 'candidate', 'prediction']
        for col in possible_cand_cols:
            if col in df.columns:
                cand_col = col
                break
        if cand_col is None:
            raise ValueError(f"Could not auto-detect candidate column. Available columns: {df.columns.tolist()}")
    
    print(f"Using columns: Reference='{ref_col}', Candidate='{cand_col}'")
    
    # Verify columns exist
    if ref_col not in df.columns:
        raise ValueError(f"Reference column '{ref_col}' not found in DataFrame")
    if cand_col not in df.columns:
        raise ValueError(f"Candidate column '{cand_col}' not found in DataFrame")
    
    # Run evaluation
    results = evaluate_arabic_captions(
        df=df,
        ref_col=ref_col,
        cand_col=cand_col,
        save_results=save_results,
        create_visualizations=create_plots
    )
    
    # Print additional analysis
    print_correlation_analysis(results['results_df'])
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Arabic image captions")
    parser.add_argument("csv_file", help="Path to CSV file with evaluation data")
    parser.add_argument("--ref_col", help="Reference column name")
    parser.add_argument("--cand_col", help="Candidate column name") 
    parser.add_argument("--output_dir", default="./evaluation_output/", help="Output directory")
    parser.add_argument("--no_save", action="store_true", help="Don't save results")
    parser.add_argument("--no_plots", action="store_true", help="Don't create plots")
    parser.add_argument("--llm_judge", action="store_true", help="Enable LLM judge evaluation")
    parser.add_argument("--api_key", help="OpenAI API key for LLM judge")
    parser.add_argument("--model_id", default="gpt-4", help="Model ID for LLM judge")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples for LLM judge")
    
    args = parser.parse_args()
    
    # Setup LLM judge config if requested
    llm_config = None
    if args.llm_judge:
        if not args.api_key:
            print("❌ API key required for LLM judge evaluation")
            exit(1)
        llm_config = {
            'api_key': args.api_key,
            'model_id': args.model_id,
            'max_samples': args.max_samples,
            'run_evaluation': True
        }
    
    # Run evaluation
    try:
        results = evaluate_arabic_captions(
            df=pd.read_csv(args.csv_file),
            ref_col=args.ref_col or config.DEFAULT_COLUMNS["reference"],
            cand_col=args.cand_col or config.DEFAULT_COLUMNS["candidate"],
            save_results=not args.no_save,
            create_visualizations=not args.no_plots,
            output_dir=args.output_dir,
            llm_judge_config=llm_config
        )
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)