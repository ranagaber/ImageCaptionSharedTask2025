#!/usr/bin/env python3
"""
Simple script to run Arabic caption evaluation
"""

import sys
import os
import pandas as pd
from evaluation_main import evaluate_arabic_captions, quick_evaluate


def main():
    """Main function for simple evaluation."""
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation.py <csv_file> [reference_column] [candidate_column]")
        print("Example: python run_evaluation.py results.csv Description arabic_caption")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    ref_col = sys.argv[2] if len(sys.argv) > 2 else None
    cand_col = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    print("🚀 Starting Arabic Caption Evaluation...")
    
    try:
        results = quick_evaluate(
            csv_path=csv_file,
            ref_col=ref_col,
            cand_col=cand_col,
            save_results=True,
            create_plots=True
        )
        
        print("\n✅ Evaluation completed successfully!")
        print(f"Results saved to: ./evaluation_output/")
        
        # Print quick summary
        summary = results['summary']
        print(f"\nQuick Summary:")
        print(f"BLEU-4: {summary.get('bleu4_mean', 0):.4f}")
        print(f"ROUGE-L: {summary.get('rougeL_mean', 0):.4f}")
        print(f"Semantic Similarity: {summary.get('semantic_similarity_mean', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()