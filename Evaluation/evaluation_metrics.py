"""
Core evaluation metrics for Arabic caption evaluation
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import warnings

# NLP libraries
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
import evaluation_config as config
from arabic_text_processor import ArabicTextProcessor

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class EvaluationMetrics:
    """Core evaluation metrics for Arabic captions."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.text_processor = ArabicTextProcessor()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            config.METRICS_CONFIG["rouge_types"], 
            use_stemmer=False
        )
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate BLEU scores with advanced tokenization.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Dictionary with BLEU scores
        """
        ref_tokens = self.text_processor.advanced_tokenize_arabic(reference)
        cand_tokens = self.text_processor.advanced_tokenize_arabic(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        # BLEU expects list of references
        reference_list = [ref_tokens]
        
        try:
            weights = config.METRICS_CONFIG["bleu_weights"]
            bleu_scores = {}
            
            for bleu_type, weight in weights.items():
                score = sentence_bleu(
                    reference_list, 
                    cand_tokens, 
                    weights=weight, 
                    smoothing_function=self.smoothing
                )
                bleu_scores[bleu_type] = round(score, 4)
                
        except Exception:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        return bleu_scores
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores with Arabic-specific cleaning.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Dictionary with ROUGE scores
        """
        ref_clean = self.text_processor.clean_for_rouge(reference)
        cand_clean = self.text_processor.clean_for_rouge(candidate)
        
        if not ref_clean or not cand_clean:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(ref_clean, cand_clean)
            return {
                'rouge1': round(scores['rouge1'].fmeasure, 4),
                'rouge2': round(scores['rouge2'].fmeasure, 4),
                'rougeL': round(scores['rougeL'].fmeasure, 4)
            }
        except Exception:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_cosine_similarity(self, reference: str, candidate: str, 
                                  analyzer: str = 'char') -> float:
        """
        Calculate cosine similarity with normalized text.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            analyzer: 'char' or 'word' for different tokenization
            
        Returns:
            Cosine similarity score
        """
        if analyzer == 'char':
            ref_clean = self.text_processor.normalize_arabic_text(reference)
            cand_clean = self.text_processor.normalize_arabic_text(candidate)
            ngram_range = config.TFIDF_CONFIG["char_ngram_range"]
        else:  # word
            ref_tokens = self.text_processor.advanced_tokenize_arabic(reference)
            cand_tokens = self.text_processor.advanced_tokenize_arabic(candidate)
            ref_clean = ' '.join(ref_tokens)
            cand_clean = ' '.join(cand_tokens)
            ngram_range = config.TFIDF_CONFIG["word_ngram_range"]
        
        if not ref_clean or not cand_clean:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                lowercase=config.TFIDF_CONFIG["lowercase"],
                strip_accents=config.TFIDF_CONFIG["strip_accents"]
            )
            vectors = vectorizer.fit_transform([ref_clean, cand_clean])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return round(similarity, 4)
        except Exception:
            return 0.0
    
    def calculate_jaccard_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculate Jaccard similarity coefficient.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Jaccard similarity score
        """
        ref_tokens = set(self.text_processor.advanced_tokenize_arabic(reference))
        cand_tokens = set(self.text_processor.advanced_tokenize_arabic(candidate))
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        intersection = len(ref_tokens.intersection(cand_tokens))
        union = len(ref_tokens.union(cand_tokens))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        return round(jaccard_sim, 4)
    
    def calculate_lin_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculate Lin similarity (Dice coefficient).
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Lin similarity score
        """
        ref_tokens = set(self.text_processor.advanced_tokenize_arabic(reference))
        cand_tokens = set(self.text_processor.advanced_tokenize_arabic(candidate))
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        intersection = len(ref_tokens.intersection(cand_tokens))
        total = len(ref_tokens) + len(cand_tokens)
        
        if total == 0:
            return 0.0
        
        lin_sim = (2.0 * intersection) / total
        return round(lin_sim, 4)
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """
        Enhanced semantic similarity using weighted word overlap.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Semantic similarity score
        """
        ref_tokens = self.text_processor.advanced_tokenize_arabic(reference)
        cand_tokens = self.text_processor.advanced_tokenize_arabic(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Count word frequencies
        ref_counter = Counter(ref_tokens)
        cand_counter = Counter(cand_tokens)
        
        # Calculate weighted overlap
        common_words = set(ref_tokens).intersection(set(cand_tokens))
        
        if not common_words:
            return 0.0
        
        # Calculate precision and recall with frequency weighting
        overlap_score = 0
        total_ref_freq = sum(ref_counter.values())
        total_cand_freq = sum(cand_counter.values())
        
        for word in common_words:
            # Weight by frequency
            ref_weight = ref_counter[word] / total_ref_freq
            cand_weight = cand_counter[word] / total_cand_freq
            overlap_score += min(ref_weight, cand_weight)
        
        # Normalize by average document length
        avg_length = (len(ref_tokens) + len(cand_tokens)) / 2
        normalized_score = overlap_score * avg_length
        
        return round(min(normalized_score, 1.0), 4)
    
    def evaluate_single_pair(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Evaluate a single reference-candidate pair with all metrics.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            Dictionary with all evaluation scores
        """
        # BLEU scores
        bleu_scores = self.calculate_bleu_scores(reference, candidate)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(reference, candidate)
        
        # Similarity metrics
        char_cosine_sim = self.calculate_cosine_similarity(reference, candidate, 'char')
        word_cosine_sim = self.calculate_cosine_similarity(reference, candidate, 'word')
        jaccard_sim = self.calculate_jaccard_similarity(reference, candidate)
        lin_sim = self.calculate_lin_similarity(reference, candidate)
        semantic_sim = self.calculate_semantic_similarity(reference, candidate)
        
        # Combine all metrics
        results = {
            **bleu_scores,
            **rouge_scores,
            'char_cosine_similarity': char_cosine_sim,
            'word_cosine_similarity': word_cosine_sim,
            'jaccard_similarity': jaccard_sim,
            'lin_similarity': lin_sim,
            'semantic_similarity': semantic_sim
        }
        
        return results


def create_metrics_calculator() -> EvaluationMetrics:
    """Factory function to create metrics calculator."""
    return EvaluationMetrics()