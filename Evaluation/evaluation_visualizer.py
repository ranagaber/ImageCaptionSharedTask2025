"""
Visualization utilities for evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import evaluation_config as config


class EvaluationVisualizer:
    """Create visualizations for evaluation results."""
    
    def __init__(self):
        """Initialize visualizer with style settings."""
        plt.style.use('default')
        sns.set_style(config.VISUALIZATION_CONFIG["style"])
        sns.set_palette(config.VISUALIZATION_CONFIG["color_palette"])
        self.figure_size = config.VISUALIZATION_CONFIG["figure_size"]
        self.dpi = config.VISUALIZATION_CONFIG["dpi"]
    
    def plot_metrics_distribution(self, results_df: pd.DataFrame, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of evaluation metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                axes[i].hist(results_df[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{config.METRICS_DISPLAY_NAMES.get(metric, metric)} Distribution')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, results_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix of evaluation metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        metrics = ['bleu4', 'rouge1', 'rougeL', 'char_cosine_similarity', 
                  'word_cosine_similarity', 'jaccard_similarity', 'lin_similarity']
        
        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("No metrics available for correlation matrix")
            return None
        
        correlation_matrix = results_df[available_metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        
        plt.title('Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_categories(self, results_df: pd.DataFrame,
                                  metric: str = 'bleu4',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance categories based on a specific metric.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to use for categorization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if metric not in results_df.columns:
            print(f"Metric {metric} not found in results")
            return None
        
        # Define categories based on BLEU-4 thresholds
        categories = config.PERFORMANCE_CATEGORIES
        
        category_counts = {}
        scores = results_df[metric]
        
        category_counts['Excellent (>0.5)'] = len(scores[scores > categories['excellent']])
        category_counts['Good (0.3-0.5)'] = len(scores[(scores > categories['good']) & (scores <= categories['excellent'])])
        category_counts['Fair (0.1-0.3)'] = len(scores[(scores > categories['fair']) & (scores <= categories['good'])])
        category_counts['Poor (≤0.1)'] = len(scores[scores <= categories['fair']])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        categories_list = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = ['green', 'orange', 'yellow', 'red']
        
        bars = ax1.bar(categories_list, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Performance Categories ({config.METRICS_DISPLAY_NAMES.get(metric, metric)})')
        ax1.set_ylabel('Number of Captions')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=categories_list, colors=colors, autopct='%1.1f%%', 
               startangle=90)
        ax2.set_title(f'Performance Distribution ({config.METRICS_DISPLAY_NAMES.get(metric, metric)})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_metric_comparison(self, results_df: pd.DataFrame,
                             metrics: List[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create box plots comparing different metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            metrics: List of metrics to compare
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("No specified metrics available for comparison")
            return None
        
        # Prepare data for plotting
        metric_data = []
        metric_names = []
        
        for metric in available_metrics:
            metric_data.append(results_df[metric].values)
            metric_names.append(config.METRICS_DISPLAY_NAMES.get(metric, metric))
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        box_plot = ax.boxplot(metric_data, labels=metric_names, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette(config.VISUALIZATION_CONFIG["color_palette"], len(available_metrics))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Comparison of Evaluation Metrics')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_length_analysis(self, detailed_results_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze the relationship between text length and performance.
        
        Args:
            detailed_results_df: DataFrame with detailed results including lengths
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if 'reference_length' not in detailed_results_df.columns or 'candidate_length' not in detailed_results_df.columns:
            print("Length columns not found in detailed results")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Length distribution
        axes[0,0].hist(detailed_results_df['reference_length'], bins=30, alpha=0.7, 
                      label='Reference', edgecolor='black')
        axes[0,0].hist(detailed_results_df['candidate_length'], bins=30, alpha=0.7, 
                      label='Candidate', edgecolor='black')
        axes[0,0].set_title('Text Length Distribution')
        axes[0,0].set_xlabel('Length (tokens)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Length ratio distribution
        if 'length_ratio' in detailed_results_df.columns:
            axes[0,1].hist(detailed_results_df['length_ratio'], bins=30, alpha=0.7, 
                          edgecolor='black')
            axes[0,1].set_title('Length Ratio Distribution (Candidate/Reference)')
            axes[0,1].set_xlabel('Length Ratio')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
        
        # Performance vs length
        if 'bleu4' in detailed_results_df.columns:
            axes[1,0].scatter(detailed_results_df['reference_length'], 
                            detailed_results_df['bleu4'], alpha=0.6)
            axes[1,0].set_title('BLEU-4 vs Reference Length')
            axes[1,0].set_xlabel('Reference Length (tokens)')
            axes[1,0].set_ylabel('BLEU-4 Score')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(detailed_results_df['reference_length'], 
                          detailed_results_df['bleu4'], 1)
            p = np.poly1d(z)
            axes[1,0].plot(detailed_results_df['reference_length'], 
                          p(detailed_results_df['reference_length']), "r--", alpha=0.8)
        
        # Performance vs length ratio
        if 'bleu4' in detailed_results_df.columns and 'length_ratio' in detailed_results_df.columns:
            axes[1,1].scatter(detailed_results_df['length_ratio'], 
                            detailed_results_df['bleu4'], alpha=0.6)
            axes[1,1].set_title('BLEU-4 vs Length Ratio')
            axes[1,1].set_xlabel('Length Ratio (Candidate/Reference)')
            axes[1,1].set_ylabel('BLEU-4 Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self, results_df: pd.DataFrame,
                                  detailed_results_df: pd.DataFrame = None,
                                  output_dir: str = "./evaluation_plots/") -> Dict[str, plt.Figure]:
        """
        Create a comprehensive set of evaluation visualizations.
        
        Args:
            results_df: DataFrame with metrics
            detailed_results_df: DataFrame with detailed results
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of figure objects
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        # Metrics distribution
        fig1 = self.plot_metrics_distribution(results_df, 
                                            f"{output_dir}/metrics_distribution.png")
        figures['metrics_distribution'] = fig1
        
        # Correlation matrix
        fig2 = self.plot_correlation_matrix(results_df,
                                          f"{output_dir}/correlation_matrix.png")
        if fig2:
            figures['correlation_matrix'] = fig2
        
        # Performance categories
        fig3 = self.plot_performance_categories(results_df,
                                              save_path=f"{output_dir}/performance_categories.png")
        if fig3:
            figures['performance_categories'] = fig3
        
        # Metric comparison
        fig4 = self.plot_metric_comparison(results_df,
                                         save_path=f"{output_dir}/metric_comparison.png")
        if fig4:
            figures['metric_comparison'] = fig4
        
        # Length analysis (if detailed results available)
        if detailed_results_df is not None:
            fig5 = self.plot_length_analysis(detailed_results_df,
                                           f"{output_dir}/length_analysis.png")
            if fig5:
                figures['length_analysis'] = fig5
        
        print(f"Saved {len(figures)} visualizations to {output_dir}")
        return figures


def create_visualizer() -> EvaluationVisualizer:
    """Factory function to create visualizer."""
    return EvaluationVisualizer()