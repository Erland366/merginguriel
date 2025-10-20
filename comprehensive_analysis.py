#!/usr/bin/env python3
"""
Comprehensive Cross-Lingual Model Merging Analysis
Analyzes the results of language model merging experiments across multiple languages
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class CrossLingualAnalyzer:
    def __init__(self):
        self.results_comparison_path = "results_comparison_20250917_064315.csv"
        self.nxn_results_path = "nxn_results/nxn_eval_20250915_101911/evaluation_matrix.csv"
        self.merged_models_path = "merged_models"

        # Load data
        self.load_data()

    def load_data(self):
        """Load all necessary data files"""
        # Load results comparison
        self.results_df = pd.read_csv(self.results_comparison_path)
        self.results_df = self.results_df.dropna()

        # Load NxN evaluation matrix
        self.nxn_df = pd.read_csv(self.nxn_results_path, index_col=0)
        self.nxn_df = self.nxn_df.dropna()

        print(f"Loaded results for {len(self.results_df)} languages")
        print(f"Loaded NxN matrix with {len(self.nxn_df)} languages")

    def find_merge_locales(self, target_locale, merge_type="similarity"):
        """Find which locales were used for merging a target language"""
        # Try to find merge directory with new naming convention first
        merge_path = None

        # Look for directories matching pattern: {merge_type}_merge_{target_locale}_{N}merged
        pattern = os.path.join(self.merged_models_path, f"{merge_type}_merge_{target_locale}_*merged")
        matching_dirs = glob.glob(pattern)

        if matching_dirs:
            merge_path = os.path.join(matching_dirs[0], "merge_details.txt")

        # Fallback to old naming convention
        if merge_path is None or not os.path.exists(merge_path):
            merge_path = os.path.join(self.merged_models_path, f"{merge_type}_merge_{target_locale}", "merge_details.txt")

        if not os.path.exists(merge_path):
            return []

        with open(merge_path, 'r') as f:
            text = f.read()

        locale_pattern = re.compile(r"^\s*- Locale: \s*([a-zA-Z]{2}-[a-zA-Z]{2})$", re.MULTILINE)
        found_locales = locale_pattern.findall(text)

        return found_locales

    def get_merge_weights(self, target_locale, merge_type="similarity"):
        """Extract merge weights for a target language"""
        # Try to find merge directory with new naming convention first
        merge_path = None

        # Look for directories matching pattern: {merge_type}_merge_{target_locale}_{N}merged
        pattern = os.path.join(self.merged_models_path, f"{merge_type}_merge_{target_locale}_*merged")
        matching_dirs = glob.glob(pattern)

        if matching_dirs:
            merge_path = os.path.join(matching_dirs[0], "merge_details.txt")

        # Fallback to old naming convention
        if merge_path is None or not os.path.exists(merge_path):
            merge_path = os.path.join(self.merged_models_path, f"{merge_type}_merge_{target_locale}", "merge_details.txt")

        if not os.path.exists(merge_path):
            return {}

        with open(merge_path, 'r') as f:
            text = f.read()

        weight_pattern = re.compile(r"^\s*- Locale: \s*([a-zA-Z]{2}-[a-zA-Z]{2}).*?Weight: ([0-9.]+)", re.MULTILINE | re.DOTALL)
        matches = weight_pattern.findall(text)

        return {locale: float(weight) for locale, weight in matches}

    def analyze_single_language(self, target_locale):
        """Analyze results for a single target language"""
        # Get merge information
        similarity_locales = self.find_merge_locales(target_locale, "similarity")
        average_locales = self.find_merge_locales(target_locale, "average")

        # Get results from comparison file
        result_row = self.results_df[self.results_df['locale'] == target_locale]
        if len(result_row) == 0:
            return None

        result_data = result_row.iloc[0]

        # Get cross-lingual scores (zero-shot performance)
        cross_lingual_scores = {}
        similarity_zero_shot_scores = []
        average_zero_shot_scores = []

        # Similarity merge cross-lingual scores (zero-shot from source languages)
        for locale in similarity_locales:
            if locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[locale, target_locale]
                cross_lingual_scores[f"sim_{locale}"] = score
                similarity_zero_shot_scores.append(score)

        # Average merge cross-lingual scores (zero-shot from source languages)
        for locale in average_locales:
            if locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[locale, target_locale]
                cross_lingual_scores[f"avg_{locale}"] = score
                average_zero_shot_scores.append(score)

        # Calculate average zero-shot performance for comparison
        avg_similarity_zero_shot = np.mean(similarity_zero_shot_scores) if similarity_zero_shot_scores else 0
        avg_average_zero_shot = np.mean(average_zero_shot_scores) if average_zero_shot_scores else 0

        # Get merge weights
        similarity_weights = self.get_merge_weights(target_locale, "similarity")
        average_weights = self.get_merge_weights(target_locale, "average")

        return {
            'target_locale': target_locale,
            'baseline_score': result_data['baseline'],  # Original finetuned model
            'similarity_merge_score': result_data['similarity'],  # Your similarity merge method
            'average_merge_score': result_data['average'],  # Your average merge method
            'similarity_locales': similarity_locales,
            'average_locales': average_locales,
            'cross_lingual_scores': cross_lingual_scores,
            'similarity_weights': similarity_weights,
            'average_weights': average_weights,
            'avg_similarity_zero_shot': avg_similarity_zero_shot,
            'avg_average_zero_shot': avg_average_zero_shot,
            'similarity_zero_shot_scores': similarity_zero_shot_scores,
            'average_zero_shot_scores': average_zero_shot_scores
        }

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis for all languages"""
        print("Generating comprehensive cross-lingual analysis...")

        all_results = []

        # Analyze each language
        for locale in self.results_df['locale']:
            result = self.analyze_single_language(locale)
            if result:
                all_results.append(result)

        # Create summary statistics
        self.create_summary_statistics(all_results)

        # Create visualizations
        self.create_key_performance_plot(all_results)
        self.create_merge_improvement_plot(all_results)
        self.create_zero_shot_comparison_plot(all_results)

        # Generate detailed CSV reports
        self.generate_detailed_csv(all_results)

        print("Analysis complete! Check the generated files:")
        print("- cross_lingual_summary.csv")
        print("- merge_details.csv")
        print("- key_performance_comparison.png")
        print("- merge_improvement_analysis.png")
        print("- zero_shot_comparison.png")

        return all_results

    def create_summary_statistics(self, all_results):
        """Create summary statistics"""
        summary_data = []

        for result in all_results:
            # Calculate improvements over zero-shot performance
            sim_improvement_vs_zero_shot = result['similarity_merge_score'] - result['avg_similarity_zero_shot']
            avg_improvement_vs_zero_shot = result['average_merge_score'] - result['avg_average_zero_shot']

            summary_data.append({
                'locale': result['target_locale'],
                'baseline': result['baseline_score'],
                'similarity_merge': result['similarity_merge_score'],
                'average_merge': result['average_merge_score'],
                'similarity_zero_shot_avg': result['avg_similarity_zero_shot'],
                'average_zero_shot_avg': result['avg_average_zero_shot'],
                'sim_improvement_vs_zero_shot': sim_improvement_vs_zero_shot,
                'avg_improvement_vs_zero_shot': avg_improvement_vs_zero_shot,
                'best_merge_method': 'similarity' if result['similarity_merge_score'] > result['average_merge_score'] else 'average',
                'num_similarity_sources': len(result['similarity_locales']),
                'num_average_sources': len(result['average_locales']),
                'merge_vs_zero_shot_advantage': max(sim_improvement_vs_zero_shot, avg_improvement_vs_zero_shot)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('cross_lingual_summary.csv', index=False)

        # Print overall statistics
        print("\n=== Key Statistics ===")
        print(f"Average baseline (finetuned): {summary_df['baseline'].mean():.4f}")
        print(f"Average similarity merge: {summary_df['similarity_merge'].mean():.4f}")
        print(f"Average average merge: {summary_df['average_merge'].mean():.4f}")
        print(f"Average similarity zero-shot: {summary_df['similarity_zero_shot_avg'].mean():.4f}")
        print(f"Average average zero-shot: {summary_df['average_zero_shot_avg'].mean():.4f}")
        print(f"Average similarity improvement over zero-shot: {summary_df['sim_improvement_vs_zero_shot'].mean():.4f}")
        print(f"Average average improvement over zero-shot: {summary_df['avg_improvement_vs_zero_shot'].mean():.4f}")

        print(f"\nLanguages where similarity merge performs better: {(summary_df['best_merge_method'] == 'similarity').sum()}")
        print(f"Languages where average merge performs better: {(summary_df['best_merge_method'] == 'average').sum()}")

        print(f"\nLanguages with positive improvement over zero-shot: {(summary_df['merge_vs_zero_shot_advantage'] > 0).sum()}")
        print(f"Languages with negative improvement over zero-shot: {(summary_df['merge_vs_zero_shot_advantage'] < 0).sum()}")

        return summary_df

    def create_key_performance_plot(self, all_results):
        """Create key performance comparison plot"""
        locales = [r['target_locale'] for r in all_results]
        similarity_merges = [r['similarity_merge_score'] for r in all_results]
        average_merges = [r['average_merge_score'] for r in all_results]
        similarity_zero_shots = [r['avg_similarity_zero_shot'] for r in all_results]
        average_zero_shots = [r['avg_average_zero_shot'] for r in all_results]

        plt.figure(figsize=(16, 8))
        x = np.arange(len(locales))
        width = 0.2

        plt.bar(x - 1.5*width, similarity_zero_shots, width, label='Similarity Zero-shot', alpha=0.7, color='lightblue')
        plt.bar(x - 0.5*width, similarity_merges, width, label='Similarity Merge', alpha=0.8, color='blue')
        plt.bar(x + 0.5*width, average_zero_shots, width, label='Average Zero-shot', alpha=0.7, color='lightcoral')
        plt.bar(x + 1.5*width, average_merges, width, label='Average Merge', alpha=0.8, color='red')

        plt.xlabel('Target Languages')
        plt.ylabel('Performance Score')
        plt.title('Cross-Lingual Performance: Merged Models vs Zero-shot')
        plt.xticks(x, locales, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('key_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_merge_improvement_plot(self, all_results):
        """Create improvement analysis plot"""
        sim_improvements = []
        avg_improvements = []
        locale_labels = []

        for result in all_results:
            sim_imp = result['similarity_merge_score'] - result['avg_similarity_zero_shot']
            avg_imp = result['average_merge_score'] - result['avg_average_zero_shot']

            sim_improvements.append(sim_imp)
            avg_improvements.append(avg_imp)
            locale_labels.append(result['target_locale'])

        plt.figure(figsize=(12, 8))
        plt.scatter(sim_improvements, avg_improvements, alpha=0.7, s=50)

        # Add diagonal line
        min_val = min(min(sim_improvements), min(avg_improvements))
        max_val = max(max(sim_improvements), max(avg_improvements))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        # Add labels for significant points
        for i, label in enumerate(locale_labels):
            if abs(sim_improvements[i] - avg_improvements[i]) > 0.02:
                plt.annotate(label, (sim_improvements[i], avg_improvements[i]),
                            fontsize=8, alpha=0.7)

        # Add zero lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        plt.xlabel('Similarity Merge Improvement over Zero-shot')
        plt.ylabel('Average Merge Improvement over Zero-shot')
        plt.title('Merge Method Improvement Analysis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('merge_improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_zero_shot_comparison_plot(self, all_results):
        """Create comparison showing merge vs zero-shot"""
        locales = [r['target_locale'] for r in all_results]
        sim_better = []
        avg_better = []

        for result in all_results:
            sim_advantage = result['similarity_merge_score'] - result['avg_similarity_zero_shot']
            avg_advantage = result['average_merge_score'] - result['avg_average_zero_shot']

            sim_better.append(sim_advantage)
            avg_better.append(avg_advantage)

        plt.figure(figsize=(14, 8))
        x = np.arange(len(locales))
        width = 0.35

        bars1 = plt.bar(x - width/2, sim_better, width, label='Similarity Merge Advantage', alpha=0.8)
        bars2 = plt.bar(x + width/2, avg_better, width, label='Average Merge Advantage', alpha=0.8)

        # Color similarity bars: blue for positive, light blue for negative
        for bar in bars1:
            if bar.get_height() >= 0:
                bar.set_color('blue')
            else:
                bar.set_color('lightblue')

        # Color average bars: red for positive, light coral for negative
        for bar in bars2:
            if bar.get_height() >= 0:
                bar.set_color('red')
            else:
                bar.set_color('lightcoral')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Target Languages')
        plt.ylabel('Advantage over Zero-shot (Positive = Better)')
        plt.title('Merge Methods vs Zero-shot Performance by Language')
        plt.xticks(x, locales, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('zero_shot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_detailed_csv(self, all_results):
        """Generate detailed CSV with all information"""
        detailed_data = []

        for result in all_results:
            row = {
                'target_locale': result['target_locale'],
                'baseline_score': result['baseline_score'],
                'similarity_merge_score': result['similarity_merge_score'],
                'average_merge_score': result['average_merge_score'],
                'similarity_zero_shot_avg': result['avg_similarity_zero_shot'],
                'average_zero_shot_avg': result['avg_average_zero_shot'],
                'sim_improvement_vs_zero_shot': result['similarity_merge_score'] - result['avg_similarity_zero_shot'],
                'avg_improvement_vs_zero_shot': result['average_merge_score'] - result['avg_average_zero_shot'],
                'best_merge_method': 'similarity' if result['similarity_merge_score'] > result['average_merge_score'] else 'average',
                'similarity_source_locales': ', '.join(result['similarity_locales']),
                'average_source_locales': ', '.join(result['average_locales']),
                'similarity_source_count': len(result['similarity_locales']),
                'average_source_count': len(result['average_locales'])
            }

            # Add weight information
            if result['similarity_weights']:
                row['similarity_weights'] = '; '.join([f"{k}:{v:.3f}" for k, v in result['similarity_weights'].items()])
            else:
                row['similarity_weights'] = ''

            if result['average_weights']:
                row['average_weights'] = '; '.join([f"{k}:{v:.3f}" for k, v in result['average_weights'].items()])
            else:
                row['average_weights'] = ''

            # Add individual zero-shot scores
            for score_name, score_value in result['cross_lingual_scores'].items():
                row[f'zero_shot_{score_name}'] = score_value

            detailed_data.append(row)

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv('merge_details.csv', index=False)

        return detailed_df

    def create_highlight_examples(self, num_examples=5):
        """Create detailed examples of interesting cases"""
        print(f"\n=== Highlight Examples (Top {num_examples}) ===")

        # Load summary data
        summary_df = pd.read_csv('cross_lingual_summary.csv')

        # Most successful merges (biggest improvement over zero-shot)
        most_successful = summary_df.nlargest(num_examples, 'merge_vs_zero_shot_advantage')
        print("\nMost Successful Merges (Biggest Improvement over Zero-shot):")
        print(most_successful[['locale', 'sim_improvement_vs_zero_shot', 'avg_improvement_vs_zero_shot', 'best_merge_method']].to_string(index=False))

        # Languages where merging hurts performance
        worst_performance = summary_df.nsmallest(num_examples, 'merge_vs_zero_shot_advantage')
        print(f"\nLanguages Where Merging Hurts Performance:")
        print(worst_performance[['locale', 'sim_improvement_vs_zero_shot', 'avg_improvement_vs_zero_shot', 'best_merge_method']].to_string(index=False))

        # Languages with biggest difference between merge methods
        summary_df['method_difference'] = abs(summary_df['sim_improvement_vs_zero_shot'] - summary_df['avg_improvement_vs_zero_shot'])
        biggest_method_diff = summary_df.nlargest(num_examples, 'method_difference')
        print(f"\nLanguages with Biggest Method Differences:")
        print(biggest_method_diff[['locale', 'sim_improvement_vs_zero_shot', 'avg_improvement_vs_zero_shot', 'best_merge_method']].to_string(index=False))

def main():
    """Main function to run the analysis"""
    analyzer = CrossLingualAnalyzer()

    # Generate comprehensive report
    all_results = analyzer.generate_comprehensive_report()

    # Create highlight examples
    analyzer.create_highlight_examples()

    print(f"\nAnalysis complete! Generated {len(all_results)} language analyses.")

if __name__ == "__main__":
    main()