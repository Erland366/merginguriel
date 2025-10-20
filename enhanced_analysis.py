#!/usr/bin/env python3
"""
Enhanced Cross-Lingual Model Merging Analysis
Shows comparison against both average and best zero-shot performance
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

class EnhancedCrossLingualAnalyzer:
    def __init__(self):
        self.results_comparison_path = "results_comparison_20250917_064315.csv"
        self.nxn_results_path = "nxn_results/nxn_eval_20250915_101911/evaluation_matrix.csv"
        self.merged_models_path = "merged_models"
        self.load_data()

    def load_data(self):
        """Load all necessary data files"""
        self.results_df = pd.read_csv(self.results_comparison_path)
        self.results_df = self.results_df.dropna()
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
        return locale_pattern.findall(text)

    def analyze_single_language(self, target_locale):
        """Analyze results for a single target language with enhanced zero-shot analysis"""
        # Get merge information
        similarity_locales = self.find_merge_locales(target_locale, "similarity")
        average_locales = self.find_merge_locales(target_locale, "average")

        # Get results from comparison file
        result_row = self.results_df[self.results_df['locale'] == target_locale]
        if len(result_row) == 0:
            return None
        result_data = result_row.iloc[0]

        # Get detailed zero-shot performance
        similarity_zero_shot_scores = []
        average_zero_shot_scores = []
        all_zero_shot_scores = []

        # Similarity merge zero-shot scores
        for locale in similarity_locales:
            if locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[locale, target_locale]
                similarity_zero_shot_scores.append(score)
                all_zero_shot_scores.append(score)

        # Average merge zero-shot scores
        for locale in average_locales:
            if locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[locale, target_locale]
                average_zero_shot_scores.append(score)
                all_zero_shot_scores.append(score)

        # Calculate statistics
        avg_similarity_zero_shot = np.mean(similarity_zero_shot_scores) if similarity_zero_shot_scores else 0
        avg_average_zero_shot = np.mean(average_zero_shot_scores) if average_zero_shot_scores else 0
        best_similarity_zero_shot = max(similarity_zero_shot_scores) if similarity_zero_shot_scores else 0
        best_average_zero_shot = max(average_zero_shot_scores) if average_zero_shot_scores else 0
        best_overall_zero_shot = max(all_zero_shot_scores) if all_zero_shot_scores else 0

        return {
            'target_locale': target_locale,
            'baseline_score': result_data['baseline'],
            'similarity_merge_score': result_data['similarity'],
            'average_merge_score': result_data['average'],
            'similarity_locales': similarity_locales,
            'average_locales': average_locales,
            'similarity_zero_shot_scores': similarity_zero_shot_scores,
            'average_zero_shot_scores': average_zero_shot_scores,
            'avg_similarity_zero_shot': avg_similarity_zero_shot,
            'avg_average_zero_shot': avg_average_zero_shot,
            'best_similarity_zero_shot': best_similarity_zero_shot,
            'best_average_zero_shot': best_average_zero_shot,
            'best_overall_zero_shot': best_overall_zero_shot,
            'all_zero_shot_scores': all_zero_shot_scores
        }

    def generate_enhanced_report(self):
        """Generate enhanced analysis comparing against both average and best zero-shot"""
        print("Generating enhanced cross-lingual analysis...")

        all_results = []
        for locale in self.results_df['locale']:
            result = self.analyze_single_language(locale)
            if result:
                all_results.append(result)

        self.create_enhanced_summary(all_results)
        self.create_enhanced_visualizations(all_results)
        self.generate_enhanced_csv(all_results)

        print("Enhanced analysis complete!")
        print("- enhanced_cross_lingual_summary.csv")
        print("- enhanced_performance_comparison.png")
        print("- zero_shot_analysis.png")
        print("- best_vs_average_comparison.png")

        return all_results

    def create_enhanced_summary(self, all_results):
        """Create enhanced summary statistics"""
        summary_data = []

        for result in all_results:
            # Compare against average zero-shot
            sim_vs_avg_zero = result['similarity_merge_score'] - result['avg_similarity_zero_shot']
            avg_vs_avg_zero = result['average_merge_score'] - result['avg_average_zero_shot']

            # Compare against best zero-shot
            sim_vs_best_zero = result['similarity_merge_score'] - result['best_overall_zero_shot']
            avg_vs_best_zero = result['average_merge_score'] - result['best_overall_zero_shot']

            summary_data.append({
                'locale': result['target_locale'],
                'baseline': result['baseline_score'],
                'similarity_merge': result['similarity_merge_score'],
                'average_merge': result['average_merge_score'],
                'avg_similarity_zero_shot': result['avg_similarity_zero_shot'],
                'avg_average_zero_shot': result['avg_average_zero_shot'],
                'best_overall_zero_shot': result['best_overall_zero_shot'],
                'sim_vs_avg_zero': sim_vs_avg_zero,
                'avg_vs_avg_zero': avg_vs_avg_zero,
                'sim_vs_best_zero': sim_vs_best_zero,
                'avg_vs_best_zero': avg_vs_best_zero,
                'best_merge_method': 'similarity' if result['similarity_merge_score'] > result['average_merge_score'] else 'average',
                'beats_avg_zero_shot': max(sim_vs_avg_zero, avg_vs_avg_zero) > 0,
                'beats_best_zero_shot': max(sim_vs_best_zero, avg_vs_best_zero) > 0
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('enhanced_cross_lingual_summary.csv', index=False)

        print("\n=== Enhanced Statistics ===")
        print(f"Languages where merge beats AVERAGE zero-shot: {summary_df['beats_avg_zero_shot'].sum()}/{len(summary_df)} ({summary_df['beats_avg_zero_shot'].mean()*100:.1f}%)")
        print(f"Languages where merge beats BEST zero-shot: {summary_df['beats_best_zero_shot'].sum()}/{len(summary_df)} ({summary_df['beats_best_zero_shot'].mean()*100:.1f}%)")

        print(f"\nAverage improvement vs average zero-shot: {summary_df[['sim_vs_avg_zero', 'avg_vs_avg_zero']].mean().mean():.4f}")
        print(f"Average improvement vs best zero-shot: {summary_df[['sim_vs_best_zero', 'avg_vs_best_zero']].mean().mean():.4f}")

        return summary_df

    def create_enhanced_visualizations(self, all_results):
        """Create enhanced visualizations"""
        # 1. Main comparison plot
        self.create_enhanced_performance_plot(all_results)

        # 2. Zero-shot analysis plot
        self.create_zero_shot_analysis_plot(all_results)

        # 3. Best vs Average comparison
        self.create_best_vs_average_plot(all_results)

    def create_enhanced_performance_plot(self, all_results):
        """Create enhanced performance comparison"""
        locales = [r['target_locale'] for r in all_results]
        similarities = [r['similarity_merge_score'] for r in all_results]
        averages = [r['average_merge_score'] for r in all_results]
        avg_zeros = [r['avg_similarity_zero_shot'] for r in all_results]
        best_zeros = [r['best_overall_zero_shot'] for r in all_results]

        plt.figure(figsize=(16, 8))
        x = np.arange(len(locales))
        width = 0.2

        plt.bar(x - 1.5*width, avg_zeros, width, label='Avg Zero-shot', alpha=0.7, color='lightgray')
        plt.bar(x - 0.5*width, best_zeros, width, label='Best Zero-shot', alpha=0.7, color='darkgray')
        plt.bar(x + 0.5*width, similarities, width, label='Similarity Merge', alpha=0.8, color='blue')
        plt.bar(x + 1.5*width, averages, width, label='Average Merge', alpha=0.8, color='red')

        plt.xlabel('Target Languages')
        plt.ylabel('Performance Score')
        plt.title('Enhanced Cross-Lingual Performance: Merged Models vs Zero-shot (Average & Best)')
        plt.xticks(x, locales, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('enhanced_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_zero_shot_analysis_plot(self, all_results):
        """Create detailed zero-shot analysis"""
        locales = [r['target_locale'] for r in all_results]
        sim_vs_avg = [r['similarity_merge_score'] - r['avg_similarity_zero_shot'] for r in all_results]
        avg_vs_avg = [r['average_merge_score'] - r['avg_average_zero_shot'] for r in all_results]
        sim_vs_best = [r['similarity_merge_score'] - r['best_overall_zero_shot'] for r in all_results]
        avg_vs_best = [r['average_merge_score'] - r['best_overall_zero_shot'] for r in all_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: vs Average zero-shot
        x = np.arange(len(locales))
        width = 0.35

        bars1 = ax1.bar(x - width/2, sim_vs_avg, width, label='Similarity vs Avg Zero-shot', alpha=0.8)
        bars2 = ax1.bar(x + width/2, avg_vs_avg, width, label='Average vs Avg Zero-shot', alpha=0.8)

        for bar in bars1:
            bar.set_color('blue' if bar.get_height() >= 0 else 'lightblue')
        for bar in bars2:
            bar.set_color('red' if bar.get_height() >= 0 else 'lightcoral')

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_ylabel('Advantage over Average Zero-shot')
        ax1.set_title('Merge Methods vs Average Zero-shot Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(locales, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: vs Best zero-shot
        bars3 = ax2.bar(x - width/2, sim_vs_best, width, label='Similarity vs Best Zero-shot', alpha=0.8)
        bars4 = ax2.bar(x + width/2, avg_vs_best, width, label='Average vs Best Zero-shot', alpha=0.8)

        for bar in bars3:
            bar.set_color('blue' if bar.get_height() >= 0 else 'lightblue')
        for bar in bars4:
            bar.set_color('red' if bar.get_height() >= 0 else 'lightcoral')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Advantage over Best Zero-shot')
        ax2.set_title('Merge Methods vs Best Zero-shot Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(locales, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('zero_shot_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_best_vs_average_plot(self, all_results):
        """Create comparison showing when merge beats both average and best zero-shot"""
        summary_df = pd.read_csv('enhanced_cross_lingual_summary.csv')

        # Create categories
        categories = []
        for _, row in summary_df.iterrows():
            if row['beats_avg_zero_shot'] and row['beats_best_zero_shot']:
                categories.append('Beats Both')
            elif row['beats_avg_zero_shot']:
                categories.append('Beats Avg Only')
            elif row['beats_best_zero_shot']:
                categories.append('Beats Best Only')
            else:
                categories.append('Beats Neither')

        summary_df['category'] = categories

        # Count categories
        category_counts = summary_df['category'].value_counts()

        # Create pie chart
        plt.figure(figsize=(10, 8))
        colors = ['green', 'lightgreen', 'orange', 'red']
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Merge Performance vs Zero-shot Benchmarks')
        plt.tight_layout()
        plt.savefig('best_vs_average_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print detailed breakdown
        print(f"\n=== Performance Breakdown ===")
        for category, count in category_counts.items():
            print(f"{category}: {count} languages ({count/len(summary_df)*100:.1f}%)")

    def generate_enhanced_csv(self, all_results):
        """Generate enhanced CSV with all information"""
        detailed_data = []

        for result in all_results:
            row = {
                'target_locale': result['target_locale'],
                'baseline_score': result['baseline_score'],
                'similarity_merge_score': result['similarity_merge_score'],
                'average_merge_score': result['average_merge_score'],
                'avg_similarity_zero_shot': result['avg_similarity_zero_shot'],
                'avg_average_zero_shot': result['avg_average_zero_shot'],
                'best_similarity_zero_shot': result['best_similarity_zero_shot'],
                'best_average_zero_shot': result['best_average_zero_shot'],
                'best_overall_zero_shot': result['best_overall_zero_shot'],
                'sim_vs_avg_zero': result['similarity_merge_score'] - result['avg_similarity_zero_shot'],
                'avg_vs_avg_zero': result['average_merge_score'] - result['avg_average_zero_shot'],
                'sim_vs_best_zero': result['similarity_merge_score'] - result['best_overall_zero_shot'],
                'avg_vs_best_zero': result['average_merge_score'] - result['best_overall_zero_shot'],
                'best_merge_method': 'similarity' if result['similarity_merge_score'] > result['average_merge_score'] else 'average',
                'similarity_source_locales': ', '.join(result['similarity_locales']),
                'average_source_locales': ', '.join(result['average_locales'])
            }

            # Add individual zero-shot scores
            for i, score in enumerate(result['similarity_zero_shot_scores']):
                row[f'sim_zero_shot_{i+1}'] = score
            for i, score in enumerate(result['average_zero_shot_scores']):
                row[f'avg_zero_shot_{i+1}'] = score

            detailed_data.append(row)

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv('enhanced_merge_details.csv', index=False)

        return detailed_df

if __name__ == "__main__":
    analyzer = EnhancedCrossLingualAnalyzer()
    all_results = analyzer.generate_enhanced_report()