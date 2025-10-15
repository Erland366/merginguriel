#!/usr/bin/env python3
"""
Advanced Results Analysis System for MergingUriel
Follows the sophisticated analysis patterns from enhanced_analysis.py and comprehensive_analysis.py
Includes advanced merging methods and ensemble analysis

Features:
- Advanced merging methods analysis (TIES, Task Arithmetic, SLERP, RegMean, DARE, Fisher)
- Ensemble inference methods analysis (majority, weighted_majority, soft, uriel_logits)
- Zero-shot performance comparison using N-x-N evaluation matrices
- Merge details extraction from merge_details.txt files
- Weight extraction and analysis from merge details
- Comprehensive baseline comparisons against both average and best zero-shot
- Publication-ready styling following user's established patterns
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# Set plotting style to match user preferences from enhanced_analysis.py
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300

class AdvancedResultsAnalyzer:
    """
    Advanced results analyzer that follows user's established patterns from
    enhanced_analysis.py and comprehensive_analysis.py
    """

    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.merged_models_path = self.results_dir / "merged_models"
        self.ensemble_results_path = self.results_dir / "ensemble_results"

        # Data storage
        self.results_dfs = {}
        self.nxn_df = None
        self.experiment_results = {}

        # Load all available data
        self.load_all_data()

    def load_all_data(self):
        """Load all available CSV files and N-x-N evaluation data"""
        print("Loading data files...")

        # Load latest aggregated results
        csv_files = glob.glob(str(self.results_dir / "results_*.csv"))
        if not csv_files:
            # Try alternative naming patterns
            csv_files.extend(glob.glob(str(self.results_dir / "*results*.csv")))
            csv_files.extend(glob.glob(str(self.results_dir / "*comparison*.csv")))

        if not csv_files:
            raise FileNotFoundError("No results CSV files found!")

        # Load the most recent CSV file
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"Loading main results from: {latest_csv}")

        self.main_results_df = pd.read_csv(latest_csv)
        # Don't drop all NaN rows, just clean up empty locale rows
        self.main_results_df = self.main_results_df.dropna(subset=['locale'])

        # Try to load N-x-N evaluation data
        nxn_files = glob.glob(str(self.results_dir / "nxn_results" / "*" / "evaluation_matrix.csv"))
        if nxn_files:
            latest_nxn = max(nxn_files, key=os.path.getctime)
            print(f"Loading N-x-N evaluation from: {latest_nxn}")
            self.nxn_df = pd.read_csv(latest_nxn, index_col=0)
            self.nxn_df = self.nxn_df.dropna()
        else:
            print("Warning: No N-x-N evaluation matrix found")

        # Load experiment results from directories
        self.load_experiment_directories()

        print(f"Loaded main results for {len(self.main_results_df)} entries")
        if self.nxn_df is not None:
            print(f"Loaded N-x-N matrix with {len(self.nxn_df)} languages")

    def load_experiment_directories(self):
        """Load results from individual experiment directories"""
        print("Loading experiment directories...")

        # Look for experiment directories
        exp_patterns = [
            "*_merge_*",  # similarity_merge_sq-AL, average_merge_sq-AL, etc.
            "ensemble_*_*",  # ensemble_majority_sq-AL, ensemble_uriel_logits_sq-AL, etc.
            "*baseline_*"  # baseline experiments
        ]

        for pattern in exp_patterns:
            for exp_dir in self.results_dir.glob(pattern):
                if exp_dir.is_dir():
                    result = self.load_experiment_result(exp_dir)
                    if result:
                        self.experiment_results[exp_dir.name] = result

        print(f"Loaded {len(self.experiment_results)} experiment results")

    def load_experiment_result(self, exp_dir: Path) -> Optional[Dict]:
        """Load result from a single experiment directory"""
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            return None

        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            # Extract key information
            return {
                'directory': exp_dir.name,
                'type': self.detect_experiment_type(exp_dir.name),
                'target_locale': data.get('target_locale'),
                'accuracy': data.get('performance', {}).get('accuracy', 0),
                'baseline_accuracy': data.get('performance', {}).get('baseline_accuracy', 0),
                'experiment_info': data.get('experiment_info', {}),
                'merge_details': self.load_merge_details(exp_dir) if 'merge' in exp_dir.name else None
            }
        except Exception as e:
            print(f"Error loading {exp_dir}: {e}")
            return None

    def detect_experiment_type(self, dir_name: str) -> str:
        """Detect experiment type from directory name"""
        if 'baseline' in dir_name:
            return 'baseline'
        elif 'ensemble' in dir_name:
            return 'ensemble'
        elif 'merge' in dir_name:
            # Extract merge method (similarity, average, ties, task_arithmetic, etc.)
            for method in ['similarity', 'average', 'ties', 'task_arithmetic', 'slerp', 'regmean', 'dare', 'fisher']:
                if method in dir_name:
                    return f'merge_{method}'
            return 'merge_unknown'
        else:
            return 'unknown'

    def load_merge_details(self, exp_dir: Path) -> Optional[Dict]:
        """Load merge details from merge_details.txt file"""
        merge_details_file = exp_dir / "merge_details.txt"
        if not merge_details_file.exists():
            return None

        try:
            with open(merge_details_file, 'r') as f:
                content = f.read()

            # Parse locales and weights
            locales = []
            weights = {}

            # Pattern to match locale lines
            locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
            locales = locale_pattern.findall(content)

            # Pattern to match locale-weight pairs
            weight_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2}).*?Weight:\s*([0-9.]+)",
                                       re.MULTILINE | re.DOTALL)
            weight_matches = weight_pattern.findall(content)
            weights = {locale: float(weight) for locale, weight in weight_matches}

            return {
                'source_locales': locales,
                'weights': weights,
                'content': content
            }
        except Exception as e:
            print(f"Error loading merge details from {exp_dir}: {e}")
            return None

    def find_merge_locales(self, target_locale: str, merge_type: str = "similarity") -> List[str]:
        """Find which locales were used for merging a target language"""
        merge_dir = self.merged_models_path / f"{merge_type}_merge_{target_locale}"
        merge_details_file = merge_dir / "merge_details.txt"

        if not merge_details_file.exists():
            return []

        try:
            with open(merge_details_file, 'r') as f:
                content = f.read()

            locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
            return locale_pattern.findall(content)
        except Exception:
            return []

    def get_zero_shot_scores(self, target_locale: str, source_locales: List[str]) -> Dict[str, float]:
        """Get zero-shot scores for target locale from source locales"""
        if self.nxn_df is None:
            return {}

        scores = {}
        for locale in source_locales:
            if locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[locale, target_locale]
                if not pd.isna(score):
                    scores[locale] = float(score)

        return scores

    def get_best_source_performance(self, target_locale: str, source_locales: List[str]) -> float:
        """Get the best performance among the actual source languages used for a target locale"""
        zero_shot_scores = self.get_zero_shot_scores(target_locale, source_locales)
        if zero_shot_scores:
            return max(zero_shot_scores.values())
        return 0.0

    def extract_source_locales_from_details(self, target_locale: str) -> List[str]:
        """Extract source locales from merge_details.txt files for any merge type"""
        # Try to find merge_details.txt from any merge type for this target locale
        merge_patterns = [
            f"merged_models/similarity_merge_{target_locale}/merge_details.txt",
            f"merged_models/average_merge_{target_locale}/merge_details.txt",
            f"merged_models/*merge_{target_locale}/merge_details.txt"
        ]

        for pattern in merge_patterns:
            merge_details_files = list(self.results_dir.glob(pattern))
            if merge_details_files:
                # Use the first one found
                merge_details_file = merge_details_files[0]
                try:
                    with open(merge_details_file, 'r') as f:
                        content = f.read()

                    # Extract locales using regex pattern for the new format
                    locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
                    locales = locale_pattern.findall(content)
                    if locales:
                        return locales
                except Exception as e:
                    print(f"Error reading {merge_details_file}: {e}")
                    continue

        # Fallback: try to find from any directory containing the target locale
        for exp_dir in self.results_dir.glob(f"merged_models/*{target_locale}"):
            if exp_dir.is_dir():
                merge_details_file = exp_dir / "merge_details.txt"
                if merge_details_file.exists():
                    try:
                        with open(merge_details_file, 'r') as f:
                            content = f.read()

                        locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
                        locales = locale_pattern.findall(content)
                        if locales:
                            return locales
                    except Exception:
                        continue

        return []

    def analyze_advanced_merging_methods(self) -> List[Dict]:
        """Analyze results for advanced merging methods"""
        print("Analyzing advanced merging methods...")

        advanced_methods = ['ties', 'task_arithmetic', 'slerp', 'regmean', 'dare', 'fisher']
        results = []

        for locale in self.main_results_df['locale'].unique():
            locale_data = {'target_locale': locale}

            # Get baseline data
            locale_row = self.main_results_df[self.main_results_df['locale'] == locale]
            if len(locale_row) == 0:
                continue

            main_data = locale_row.iloc[0]
            locale_data['baseline'] = main_data.get('baseline', 0)
            locale_data['similarity'] = main_data.get('similarity', 0)
            locale_data['average'] = main_data.get('average', 0)

            # Look for advanced method results
            for method in advanced_methods:
                method_key = method if method in main_data else f'{method}_merge'
                if method_key in main_data:
                    locale_data[method] = main_data[method_key]

            # Extract source locales from merge details
            source_locales = self.extract_source_locales_from_details(locale)
            zero_shot_scores = self.get_zero_shot_scores(locale, source_locales)

            if zero_shot_scores:
                locale_data['avg_zero_shot'] = np.mean(list(zero_shot_scores.values()))
                locale_data['best_zero_shot'] = max(zero_shot_scores.values())
                locale_data['best_source'] = self.get_best_source_performance(locale, source_locales)
                locale_data['source_locales'] = source_locales
            else:
                locale_data['avg_zero_shot'] = 0
                locale_data['best_zero_shot'] = 0
                locale_data['best_source'] = 0
                locale_data['source_locales'] = []

            results.append(locale_data)

        return results

    def analyze_ensemble_methods(self) -> List[Dict]:
        """Analyze ensemble inference methods from the main aggregated results"""
        print("Analyzing ensemble methods...")

        ensemble_methods = ['ensemble_majority', 'ensemble_weighted_majority', 'ensemble_soft', 'ensemble_uriel_logits']
        results = []

        # Extract ensemble data from main results dataframe
        for _, row in self.main_results_df.iterrows():
            locale = row['locale']

            # Get baseline accuracy for comparison
            baseline_accuracy = row.get('baseline', 0)

            # Extract source locales from merge details (same as merging methods)
            source_locales = self.extract_source_locales_from_details(locale)
            zero_shot_scores = self.get_zero_shot_scores(locale, source_locales)

            # Process each ensemble method that exists in the dataframe
            for method in ensemble_methods:
                if method in row and pd.notna(row[method]):
                    # Extract method name from ensemble_X
                    method_name = method.replace('ensemble_', '')

                    locale_data = {
                        'target_locale': locale,
                        'ensemble_method': method_name,
                        'ensemble_accuracy': row[method],
                        'baseline_accuracy': baseline_accuracy,
                        'source_locales': source_locales
                    }

                    # Calculate zero-shot baseline for comparison using same source locales
                    if zero_shot_scores:
                        locale_data['avg_zero_shot'] = np.mean(list(zero_shot_scores.values()))
                        locale_data['best_zero_shot'] = max(zero_shot_scores.values())
                        locale_data['best_source'] = self.get_best_source_performance(locale, source_locales)
                    else:
                        locale_data['avg_zero_shot'] = 0
                        locale_data['best_zero_shot'] = 0
                        locale_data['best_source'] = 0

                    results.append(locale_data)

        return results

    def create_advanced_performance_plot(self, merging_results: List[Dict]):
        """Create comprehensive performance comparison for advanced methods"""
        print("Creating advanced performance comparison...")

        # Filter results that have advanced methods
        advanced_methods = ['ties', 'task_arithmetic', 'slerp', 'regmean', 'dare', 'fisher']
        method_data = {method: [] for method in advanced_methods}
        method_data['similarity'] = []
        method_data['average'] = []
        method_data['baseline'] = []
        method_data['avg_zero_shot'] = []
        method_data['best_zero_shot'] = []
        locales = []

        for result in merging_results:
            locales.append(result['target_locale'])
            method_data['baseline'].append(result.get('baseline', 0))
            method_data['similarity'].append(result.get('similarity', 0))
            method_data['average'].append(result.get('average', 0))
            method_data['avg_zero_shot'].append(result.get('avg_zero_shot', 0))
            method_data['best_zero_shot'].append(result.get('best_zero_shot', 0))

            for method in advanced_methods:
                method_data[method].append(result.get(method, 0))

        # Create subplots for better readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

        # Plot 1: All methods comparison
        x = np.arange(len(locales))
        width = 0.08
        bar_positions = np.arange(-3, 4) * width
        methods = ['baseline', 'avg_zero_shot', 'similarity', 'average'] + advanced_methods[:3]
        colors = ['gray', 'lightgray', 'blue', 'red', 'green', 'orange', 'purple']

        for i, (method, color) in enumerate(zip(methods, colors)):
            if method in method_data and method_data[method]:
                ax1.bar(x + bar_positions[i], method_data[method], width,
                       label=method.replace('_', ' ').title(), alpha=0.8, color=color)

        ax1.set_xlabel('Target Languages')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Advanced Merging Methods Performance Comparison (Part 1)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(locales, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Remaining advanced methods
        methods_2 = advanced_methods[3:]
        colors_2 = ['brown', 'cyan', 'magenta']
        bar_positions_2 = np.arange(-1, len(methods_2)) * width

        for i, (method, color) in enumerate(zip(methods_2, colors_2)):
            if method in method_data and method_data[method]:
                ax2.bar(x + bar_positions_2[i], method_data[method], width,
                       label=method.replace('_', ' ').title(), alpha=0.8, color=color)

        # Also include baseline for reference
        ax2.bar(x + bar_positions_2[-1], method_data['baseline'], width,
               label='Baseline', alpha=0.8, color='gray')

        ax2.set_xlabel('Target Languages')
        ax2.set_ylabel('Performance Score')
        ax2.set_title('Advanced Merging Methods Performance Comparison (Part 2)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(locales, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/advanced_merging_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_ensemble_comparison_plot(self, ensemble_results: List[Dict]):
        """Create ensemble methods comparison"""
        print("Creating ensemble comparison...")

        if not ensemble_results:
            print("No ensemble results found")
            return

        # Group by locale
        locale_data = {}
        for result in ensemble_results:
            locale = result['target_locale']
            if locale not in locale_data:
                locale_data[locale] = {}
            locale_data[locale][result['ensemble_method']] = result['ensemble_accuracy']
            locale_data[locale]['baseline'] = result.get('baseline_accuracy', 0)
            locale_data[locale]['avg_zero_shot'] = result.get('avg_zero_shot', 0)
            locale_data[locale]['best_zero_shot'] = result.get('best_zero_shot', 0)

        locales = list(locale_data.keys())
        methods = ['majority', 'weighted_majority', 'soft', 'uriel_logits']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Plot 1: Ensemble methods vs baseline
        x = np.arange(len(locales))
        width = 0.15

        bars = []
        labels = []

        # Baseline
        baseline_scores = [locale_data[locale].get('baseline', 0) for locale in locales]
        bars.append(ax1.bar(x - 1.5*width, baseline_scores, width, label='Baseline', alpha=0.7, color='gray'))

        # Ensemble methods
        for i, method in enumerate(methods):
            scores = []
            for locale in locales:
                scores.append(locale_data[locale].get(method, 0))
            bars.append(ax1.bar(x + (i - 0.5)*width, scores, width,
                               label=method.replace('_', ' ').title(), alpha=0.8))

        ax1.set_xlabel('Target Languages')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Ensemble Methods Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(locales, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Ensemble advantage over zero-shot
        for i, method in enumerate(methods):
            advantages = []
            for locale in locales:
                ensemble_score = locale_data[locale].get(method, 0)
                avg_zero_shot = locale_data[locale].get('avg_zero_shot', 0)
                advantages.append(ensemble_score - avg_zero_shot)

            bars = ax2.bar(x + (i - 1.5)*width, advantages, width,
                          label=f'{method} vs Zero-shot', alpha=0.8)

            # Color bars: green for positive, red for negative
            for bar in bars:
                if bar.get_height() >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('lightcoral')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Target Languages')
        ax2.set_ylabel('Advantage over Average Zero-shot')
        ax2.set_title('Ensemble Methods Advantage Over Zero-shot Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(locales, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/ensemble_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_comprehensive_summary(self, merging_results: List[Dict], ensemble_results: List[Dict], nvn_df=None, available_locales=None):
        """Create comprehensive summary CSV with all methods vs avg_zero_shot, best_zero_shot, and best_source"""
        print("Creating comprehensive summary...")

        # Create a mapping from locale to all results
        locale_data = {}

        # Process merging results
        for result in merging_results:
            locale = result['target_locale']
            if locale not in locale_data:
                locale_data[locale] = {
                    'locale': locale,
                    'baseline': result.get('baseline', 0),
                    'avg_zero_shot': result.get('avg_zero_shot', 0),
                    'best_zero_shot': result.get('best_zero_shot', 0),
                    'best_source': result.get('best_source', 0),
                    'source_locales': result.get('source_locales', [])
                }

            # Add basic and advanced merging methods
            merging_methods = ['similarity', 'average', 'ties', 'task_arithmetic', 'slerp', 'regmean', 'dare', 'fisher']
            for method in merging_methods:
                if method in result:
                    locale_data[locale][method] = result[method]
                    # Calculate improvements vs all three baselines
                    if locale_data[locale]['avg_zero_shot'] > 0:
                        locale_data[locale][f'{method}_vs_avg_zero'] = result[method] - locale_data[locale]['avg_zero_shot']
                    if locale_data[locale]['best_zero_shot'] > 0:
                        locale_data[locale][f'{method}_vs_best_zero'] = result[method] - locale_data[locale]['best_zero_shot']
                    if locale_data[locale]['best_source'] > 0:
                        locale_data[locale][f'{method}_vs_best_source'] = result[method] - locale_data[locale]['best_source']

        # Process ensemble results and merge into the same locale rows
        for result in ensemble_results:
            locale = result['target_locale']
            method = result['ensemble_method']  # 'majority', 'weighted_majority', 'soft', 'uriel_logits'
            accuracy = result.get('ensemble_accuracy', 0)

            if locale not in locale_data:
                locale_data[locale] = {
                    'locale': locale,
                    'baseline': result.get('baseline_accuracy', 0),
                    'avg_zero_shot': result.get('avg_zero_shot', 0),
                    'best_zero_shot': result.get('best_zero_shot', 0),
                    'best_source': result.get('best_source', 0),
                    'source_locales': result.get('source_locales', [])
                }

            # Add ensemble method
            locale_data[locale][method] = accuracy
            # Calculate improvements vs all three baselines
            if locale_data[locale]['avg_zero_shot'] > 0:
                locale_data[locale][f'{method}_vs_avg_zero'] = accuracy - locale_data[locale]['avg_zero_shot']
            if locale_data[locale]['best_zero_shot'] > 0:
                locale_data[locale][f'{method}_vs_best_zero'] = accuracy - locale_data[locale]['best_zero_shot']
            if locale_data[locale]['best_source'] > 0:
                locale_data[locale][f'{method}_vs_best_source'] = accuracy - locale_data[locale]['best_source']

        summary_df = pd.DataFrame(list(locale_data.values()))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_df.to_csv(f'advanced_analysis_summary_{timestamp}.csv', index=False)

        print(f"Comprehensive summary saved to: advanced_analysis_summary_{timestamp}.csv")

        # Print summary statistics
        self.print_summary_statistics(summary_df)

        return summary_df

    def generate_pure_scores_plots(self, summary_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Generate individual plots showing pure performance scores for each method"""
        print("Generating pure scores plots...")

        # Identify all method columns (pure performance, not comparisons)
        method_cols = [col for col in summary_df.columns if col not in ['locale', 'baseline', 'avg_zero_shot', 'best_zero_shot', 'best_source', 'source_locales'] and not '_vs_' in col]

        if not method_cols:
            print("No method columns found for pure scores plots")
            return

        locales = summary_df['locale'].tolist()

        for method in method_cols:
            # Create individual plot for each method
            fig, ax = plt.subplots(figsize=(20, 8))

            # Get method data
            method_data = summary_df[method].fillna(0).tolist()

            # Also include baseline and all three zero-shot baselines for reference
            baseline_data = summary_df.get('baseline', pd.Series([0]*len(locales))).fillna(0).tolist()
            avg_zero_data = summary_df.get('avg_zero_shot', pd.Series([0]*len(locales))).fillna(0).tolist()
            best_zero_data = summary_df.get('best_zero_shot', pd.Series([0]*len(locales))).fillna(0).tolist()
            best_source_data = summary_df.get('best_source', pd.Series([0]*len(locales))).fillna(0).tolist()

            x = np.arange(len(locales))
            width = 0.16

            # Plot all baselines and method
            bars1 = ax.bar(x - 2*width, baseline_data, width, label='Baseline', alpha=0.7, color='gray')
            bars2 = ax.bar(x - width, avg_zero_data, width, label='Avg Zero-shot', alpha=0.7, color='lightblue')
            bars3 = ax.bar(x, best_zero_data, width, label='Best Zero-shot', alpha=0.7, color='lightgreen')
            bars4 = ax.bar(x + width, best_source_data, width, label='Best Source', alpha=0.7, color='lightcoral')
            bars5 = ax.bar(x + 2*width, method_data, width, label=method.replace('_', ' ').title(), alpha=0.8, color='royalblue')

            # Add value labels on method bars
            for i, bar in enumerate(bars5):
                height = bar.get_height()
                if height > 0.01:  # Only show labels for non-zero values
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_xlabel('Target Languages')
            ax.set_ylabel('Performance Score')
            ax.set_title(f'Pure Performance: {method.replace("_", " ").title()} vs All Baselines')
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / f"pure_scores_{method}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Pure scores plot for {method} saved to: {output_file}")

    def generate_vs_avg_zero_plots(self, summary_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Generate individual plots showing improvement vs average zero-shot baseline for each method"""
        print("Generating vs average zero-shot plots...")

        # Identify all vs_avg_zero columns
        vs_avg_cols = [col for col in summary_df.columns if col.endswith('_vs_avg_zero')]

        if not vs_avg_cols:
            print("No vs_avg_zero columns found for comparison plots")
            return

        locales = summary_df['locale'].tolist()

        for col in vs_avg_cols:
            method_name = col.replace('_vs_avg_zero', '')
            improvement_data = summary_df[col].fillna(0).tolist()

            # Create individual plot for each method
            fig, ax = plt.subplots(figsize=(16, 8))

            x = np.arange(len(locales))
            width = 0.6

            bars = ax.bar(x, improvement_data, width, alpha=0.8)

            # Color bars: green for positive (improvement), red for negative (degradation)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if abs(height) > 0.001:  # Only show significant differences
                    if height >= 0:
                        bar.set_color('green')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'+{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen')
                    else:
                        bar.set_color('lightcoral')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='top', fontsize=8, fontweight='bold', color='darkred')

            # Add reference line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)

            # Add statistics
            improvements = [h for h in improvement_data if abs(h) > 0.001]
            if improvements:
                mean_improvement = np.mean(improvements)
                positive_count = sum(1 for h in improvements if h > 0)
                total_count = len(improvements)
                win_rate = (positive_count / total_count) * 100

                # Add text box with statistics
                stats_text = f'Mean: {mean_improvement:+.4f}\nWin Rate: {win_rate:.1f}%\nCount: {total_count}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Target Languages')
            ax.set_ylabel('Improvement over Average Zero-shot')
            ax.set_title(f'{method_name.replace("_", " ").title()} vs Average Zero-shot Baseline')
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / f"vs_avg_zero_{method_name}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ vs avg zero-shot plot for {method_name} saved to: {output_file}")

    def generate_vs_best_zero_plots(self, summary_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Generate individual plots showing improvement vs best zero-shot baseline for each method"""
        print("Generating vs best zero-shot plots...")

        # Identify all vs_best_zero columns
        vs_best_cols = [col for col in summary_df.columns if col.endswith('_vs_best_zero')]

        if not vs_best_cols:
            print("No vs_best_zero columns found for comparison plots")
            return

        locales = summary_df['locale'].tolist()

        for col in vs_best_cols:
            method_name = col.replace('_vs_best_zero', '')
            improvement_data = summary_df[col].fillna(0).tolist()

            # Create individual plot for each method
            fig, ax = plt.subplots(figsize=(16, 8))

            x = np.arange(len(locales))
            width = 0.6

            bars = ax.bar(x, improvement_data, width, alpha=0.8)

            # Color bars: blue for positive (improvement), orange for negative (degradation)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if abs(height) > 0.001:  # Only show significant differences
                    if height >= 0:
                        bar.set_color('royalblue')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'+{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkblue')
                    else:
                        bar.set_color('orange')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='top', fontsize=8, fontweight='bold', color='darkorange')

            # Add reference line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)

            # Add statistics
            improvements = [h for h in improvement_data if abs(h) > 0.001]
            if improvements:
                mean_improvement = np.mean(improvements)
                positive_count = sum(1 for h in improvements if h > 0)
                total_count = len(improvements)
                win_rate = (positive_count / total_count) * 100

                # Add text box with statistics
                stats_text = f'Mean: {mean_improvement:+.4f}\nWin Rate: {win_rate:.1f}%\nCount: {total_count}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            ax.set_xlabel('Target Languages')
            ax.set_ylabel('Improvement over Best Zero-shot')
            ax.set_title(f'{method_name.replace("_", " ").title()} vs Best Zero-shot Baseline')
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / f"vs_best_zero_{method_name}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ vs best zero-shot plot for {method_name} saved to: {output_file}")

    def generate_vs_best_source_plots(self, summary_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Generate individual plots showing improvement vs best source baseline for each method"""
        print("Generating vs best source plots...")

        # Identify all vs_best_source columns
        vs_best_source_cols = [col for col in summary_df.columns if col.endswith('_vs_best_source')]

        if not vs_best_source_cols:
            print("No vs_best_source columns found for comparison plots")
            return

        locales = summary_df['locale'].tolist()

        for col in vs_best_source_cols:
            method_name = col.replace('_vs_best_source', '')
            improvement_data = summary_df[col].fillna(0).tolist()

            # Create individual plot for each method
            fig, ax = plt.subplots(figsize=(16, 8))

            x = np.arange(len(locales))
            width = 0.6

            bars = ax.bar(x, improvement_data, width, alpha=0.8)

            # Color bars: green for positive (improvement), red for negative (degradation)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if abs(height) > 0.001:  # Only show significant differences
                    if height >= 0:
                        bar.set_color('forestgreen')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'+{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen')
                    else:
                        bar.set_color('indianred')
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='top', fontsize=8, fontweight='bold', color='darkred')

            # Add reference line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)

            # Add statistics
            improvements = [h for h in improvement_data if abs(h) > 0.001]
            if improvements:
                mean_improvement = np.mean(improvements)
                positive_count = sum(1 for h in improvements if h > 0)
                total_count = len(improvements)
                win_rate = (positive_count / total_count) * 100

                # Add text box with statistics
                stats_text = f'Mean: {mean_improvement:+.4f}\nWin Rate: {win_rate:.1f}%\nCount: {total_count}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

            ax.set_xlabel('Target Languages')
            ax.set_ylabel('Improvement over Best Source')
            ax.set_title(f'{method_name.replace("_", " ").title()} vs Best Source Baseline (Fair Comparison)')
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / f"vs_best_source_{method_name}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ vs best source plot for {method_name} saved to: {output_file}")

    def print_summary_statistics(self, summary_df: pd.DataFrame):
        """Print comprehensive summary statistics"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)

        # Basic stats
        print(f"\nTotal locales analyzed: {len(summary_df)}")

        # Basic methods analysis
        print(f"\n--- BASIC METHODS ANALYSIS ---")
        print(f"Average baseline: {summary_df['baseline'].mean():.4f}")
        print(f"Average similarity merge: {summary_df['similarity'].mean():.4f}")
        print(f"Average average merge: {summary_df['average'].mean():.4f}")

        # Advanced merging methods analysis
        print(f"\n--- ADVANCED MERGING METHODS ANALYSIS ---")
        for method in ['ties', 'task_arithmetic', 'slerp', 'regmean', 'dare', 'fisher']:
            if method in summary_df.columns:
                avg_score = summary_df[method].mean()
                print(f"Average {method}: {avg_score:.4f}")

        if 'similarity_vs_avg_zero' in summary_df.columns:
            avg_improvement = summary_df['similarity_vs_avg_zero'].mean()
            print(f"Average similarity improvement vs avg zero-shot: {avg_improvement:.4f}")

        # Ensemble methods analysis
        print(f"\n--- ENSEMBLE METHODS ANALYSIS ---")
        ensemble_methods = ['ensemble_majority', 'ensemble_weighted_majority', 'ensemble_soft', 'ensemble_uriel_logits']
        for method in ensemble_methods:
            if method in summary_df.columns:
                avg_score = summary_df[method].mean()
                method_name = method.replace('ensemble_', '')
                print(f"Average ensemble {method_name}: {avg_score:.4f}")

                improvement_col = f"{method_name}_vs_avg_zero"
                if improvement_col in summary_df.columns:
                    avg_improvement = summary_df[improvement_col].mean()
                    print(f"Average {method_name} improvement vs avg zero-shot: {avg_improvement:.4f}")

        # Zero-shot comparison
        if 'avg_zero_shot' in summary_df.columns:
            print(f"\n--- ZERO-SHOT COMPARISON ---")
            print(f"Average zero-shot performance: {summary_df['avg_zero_shot'].mean():.4f}")
            print(f"Best zero-shot performance: {summary_df['best_zero_shot'].max():.4f}")

    def generate_advanced_analysis(self):
        """Main method to generate complete advanced analysis"""
        print("Starting Advanced Results Analysis...")
        print("="*80)

        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Analyze advanced merging methods
        merging_results = self.analyze_advanced_merging_methods()

        # Analyze ensemble methods
        ensemble_results = self.analyze_ensemble_methods()

        # Create comprehensive summary with all methods vs both baselines
        summary_df = self.create_comprehensive_summary(merging_results, ensemble_results, self.nxn_df)

        if summary_df is not None:
            # Generate individual plots for each method and comparison type
            self.generate_pure_scores_plots(summary_df, plots_dir, timestamp)
            self.generate_vs_avg_zero_plots(summary_df, plots_dir, timestamp)
            self.generate_vs_best_zero_plots(summary_df, plots_dir, timestamp)
            self.generate_vs_best_source_plots(summary_df, plots_dir, timestamp)

        print(f"\nAdvanced analysis complete!")
        print(f"Generated files:")
        print(f"- advanced_analysis_summary_{timestamp}.csv")
        print(f"- Individual pure score plots for each method")
        print(f"- Individual vs avg zero-shot comparison plots for each method")
        print(f"- Individual vs best zero-shot comparison plots for each method")
        print(f"- Individual vs best source comparison plots for each method (FAIR COMPARISON)")
        print(f"All plots saved in plots/ directory with {timestamp} suffix")

        return summary_df

def main():
    """Main function to run the advanced analysis system"""
    try:
        analyzer = AdvancedResultsAnalyzer()
        summary_df = analyzer.generate_advanced_analysis()
        return summary_df
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        return None

if __name__ == "__main__":
    exit(main())