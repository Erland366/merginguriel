#!/usr/bin/env python3
"""
Script to aggregate and compare results from large-scale experiments.
Enhanced version with comprehensive automated evaluation reports including N-vs-N baselines.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import glob
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment extracted from merge_details.txt or folder name."""
    experiment_type: str
    locale: str
    target_lang: Optional[str] = None
    source_languages: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    merge_mode: Optional[str] = None
    num_languages: Optional[int] = None
    timestamp: Optional[str] = None
    folder_name: Optional[str] = None


@dataclass
class BaselineData:
    """Baseline performance data from N-vs-N evaluation."""
    best_source_accuracy: Optional[float] = None
    best_source_language: Optional[str] = None
    best_overall_accuracy: Optional[float] = None
    best_overall_language: Optional[str] = None

def load_results_from_folder(folder_path):
    """Load results.json from a folder."""
    results_file = os.path.join(folder_path, "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {results_file}: {e}")
            return None
    return None


def parse_merge_details(merge_details_path: str) -> Optional[Dict[str, Any]]:
    """Parse merge_details.txt file to extract experiment metadata."""
    if not os.path.exists(merge_details_path):
        return None

    try:
        with open(merge_details_path, 'r') as f:
            content = f.read()

        # Parse key-value pairs from the merge details
        details = {}
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                details[key.strip()] = value.strip()

        return details
    except Exception as e:
        logger.warning(f"Error parsing merge details from {merge_details_path}: {e}")
        return None


def extract_locale_from_model_path(model_path: str) -> Optional[str]:
    """Extract locale from model path like './haryos_model/xlm-roberta-base_massive_k_sq-AL'."""
    # Pattern: xlm-roberta-base_massive_k_{locale}
    match = re.search(r'massive_k_([a-z]{2}-[A-Z]{2})', model_path)
    if match:
        return match.group(1)
    return None


def parse_experiment_metadata(folder_name: str, folder_path: str) -> ExperimentMetadata:
    """Parse experiment metadata from folder name and merge_details.txt if available."""

    # First, try to parse from merge_details.txt
    merge_details_path = os.path.join(folder_path, "merge_details.txt")
    merge_details = parse_merge_details(merge_details_path)

    if merge_details:
        # Extract metadata from merge_details.txt
        experiment_type = merge_details.get('Merge Mode', 'unknown').lower()
        target_lang = merge_details.get('Target Language', '')

        # Extract source languages and weights from the models section
        source_languages = []
        weights = {}

        # Find the "--- Merged Models and Weights ---" section
        content = open(merge_details_path, 'r').read()
        models_section = re.split(r'--- Merged Models and Weights ---', content, flags=re.MULTILINE)
        if len(models_section) > 1:
            models_content = models_section[1]

            # Parse each model entry
            model_pattern = r'\d+\.\s*Model:\s*(.+)\s*-?\s*(?:Subfolder:\s*(.+)\s*)?-?\s*(?:Language:\s*(.+)\s*)?-?\s*(?:Locale:\s*(.+)\s*)?-?\s*Weight:\s*([\d.]+)\s*\(([\d.]+)%\s*of total\)'
            matches = re.findall(model_pattern, models_content, re.MULTILINE)

            for model_path, subfolder, language, locale, weight, weight_percent in matches:
                if locale:
                    source_languages.append(locale)
                    weights[locale] = float(weight)
                else:
                    # Try to extract locale from model path
                    extracted_locale = extract_locale_from_model_path(model_path)
                    if extracted_locale:
                        source_languages.append(extracted_locale)
                        weights[extracted_locale] = float(weight)

        return ExperimentMetadata(
            experiment_type=experiment_type,
            locale=target_lang,
            target_lang=target_lang,
            source_languages=source_languages,
            weights=weights,
            merge_mode=experiment_type,
            num_languages=len(source_languages) if source_languages else None,
            timestamp=merge_details.get('Timestamp (UTC)', ''),
            folder_name=folder_name
        )

    # Fallback: parse from folder name
    parts = folder_name.split('_')

    # Handle different patterns:
    # 1. "baseline_sq-AL" -> type="baseline", locale="sq-AL"
    # 2. "similarity_sq-AL" -> type="similarity", locale="sq-AL"
    # 3. "average_sq-AL" -> type="average", locale="sq-AL"
    # 4. "xlm-roberta-base_massive_k_sq-AL_alpha_0.5_sq-AL_epoch-9_sq-AL" -> type="baseline", locale="sq-AL"

    if parts[0] in ['baseline', 'similarity', 'average', 'fisher_simple', 'fisher_dataset', 'ties', 'dare', 'slerp', 'task_arithmetic', 'regmean', 'breadcrumbs']:
        # Simple case: prefix_locale
        exp_type = parts[0]
        locale = '_'.join(parts[1:])
    elif 'massive_k_' in folder_name:
        # Complex case: baseline with full model name
        exp_type = 'baseline'
        # Find the locale part (usually the last part before the final underscore)
        if len(parts) >= 2:
            locale = parts[-1]  # Last part is usually the locale
        else:
            locale = 'unknown'
    else:
        # Fallback - try to extract experiment type from other patterns
        exp_type = 'unknown'
        locale = folder_name

    return ExperimentMetadata(
        experiment_type=exp_type,
        locale=locale,
        target_lang=locale,
        source_languages=None,
        weights=None,
        merge_mode=exp_type,
        num_languages=None,
        timestamp=None,
        folder_name=folder_name
    )

def extract_accuracy(results):
    """Extract accuracy from results."""
    if results and 'performance' in results:
        return results['performance']['accuracy']
    return None

def get_experiment_folders():
    """Get all experiment folders from the results directory."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return []

    folders = []
    for item in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, item)
        if os.path.isdir(folder_path):
            folders.append(item)

    return sorted(folders)


def find_evaluation_matrix(nxn_results_dir: str = "nxn_results") -> Optional[str]:
    """Find the most recent evaluation_matrix.csv file in nxn_results directories."""
    if not os.path.exists(nxn_results_dir):
        logger.warning(f"N-x-N results directory not found: {nxn_results_dir}")
        return None

    # Look for evaluation_matrix.csv files in subdirectories
    pattern = os.path.join(nxn_results_dir, "*", "evaluation_matrix.csv")
    matrix_files = glob.glob(pattern)

    if not matrix_files:
        logger.warning("No evaluation_matrix.csv files found in N-x-N results directories")
        return None

    # Return the most recent one based on directory modification time
    latest_file = max(matrix_files, key=os.path.getmtime)
    logger.info(f"Using evaluation matrix: {latest_file}")
    return latest_file


def load_evaluation_matrix(matrix_path: str) -> Optional[pd.DataFrame]:
    """Load the evaluation matrix from CSV file."""
    try:
        df = pd.read_csv(matrix_path, index_col=0)
        logger.info(f"Loaded evaluation matrix with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading evaluation matrix from {matrix_path}: {e}")
        return None


def get_baseline_for_target(target_locale: str, source_locales: List[str], evaluation_matrix: pd.DataFrame) -> BaselineData:
    """Calculate baseline data for a target locale given source locales."""
    baseline = BaselineData()

    if target_locale not in evaluation_matrix.index:
        logger.warning(f"Target locale {target_locale} not found in evaluation matrix")
        return baseline

    # Get performance of source languages on target locale
    source_performances = {}
    for source_locale in source_locales:
        if source_locale in evaluation_matrix.index and target_locale in evaluation_matrix.columns:
            accuracy = evaluation_matrix.loc[source_locale, target_locale]
            if pd.notna(accuracy):
                source_performances[source_locale] = accuracy

    # Best source language baseline
    if source_performances:
        best_source_locale = max(source_performances, key=source_performances.get)
        baseline.best_source_accuracy = source_performances[best_source_locale]
        baseline.best_source_language = best_source_locale

    # Best overall zero-shot baseline (excluding target locale itself)
    target_column = evaluation_matrix[target_locale] if target_locale in evaluation_matrix.columns else None
    if target_column is not None:
        # Exclude target locale's performance on itself (diagonal)
        overall_performances = target_column.drop(target_locale, errors='ignore')
        overall_performances = overall_performances[overall_performances.notna()]

        if len(overall_performances) > 0:
            best_overall_locale = overall_performances.idxmax()
            baseline.best_overall_accuracy = overall_performances[best_overall_locale]
            baseline.best_overall_language = best_overall_locale

    return baseline

def parse_folder_name(folder_name):
    """Parse folder name to extract experiment type and locale."""
    parts = folder_name.split('_')
    
    # Handle different patterns:
    # 1. "baseline_sq-AL" -> type="baseline", locale="sq-AL"
    # 2. "similarity_sq-AL" -> type="similarity", locale="sq-AL"
    # 3. "average_sq-AL" -> type="average", locale="sq-AL"
    # 4. "xlm-roberta-base_massive_k_sq-AL_alpha_0.5_sq-AL_epoch-9_sq-AL" -> type="baseline", locale="sq-AL"
    
    if parts[0] in ['baseline', 'similarity', 'average']:
        # Simple case: prefix_locale
        exp_type = parts[0]
        locale = '_'.join(parts[1:])
    elif 'massive_k_' in folder_name:
        # Complex case: baseline with full model name
        exp_type = 'baseline'
        # Find the locale part (usually the last part before the final underscore)
        if len(parts) >= 2:
            locale = parts[-1]  # Last part is usually the locale
        else:
            locale = 'unknown'
    else:
        # Fallback
        exp_type = 'unknown'
        locale = folder_name
    
    return exp_type, locale

def aggregate_results(evaluation_matrix: Optional[pd.DataFrame] = None):
    """Aggregate results from all experiment folders with dynamic parsing."""
    folders = get_experiment_folders()

    data = []

    for folder in folders:
        folder_path = os.path.join("results", folder)
        results = load_results_from_folder(folder_path)

        if results:
            # Use new dynamic metadata parsing
            metadata = parse_experiment_metadata(folder, folder_path)
            accuracy = extract_accuracy(results)

            # Extract additional info
            model_info = results.get('evaluation_info', {})
            perf_info = results.get('performance', {})

            # Calculate baseline data if evaluation matrix is available
            baseline_data = None
            if evaluation_matrix is not None and metadata.source_languages:
                baseline_data = get_baseline_for_target(
                    metadata.locale,
                    metadata.source_languages,
                    evaluation_matrix
                )

            # Build the data row
            data_row = {
                'locale': metadata.locale,
                'experiment_type': metadata.experiment_type,
                'folder_name': folder,
                'accuracy': accuracy,
                'correct_predictions': perf_info.get('correct_predictions'),
                'total_predictions': perf_info.get('total_predictions'),
                'error_rate': perf_info.get('error_rate'),
                'model_name': model_info.get('model_name'),
                'subfolder': model_info.get('subfolder'),
                'timestamp': model_info.get('timestamp'),
                # New metadata fields
                'target_lang': metadata.target_lang,
                'source_languages': metadata.source_languages,
                'weights': metadata.weights,
                'merge_mode': metadata.merge_mode,
                'num_languages': metadata.num_languages,
                'merge_timestamp': metadata.timestamp
            }

            # Add baseline data if available
            if baseline_data:
                data_row.update({
                    'best_source_accuracy': baseline_data.best_source_accuracy,
                    'best_source_language': baseline_data.best_source_language,
                    'best_overall_accuracy': baseline_data.best_overall_accuracy,
                    'best_overall_language': baseline_data.best_overall_language
                })

            data.append(data_row)

    return pd.DataFrame(data)

def create_comparison_table(df):
    """Create a dynamic comparison table with all experiment types and baseline data."""
    # Get unique experiment types dynamically
    experiment_types = sorted(df['experiment_type'].unique())

    # Pivot the data to have one row per locale
    pivot_df = df.pivot_table(
        index='locale',
        columns='experiment_type',
        values='accuracy',
        aggfunc='first'
    ).reset_index()

    # Flatten column names
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename_axis(None, axis=1)

    # Ensure we have baseline if it exists, and include all discovered experiment types
    baseline_columns = ['baseline']
    discovered_columns = [col for col in experiment_types if col not in baseline_columns]

    # Create final column order: locale, baseline, other experiment types, baseline data, improvements
    final_columns = ['locale'] + baseline_columns + discovered_columns

    # Add baseline data columns if they exist
    baseline_data_columns = ['best_source_accuracy', 'best_overall_accuracy']
    for col in baseline_data_columns:
        if col in df.columns:
            final_columns.append(col)

    # Ensure all columns exist
    for col in final_columns:
        if col not in pivot_df.columns and col != 'locale':
            pivot_df[col] = None

    # Reorder columns
    pivot_df = pivot_df[final_columns]

    # Calculate improvements against baseline
    if 'baseline' in pivot_df.columns:
        for exp_type in discovered_columns:
            if exp_type in pivot_df.columns:
                improvement_col = f'{exp_type}_improvement'
                pivot_df[improvement_col] = pivot_df.apply(
                    lambda row: row[exp_type] - row['baseline']
                    if row['baseline'] is not None and row[exp_type] is not None
                    else None, axis=1
                )

        # Calculate improvements against best source baseline
        if 'best_source_accuracy' in pivot_df.columns:
            for exp_type in discovered_columns:
                if exp_type in pivot_df.columns:
                    improvement_col = f'{exp_type}_vs_best_source'
                    pivot_df[improvement_col] = pivot_df.apply(
                        lambda row: row[exp_type] - row['best_source_accuracy']
                        if row['best_source_accuracy'] is not None and row[exp_type] is not None
                        else None, axis=1
                    )

        # Calculate improvements against best overall baseline
        if 'best_overall_accuracy' in pivot_df.columns:
            for exp_type in discovered_columns:
                if exp_type in pivot_df.columns:
                    improvement_col = f'{exp_type}_vs_best_overall'
                    pivot_df[improvement_col] = pivot_df.apply(
                        lambda row: row[exp_type] - row['best_overall_accuracy']
                        if row['best_overall_accuracy'] is not None and row[exp_type] is not None
                        else None, axis=1
                    )

    return pivot_df

def generate_summary_stats(df):
    """Generate dynamic summary statistics for all experiment types."""
    summary = {}

    # Get unique experiment types dynamically
    experiment_types = sorted(df['experiment_type'].unique())

    # Overall statistics by experiment type
    for exp_type in experiment_types:
        if exp_type in df['experiment_type'].values:
            type_df = df[df['experiment_type'] == exp_type]
            valid_acc = type_df[type_df['accuracy'].notna()]

            if len(valid_acc) > 0:
                summary[exp_type] = {
                    'count': len(valid_acc),
                    'mean_accuracy': valid_acc['accuracy'].mean(),
                    'std_accuracy': valid_acc['accuracy'].std(),
                    'min_accuracy': valid_acc['accuracy'].min(),
                    'max_accuracy': valid_acc['accuracy'].max()
                }

    # Add baseline statistics if available
    baseline_types = ['best_source_accuracy', 'best_overall_accuracy']
    for baseline_type in baseline_types:
        if baseline_type in df.columns:
            valid_baseline = df[df[baseline_type].notna()]
            if len(valid_baseline) > 0:
                summary[f'baseline_{baseline_type}'] = {
                    'count': len(valid_baseline),
                    'mean_accuracy': valid_baseline[baseline_type].mean(),
                    'std_accuracy': valid_baseline[baseline_type].std(),
                    'min_accuracy': valid_baseline[baseline_type].min(),
                    'max_accuracy': valid_baseline[baseline_type].max()
                }

    return summary


def generate_win_rate_analysis(df, comparison_df):
    """Generate win rate analysis comparing merging methods against baselines."""
    win_analysis = {}

    # Get experiment types (excluding baseline)
    experiment_types = [col for col in comparison_df.columns if col not in
                       ['locale', 'baseline', 'best_source_accuracy', 'best_overall_accuracy']]

    for exp_type in experiment_types:
        if exp_type not in comparison_df.columns:
            continue

        win_stats = {
            'vs_baseline_wins': 0,
            'vs_baseline_total': 0,
            'vs_best_source_wins': 0,
            'vs_best_source_total': 0,
            'vs_best_overall_wins': 0,
            'vs_best_overall_total': 0,
            'avg_improvement_vs_baseline': 0.0,
            'avg_improvement_vs_best_source': 0.0,
            'avg_improvement_vs_best_overall': 0.0
        }

        improvements_vs_baseline = []
        improvements_vs_best_source = []
        improvements_vs_best_overall = []

        for _, row in comparison_df.iterrows():
            if row[exp_type] is not None:
                # Compare against baseline
                if row.get('baseline') is not None:
                    win_stats['vs_baseline_total'] += 1
                    improvement = row[exp_type] - row['baseline']
                    improvements_vs_baseline.append(improvement)
                    if improvement > 0:
                        win_stats['vs_baseline_wins'] += 1

                # Compare against best source
                if row.get('best_source_accuracy') is not None:
                    win_stats['vs_best_source_total'] += 1
                    improvement = row[exp_type] - row['best_source_accuracy']
                    improvements_vs_best_source.append(improvement)
                    if improvement > 0:
                        win_stats['vs_best_source_wins'] += 1

                # Compare against best overall
                if row.get('best_overall_accuracy') is not None:
                    win_stats['vs_best_overall_total'] += 1
                    improvement = row[exp_type] - row['best_overall_accuracy']
                    improvements_vs_best_overall.append(improvement)
                    if improvement > 0:
                        win_stats['vs_best_overall_wins'] += 1

        # Calculate average improvements
        if improvements_vs_baseline:
            win_stats['avg_improvement_vs_baseline'] = sum(improvements_vs_baseline) / len(improvements_vs_baseline)
        if improvements_vs_best_source:
            win_stats['avg_improvement_vs_best_source'] = sum(improvements_vs_best_source) / len(improvements_vs_best_source)
        if improvements_vs_best_overall:
            win_stats['avg_improvement_vs_best_overall'] = sum(improvements_vs_best_overall) / len(improvements_vs_best_overall)

        win_analysis[exp_type] = win_stats

    return win_analysis

def save_results(df, comparison_df, summary, win_analysis=None):
    """Save all results to files with comprehensive exports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw data
    raw_filename = f'results_aggregated_{timestamp}.csv'
    df.to_csv(raw_filename, index=False)
    logger.info(f"Raw results saved to {raw_filename}")

    # Save comparison table
    comparison_filename = f'results_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_filename, index=False)
    logger.info(f"Comparison table saved to {comparison_filename}")

    # Save comprehensive results with all metrics in one file
    comprehensive_filename = f'results_comprehensive_{timestamp}.csv'
    comparison_df.to_csv(comprehensive_filename, index=False)
    logger.info(f"Comprehensive results saved to {comprehensive_filename}")

    # Save summary statistics
    summary_filename = f'results_summary_{timestamp}.json'
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary statistics saved to {summary_filename}")

    # Save win rate analysis if available
    if win_analysis:
        win_analysis_filename = f'results_win_analysis_{timestamp}.json'
        with open(win_analysis_filename, 'w') as f:
            json.dump(win_analysis, f, indent=2)
        logger.info(f"Win rate analysis saved to {win_analysis_filename}")

    # Save comprehensive markdown report
    markdown_content = create_comprehensive_markdown_report(comparison_df, summary, win_analysis)
    markdown_filename = f'results_report_{timestamp}.md'
    with open(markdown_filename, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Comprehensive report saved to {markdown_filename}")

    return {
        'raw_filename': raw_filename,
        'comparison_filename': comparison_filename,
        'comprehensive_filename': comprehensive_filename,
        'summary_filename': summary_filename,
        'markdown_filename': markdown_filename,
        'win_analysis_filename': win_analysis_filename if win_analysis else None
    }

def create_comprehensive_markdown_report(comparison_df, summary, win_analysis=None):
    """Create a comprehensive markdown report with baseline comparisons and analysis."""
    report = "# Comprehensive Experiment Results Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Executive Summary
    report += "## Executive Summary\n\n"
    report += f"This report presents the results of large-scale language model merging experiments.\n"
    report += f"The analysis includes **{len(comparison_df)}** locales and **{len(summary)}** experiment types.\n\n"

    # Summary Statistics
    report += "## Summary Statistics\n\n"
    for exp_type, stats in summary.items():
        if exp_type.startswith('baseline_'):
            continue  # Handle baseline statistics separately

        report += f"### {exp_type.upper()}\n"
        report += f"- Number of experiments: {stats['count']}\n"
        report += f"- Mean accuracy: {stats['mean_accuracy']:.4f}\n"
        report += f"- Standard deviation: {stats['std_accuracy']:.4f}\n"
        report += f"- Min accuracy: {stats['min_accuracy']:.4f}\n"
        report += f"- Max accuracy: {stats['max_accuracy']:.4f}\n\n"

    # Baseline Statistics
    baseline_summary_exists = any(key.startswith('baseline_') for key in summary.keys())
    if baseline_summary_exists:
        report += "### BASELINE PERFORMANCE\n\n"
        for key in ['baseline_best_source_accuracy', 'baseline_best_overall_accuracy']:
            if key in summary:
                stats = summary[key]
                baseline_name = "Best Source Language" if "best_source" in key else "Best Overall Zero-Shot"
                report += f"**{baseline_name}**\n"
                report += f"- Number of comparisons: {stats['count']}\n"
                report += f"- Mean accuracy: {stats['mean_accuracy']:.4f}\n"
                report += f"- Standard deviation: {stats['std_accuracy']:.4f}\n"
                report += f"- Min accuracy: {stats['min_accuracy']:.4f}\n"
                report += f"- Max accuracy: {stats['max_accuracy']:.4f}\n\n"

    # Win Rate Analysis
    if win_analysis:
        report += "## Win Rate Analysis\n\n"
        report += "This section shows how often each merging method outperforms the baselines.\n\n"

        for exp_type, stats in win_analysis.items():
            report += f"### {exp_type.upper()}\n"

            if stats['vs_baseline_total'] > 0:
                win_rate_vs_baseline = (stats['vs_baseline_wins'] / stats['vs_baseline_total']) * 100
                report += f"- **vs Native Baseline**: {stats['vs_baseline_wins']}/{stats['vs_baseline_total']} wins ({win_rate_vs_baseline:.1f}%)\n"
                report += f"  - Average improvement: {stats['avg_improvement_vs_baseline']:+.4f}\n"

            if stats['vs_best_source_total'] > 0:
                win_rate_vs_source = (stats['vs_best_source_wins'] / stats['vs_best_source_total']) * 100
                report += f"- **vs Best Source Language**: {stats['vs_best_source_wins']}/{stats['vs_best_source_total']} wins ({win_rate_vs_source:.1f}%)\n"
                report += f"  - Average improvement: {stats['avg_improvement_vs_best_source']:+.4f}\n"

            if stats['vs_best_overall_total'] > 0:
                win_rate_vs_overall = (stats['vs_best_overall_wins'] / stats['vs_best_overall_total']) * 100
                report += f"- **vs Best Overall Zero-Shot**: {stats['vs_best_overall_wins']}/{stats['vs_best_overall_total']} wins ({win_rate_vs_overall:.1f}%)\n"
                report += f"  - Average improvement: {stats['avg_improvement_vs_best_overall']:+.4f}\n"

            report += "\n"

    # Dynamic Detailed Comparison Table
    report += "## Detailed Comparison Results\n\n"

    # Build table header dynamically
    headers = ["Locale"]
    if 'baseline' in comparison_df.columns:
        headers.append("Baseline")

    # Add all experiment types (excluding baseline and baseline data columns)
    experiment_columns = [col for col in comparison_df.columns if col not in
                         ['locale', 'baseline', 'best_source_accuracy', 'best_overall_accuracy'] and
                         not col.endswith('_improvement') and not col.endswith('_vs_')]
    headers.extend(experiment_columns)

    # Add baseline data if available
    if 'best_source_accuracy' in comparison_df.columns:
        headers.append("Best Source")
    if 'best_overall_accuracy' in comparison_df.columns:
        headers.append("Best Overall")

    # Add improvement columns
    for exp_col in experiment_columns:
        if f'{exp_col}_improvement' in comparison_df.columns:
            headers.append(f"{exp_col} vs Base")
        if f'{exp_col}_vs_best_source' in comparison_df.columns:
            headers.append(f"{exp_col} vs Src")

    # Create table header
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join(["-" * len(h) for h in headers]) + "|"
    report += header_row + "\n"
    report += separator_row + "\n"

    # Create table rows
    for _, row in comparison_df.iterrows():
        row_values = [row['locale']]

        # Baseline
        if 'baseline' in comparison_df.columns:
            baseline_val = f"{row['baseline']:.4f}" if row['baseline'] is not None else "N/A"
            row_values.append(baseline_val)

        # Experiment results
        for exp_col in experiment_columns:
            exp_val = f"{row[exp_col]:.4f}" if row[exp_col] is not None else "N/A"
            row_values.append(exp_val)

        # Baseline data
        if 'best_source_accuracy' in comparison_df.columns:
            source_val = f"{row['best_source_accuracy']:.4f}" if row['best_source_accuracy'] is not None else "N/A"
            row_values.append(source_val)

        if 'best_overall_accuracy' in comparison_df.columns:
            overall_val = f"{row['best_overall_accuracy']:.4f}" if row['best_overall_accuracy'] is not None else "N/A"
            row_values.append(overall_val)

        # Improvements
        for exp_col in experiment_columns:
            if f'{exp_col}_improvement' in comparison_df.columns:
                imp_val = row[f'{exp_col}_improvement']
                imp_str = f"{imp_val:+.4f}" if imp_val is not None else "N/A"
                row_values.append(imp_str)

            if f'{exp_col}_vs_best_source' in comparison_df.columns:
                imp_val = row[f'{exp_col}_vs_best_source']
                imp_str = f"{imp_val:+.4f}" if imp_val is not None else "N/A"
                row_values.append(imp_str)

        report += "| " + " | ".join(row_values) + " |\n"

    # Methodology
    report += "\n## Methodology\n\n"
    report += "### Experimental Setup\n"
    report += "- Each target language was evaluated using multiple merging approaches\n"
    report += "- Results are compared against native performance and cross-lingual baselines\n"
    report += "- Best Source Language: Best performing single source model for the target\n"
    report += "- Best Overall Zero-Shot: Best performing any language model (excluding target)\n\n"

    report += "### Performance Metrics\n"
    report += "- **Accuracy**: Primary evaluation metric on MASSIVE intent classification\n"
    report += "- **Improvement**: Difference between merged model and baseline performance\n"
    report += "- **Win Rate**: Percentage of cases where merging method outperforms baseline\n\n"

    return report

def main():
    parser = argparse.ArgumentParser(description="Aggregate and compare experiment results with comprehensive baseline analysis")
    parser.add_argument("--output-prefix", type=str, default=None,
                       help="Custom prefix for output files")
    parser.add_argument("--format", choices=['csv', 'json', 'markdown', 'all'], default='all',
                       help="Output format (default: all)")
    parser.add_argument("--show-missing", action="store_true",
                       help="Show missing/failed experiments")
    parser.add_argument("--nxn-results-dir", type=str, default="nxn_results",
                       help="Directory containing N-x-N evaluation results (default: nxn_results)")
    parser.add_argument("--evaluation-matrix", type=str, default=None,
                       help="Path to specific evaluation_matrix.csv file")
    parser.add_argument("--no-baselines", action="store_true",
                       help="Skip baseline integration (for faster processing)")
    parser.add_argument("--locales", nargs="+", default=None,
                       help="Filter results to specific locales only")
    parser.add_argument("--experiment-types", nargs="+", default=None,
                       help="Filter results to specific experiment types only")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting comprehensive results aggregation...")

    # Load evaluation matrix for baseline comparison
    evaluation_matrix = None
    if not args.no_baselines:
        if args.evaluation_matrix:
            matrix_path = args.evaluation_matrix
            logger.info(f"Using specified evaluation matrix: {matrix_path}")
        else:
            matrix_path = find_evaluation_matrix(args.nxn_results_dir)

        if matrix_path:
            evaluation_matrix = load_evaluation_matrix(matrix_path)
            if evaluation_matrix is not None:
                logger.info(f"Loaded evaluation matrix with {evaluation_matrix.shape[0]} locales")
            else:
                logger.warning("Failed to load evaluation matrix, proceeding without baseline integration")
        else:
            logger.info("No evaluation matrix found, proceeding without baseline integration")

    # Load and aggregate results
    logger.info("Aggregating experiment results...")
    df = aggregate_results(evaluation_matrix)

    if len(df) == 0:
        logger.error("No results found in the results directory.")
        return

    # Apply filters if specified
    if args.locales:
        original_count = len(df)
        df = df[df['locale'].isin(args.locales)]
        logger.info(f"Filtered to {len(df)} results for locales: {args.locales}")

    if args.experiment_types:
        original_count = len(df)
        df = df[df['experiment_type'].isin(args.experiment_types)]
        logger.info(f"Filtered to {len(df)} results for experiment types: {args.experiment_types}")

    logger.info(f"Processing {len(df)} experiment results")

    # Create comparison table
    logger.info("Creating comparison table...")
    comparison_df = create_comparison_table(df)

    # Generate summary statistics
    logger.info("Generating summary statistics...")
    summary = generate_summary_stats(df)

    # Generate win rate analysis
    logger.info("Analyzing win rates...")
    win_analysis = generate_win_rate_analysis(df, comparison_df)

    # Show missing experiments
    if args.show_missing:
        logger.info("Missing/failed experiments analysis:")
        experiment_types = sorted(df['experiment_type'].unique())
        for exp_type in experiment_types:
            if exp_type in comparison_df.columns:
                missing = comparison_df[comparison_df[exp_type].isna()]
                if len(missing) > 0:
                    logger.info(f"  {exp_type}: {len(missing)} locales missing")
                    for locale in missing['locale']:
                        logger.info(f"    - {locale}")

    # Print summary statistics
    logger.info("Summary Statistics:")
    for exp_type, stats in summary.items():
        logger.info(f"  {exp_type}: {stats['count']} experiments, "
                   f"mean accuracy = {stats['mean_accuracy']:.4f}")

    # Print win rate summary
    if win_analysis:
        logger.info("Win Rate Analysis Summary:")
        for exp_type, stats in win_analysis.items():
            if stats['vs_baseline_total'] > 0:
                win_rate = (stats['vs_baseline_wins'] / stats['vs_baseline_total']) * 100
                logger.info(f"  {exp_type}: {win_rate:.1f}% win rate vs baseline "
                           f"({stats['vs_baseline_wins']}/{stats['vs_baseline_total']})")

    # Find best performing experiments
    experiment_columns = [col for col in comparison_df.columns if col not in
                         ['locale', 'baseline', 'best_source_accuracy', 'best_overall_accuracy'] and
                         not col.endswith('_improvement') and not col.endswith('_vs_')]

    for exp_type in experiment_columns:
        if exp_type in comparison_df.columns and not comparison_df[exp_type].isna().all():
            best_idx = comparison_df[exp_type].idxmax()
            best_result = comparison_df.loc[best_idx]
            logger.info(f"Best {exp_type} result: {best_result['locale']} "
                       f"({best_result[exp_type]:.4f})")

    # Save results
    logger.info("Saving results...")
    saved_files = save_results(df, comparison_df, summary, win_analysis)

    # Print file summary
    logger.info("Files generated:")
    for file_type, filename in saved_files.items():
        if filename:
            logger.info(f"  {file_type}: {filename}")

    logger.info("Comprehensive aggregation completed successfully!")

if __name__ == "__main__":
    main()