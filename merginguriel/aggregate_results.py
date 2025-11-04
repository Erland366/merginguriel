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
from merginguriel import logger
import glob
import re

# Import centralized naming system
from merginguriel.naming_config import naming_manager

# Loguru logger imported from merginguriel package


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment extracted from merge_details.txt or folder name."""
    experiment_type: str
    locale: str
    target_lang: Optional[str] = None
    source_languages: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    merge_mode: Optional[str] = None
    similarity_type: Optional[str] = None
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


def is_merge_model_path(model_path: Optional[str]) -> bool:
    """Return True if the given model path corresponds to a merged model."""
    return bool(model_path and "merged_models" in model_path)


def parse_num_languages_from_model_path(model_path: Optional[str]) -> Optional[int]:
    """Extract the merged language count from a merged model path if encoded."""
    if not model_path:
        return None

    base_name = os.path.basename(os.path.normpath(model_path))
    match = re.search(r"_(\d+)merged$", base_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.debug(f"Failed to parse num_languages from {base_name}")
    return None


def count_languages_in_merge_details(model_path: Optional[str]) -> Optional[int]:
    """Count languages listed in merge_details.txt for a merged model."""
    if not model_path:
        return None

    merge_details_path = os.path.join(model_path, "merge_details.txt")
    if not os.path.exists(merge_details_path):
        return None

    try:
        with open(merge_details_path, "r") as f:
            content = f.read()
        matches = re.findall(r"^\s*\d+\.\s*Model:", content, re.MULTILINE)
        if matches:
            return len(matches)
    except Exception as e:
        logger.warning(f"Error counting languages in {merge_details_path}: {e}")
    return None


def determine_experiment_variant(experiment_type: str,
                                 num_languages: Optional[int],
                                 model_path: Optional[str],
                                 similarity_type: Optional[str] = None,
                                 base_model: Optional[str] = None) -> str:
    """Build a display key that differentiates merges by language count, similarity type, and base model."""
    base_type = experiment_type or "unknown"

    if not is_merge_model_path(model_path):
        # For baseline models, include full base model info
        if base_model:
            # Use full model name (xlm-roberta-base, xlm-roberta-large)
            return f"{base_type}_{base_model}"
        return base_type

    # Use full base model name for merging experiments
    model_name = base_model if base_model else "unknown_model"

    # Include similarity type and full base model for non-baseline experiments
    if similarity_type and similarity_type in ['URIEL', 'REAL'] and base_type != 'baseline':
        if num_languages:
            return f"{base_type}_{similarity_type}_{model_name}_{int(num_languages)}lang"
        else:
            return f"{base_type}_{similarity_type}_{model_name}_unknownlang"
    else:
        if num_languages:
            return f"{base_type}_{model_name}_{int(num_languages)}lang"
        else:
            return f"{base_type}_{model_name}_unknownlang"

def load_results_from_folder(folder_path):
    """Load results.json from a folder."""
    results_file = os.path.join(folder_path, "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
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
        with open(merge_details_path, "r") as f:
            content = f.read()

        # Parse key-value pairs from the merge details
        details = {}
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(":", 1)
                details[key.strip()] = value.strip()

        return details
    except Exception as e:
        logger.warning(f"Error parsing merge details from {merge_details_path}: {e}")
        return None


def extract_locale_from_model_path(model_path: str) -> Optional[str]:
    """Extract locale from model path like './haryos_model/xlm-roberta-base_massive_k_sq-AL'."""
    # Pattern: xlm-roberta-base_massive_k_{locale}
    match = re.search(r"massive_k_([a-z]{2}-[A-Z]{2})", model_path)
    if match:
        return match.group(1)
    return None


def parse_experiment_metadata(folder_name: str,
                              folder_path: str,
                              model_path: Optional[str] = None) -> ExperimentMetadata:
    """Parse experiment metadata using merge_details if present, otherwise use centralized naming system."""

    # Prefer merge_details that live alongside the actual merged model directory
    merge_details_path = None
    if model_path and os.path.isdir(model_path):
        merge_details_path = os.path.join(model_path, "merge_details.txt")
    else:
        merge_details_path = os.path.join(folder_path, "merge_details.txt")

    merge_details = None
    details_content = None
    if merge_details_path and os.path.exists(merge_details_path):
        merge_details = parse_merge_details(merge_details_path)
        try:
            with open(merge_details_path, "r") as f:
                details_content = f.read()
        except Exception as e:
            logger.warning(f"Unable to read merge details content from {merge_details_path}: {e}")
            details_content = None

    if merge_details and details_content:
        experiment_type = merge_details.get("Merge Mode", "unknown").lower()
        target_lang = merge_details.get("Target Language", "") or merge_details.get("Locale", "")

        # Extract source locales and weights
        locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
        weight_pattern = re.compile(
            r'^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2}).*?Weight:\s*([0-9.]+)',
            re.MULTILINE | re.DOTALL
        )

        source_languages = locale_pattern.findall(details_content)
        weights = {locale: float(weight) for locale, weight in weight_pattern.findall(details_content)}
        num_languages = len(source_languages) if source_languages else None

        return ExperimentMetadata(
            experiment_type=experiment_type,
            locale=target_lang,
            target_lang=target_lang,
            source_languages=source_languages or None,
            weights=weights or None,
            merge_mode=experiment_type,
            similarity_type=None,  # Not available in merge_details, will be set later
            num_languages=num_languages,
            timestamp=merge_details.get('Timestamp (UTC)', ''),
            folder_name=folder_name
        )

    # Try to parse directory name using centralized naming system
    # Only use this if we don't have merge_details (which has more accurate info)
    if not merge_details:
        try:
            # First try to parse as a results directory
            parsed = naming_manager.parse_results_dir_name(folder_name)

            return ExperimentMetadata(
                experiment_type=parsed['experiment_type'],
                locale=parsed['locale'],
                target_lang=parsed['locale'],
                source_languages=None,
                weights=None,
                merge_mode=parsed['method'],
                similarity_type=parsed.get('similarity_type', 'URIEL'),
                num_languages=parsed['num_languages'],
                timestamp=parsed.get('timestamp'),
                folder_name=folder_name
            )
        except ValueError:
            # Try to parse as a merged model directory
            try:
                parsed = naming_manager.parse_merged_model_dir_name(folder_name)

                return ExperimentMetadata(
                    experiment_type=parsed['experiment_type'],
                    locale=parsed['locale'],
                    target_lang=parsed['locale'],
                    source_languages=None,
                    weights=None,
                    merge_mode=parsed['method'],
                    similarity_type=parsed.get('similarity_type', 'URIEL'),
                    num_languages=parsed['num_languages'],
                    timestamp=parsed.get('timestamp'),
                    folder_name=folder_name
                )
            except ValueError:
                # Fallback heuristics for legacy directories
                pass

    # Fallback heuristics using model path or folder name
    experiment_type = 'baseline' if model_path and not is_merge_model_path(model_path) else 'unknown'
    locale = None
    num_languages = None

    if model_path:
        base_name = os.path.basename(os.path.normpath(model_path))
        # Handle new naming convention: {method}_{similarity_type}_merge_{locale}_{num}merged
        # And legacy convention: {method}_merge_{locale}_{num}merged
        merge_match = re.match(r'([a-zA-Z0-9]+)_(?:URIEL|REAL)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?', base_name)
        if not merge_match:
            # Fallback to legacy naming convention
            merge_match = re.match(r'([a-zA-Z0-9]+)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?', base_name)
        if merge_match:
            experiment_type = merge_match.group(1)
            locale = merge_match.group(2)
            if merge_match.group(3):
                try:
                    num_languages = int(merge_match.group(3))
                except ValueError:
                    num_languages = None
        else:
            extracted_locale = extract_locale_from_model_path(model_path)
            if extracted_locale:
                locale = extracted_locale

    if not locale:
        # Handle naming like method_5lang_locale or method_locale
        # Also handle new naming: method_URIEL_5lang_locale or method_REAL_5lang_locale
        # And new merge naming: method_URIEL_merge_locale_num or method_REAL_merge_locale_num
        num_lang_match = re.match(r'([a-zA-Z0-9]+)_(?:URIEL|REAL)_(\d+)lang_(.+)', folder_name)
        if not num_lang_match:
            num_lang_match = re.match(r'([a-zA-Z0-9]+)_(?:URIEL|REAL)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?', folder_name)
        if not num_lang_match:
            # Fallback to legacy naming
            num_lang_match = re.match(r'(.+?)_(\d+)lang_(.+)', folder_name)
        if not num_lang_match:
            num_lang_match = re.match(r'([a-zA-Z0-9]+)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?', folder_name)

        if num_lang_match:
            experiment_type = num_lang_match.group(1)
            # Check if this is a merge pattern or lang pattern
            if 'merge' in folder_name:
                locale = num_lang_match.group(2)
                if num_lang_match.group(3):
                    try:
                        num_languages = int(num_lang_match.group(3))
                    except ValueError:
                        num_languages = None
            else:
                try:
                    num_languages = int(num_lang_match.group(2))
                except ValueError:
                    num_languages = None
                locale = num_lang_match.group(3)
        elif '_' in folder_name:
            parts = folder_name.split('_')
            if len(parts[-1]) == 5 and '-' in parts[-1]:
                locale = parts[-1]
                # Remove similarity type from experiment type if present
                experiment_type_parts = [p for p in parts[:-1] if p not in ['URIEL', 'REAL']]
                experiment_type = '_'.join(experiment_type_parts) if len(experiment_type_parts) > 0 else folder_name
            else:
                # Fallback to last token as locale even if it does not exactly match xx-YY
                locale = parts[-1]
                # Remove similarity type from experiment type if present
                experiment_type_parts = [p for p in parts[:-1] if p not in ['URIEL', 'REAL']]
                experiment_type = '_'.join(experiment_type_parts) if len(experiment_type_parts) > 0 else folder_name
        else:
            locale = folder_name
            experiment_type = folder_name

    return ExperimentMetadata(
        experiment_type=experiment_type,
        locale=locale,
        target_lang=locale,
        source_languages=None,
        weights=None,
        merge_mode=experiment_type,
        num_languages=num_languages,
        timestamp=None,
        folder_name=folder_name
    )

def extract_accuracy(results):
    """Extract accuracy from results."""
    if results and 'performance' in results:
        return results['performance']['accuracy']
    return None

def get_experiment_folders(results_dir="results"):
    """Get all experiment folders from the results directory."""
    if not results_dir:
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


def aggregate_results(evaluation_matrix: Optional[pd.DataFrame] = None, results_dir: str = "results"):
    """Aggregate results from all experiment folders with dynamic parsing."""
    folders = get_experiment_folders(results_dir)

    data = []

    for folder in folders:
        folder_path = os.path.join(results_dir, folder)
        results = load_results_from_folder(folder_path)

        if results:
            eval_info = results.get('evaluation_info', {})
            perf_info = results.get('performance', {})
            model_name = eval_info.get('model_name')

            # Use new dynamic metadata parsing
            metadata = parse_experiment_metadata(folder, folder_path, model_name)
            accuracy = extract_accuracy(results)

            locale = eval_info.get('locale') or metadata.locale
            target_lang = metadata.target_lang or locale

            num_languages = metadata.num_languages
            if num_languages is None and is_merge_model_path(model_name):
                num_languages = parse_num_languages_from_model_path(model_name)
            if num_languages is None and is_merge_model_path(model_name):
                counted = count_languages_in_merge_details(model_name)
                num_languages = counted if counted is not None else None
            if num_languages is None and is_merge_model_path(model_name):
                # Default legacy merges without explicit counts to 4
                num_languages = 4

            # Extract base model using model-agnostic detection
            base_model = None
            from merginguriel.naming_config import naming_manager

            # Try to extract from folder name first
            if folder:
                try:
                    base_model = naming_manager.extract_model_family(folder)
                except ValueError:
                    pass

            # Fallback to model_name for baseline models
            if not base_model and model_name:
                try:
                    base_model = naming_manager.extract_model_family(model_name)
                except ValueError:
                    pass

            experiment_variant = determine_experiment_variant(
                metadata.merge_mode or metadata.experiment_type,
                num_languages,
                model_name,
                eval_info.get('similarity_type') or metadata.__dict__.get('similarity_type', 'URIEL'),
                base_model
            )

            # Harmonize with evaluation prefix in folder name (e.g., average_19lang_locale)
            folder_prefix = folder.rsplit('_', 1)[0] if '_' in folder else folder
            prefix_match = re.match(r'(?P<method>.+?)_(?P<count>\d+)lang$', folder_prefix)
            if prefix_match:
                experiment_variant = folder_prefix
                if not metadata.experiment_type or metadata.experiment_type == 'unknown':
                    metadata.experiment_type = prefix_match.group('method')
                try:
                    num_languages = int(prefix_match.group('count'))
                except ValueError:
                    pass

            # Calculate baseline data if evaluation matrix is available
            baseline_data = None
            if evaluation_matrix is not None and metadata.source_languages:
                baseline_data = get_baseline_for_target(
                    locale,
                    metadata.source_languages,
                    evaluation_matrix
                )

            # Build the data row with similarity type
            data_row = {
                'locale': locale,
                'experiment_type': metadata.experiment_type,
                'experiment_variant': experiment_variant,
                'folder_name': folder,
                'accuracy': accuracy,
                'correct_predictions': perf_info.get('correct_predictions'),
                'total_predictions': perf_info.get('total_predictions'),
                'error_rate': perf_info.get('error_rate'),
                'model_name': model_name,
                'subfolder': eval_info.get('subfolder'),
                'timestamp': eval_info.get('timestamp'),
                # New metadata fields
                'target_lang': target_lang,
                'source_languages': metadata.source_languages,
                'weights': metadata.weights,
                'merge_mode': metadata.merge_mode,
                'num_languages': num_languages,
                'merge_timestamp': metadata.timestamp,
                # Similarity type for differentiation
                'similarity_type': eval_info.get('similarity_type') or metadata.__dict__.get('similarity_type', 'URIEL')
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
    column_field = 'experiment_variant' if 'experiment_variant' in df.columns else 'experiment_type'

    # Get unique experiment variants dynamically
    experiment_types = sorted(df[column_field].dropna().unique())

    # Pivot the data to have one row per locale
    pivot_df = df.pivot_table(
        index='locale',
        columns=column_field,
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

    column_field = 'experiment_variant' if 'experiment_variant' in df.columns else 'experiment_type'
    experiment_types = sorted(df[column_field].dropna().unique())

    # Overall statistics by experiment type
    for exp_type in experiment_types:
        type_df = df[df[column_field] == exp_type]
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
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing experiment results (default: results)")

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
    df = aggregate_results(evaluation_matrix, args.results_dir)

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
        mask = df['experiment_type'].isin(args.experiment_types)
        if 'experiment_variant' in df.columns:
            mask = mask | df['experiment_variant'].isin(args.experiment_types)
        df = df[mask]
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
        if 'experiment_variant' in df.columns:
            experiment_types = sorted(df['experiment_variant'].dropna().unique())
        else:
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
