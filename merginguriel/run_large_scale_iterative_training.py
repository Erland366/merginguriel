#!/usr/bin/env python3
"""
Large-scale iterative training experiment runner that orchestrates iterative training across locales.

This script mirrors run_large_scale_experiment.py but focuses on iterative training methods
instead of post-training merging. It runs multiple target languages with sequential training
and generates results compatible with the existing aggregation system.

Key features:
- Supports multiple iterative training modes: similarity, average, fisher, etc.
- Auto-selects source languages based on URIEL similarity to target
- Progress tracking and resumption capability with comprehensive state management
- Results compatible with aggregate_results.py
- Resource management for long-running training experiments
- Integration with existing iterative training infrastructure
"""

import os
import sys
import pandas as pd
import subprocess
import json
import argparse
import time
import psutil
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from merginguriel import logger

# Ensure project root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add submodules for merging methods
SUBMODULES_DIR = os.path.join(REPO_ROOT, "submodules", "auto_merge_llm")
if SUBMODULES_DIR not in sys.path:
    sys.path.insert(0, SUBMODULES_DIR)

try:
    from auto_merge_llm.methods import merging_methods_dict
except Exception:
    merging_methods_dict = {}

# Import MergingUriel components
from merginguriel.similarity_utils import load_and_process_similarity

# Loguru logger imported from merginguriel package


def get_all_locales_from_similarity_matrix(similarity_type="URIEL"):
    """Extract all unique locales from the similarity matrix."""
    if similarity_type == "URIEL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "language_similarity_matrix_unified.csv")
    elif similarity_type == "REAL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "nxn_results", "nxn_eval_20251027_103544", "evaluation_matrix.csv")
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    df = pd.read_csv(similarity_matrix_path, index_col=0)
    locales = sorted(df.index.tolist())
    return locales


def get_available_locales():
    """Get list of locales that have available models."""
    models_dir = os.path.join(REPO_ROOT, "haryos_model")
    available_locales = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, item)) and item.startswith("xlm-roberta-base_massive_k_"):
                locale = item.replace("xlm-roberta-base_massive_k_", "")
                available_locales.append(locale)
    return sorted(available_locales)


def auto_select_iterative_sources(target_lang: str, max_models: int, top_k: int = 20,
                                sinkhorn_iters: int = 20) -> List[str]:
    """
    Auto-select source languages for iterative training based on similarity to target.

    Args:
        target_lang: Target language code
        max_models: Maximum number of source models to include
        top_k: Number of similar languages to consider from similarity matrix
        sinkhorn_iters: Sinkhorn normalization iterations

    Returns:
        List of selected source locale codes
    """
    logger.info(f"Auto-selecting source languages for target: {target_lang}")

    # Get similarity matrix path
    similarity_matrix_path = os.path.join(REPO_ROOT, "language_similarity_matrix_unified.csv")

    try:
        # Get similar languages using existing similarity processing
        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang, max_models,
            top_k, sinkhorn_iters, verbose=False
        )

        # Extract just the locale codes
        source_locales = [locale for locale, weight in similar_languages]

        # Get available locales to filter results
        available_locales = get_available_locales()
        available_set = set(available_locales)

        # Filter to only available locales
        final_locales = [locale for locale in source_locales if locale in available_set]

        # Limit to max_models
        final_locales = final_locales[:max_models]

        logger.info(f"Auto-selected {len(final_locales)} source languages: {final_locales}")
        return final_locales

    except Exception as e:
        logger.error(f"Error auto-selecting source languages: {e}")
        # Fallback to some reasonable defaults
        fallback_locales = ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT"]
        available_locales = get_available_locales()
        final_fallback = [loc for loc in fallback_locales if loc in available_locales][:max_models]
        logger.warning(f"Using fallback source languages: {final_fallback}")
        return final_fallback


def run_iterative_training(target_lang: str, source_locales: List[str], mode: str,
                         extra_args: List[str], output_base_dir: str) -> bool:
    """
    Run iterative training for a specific target language.

    Args:
        target_lang: Target language for training
        source_locales: List of source language locales
        mode: Iterative training mode
        extra_args: Additional arguments for training
        output_base_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting iterative training for {target_lang} with mode {mode}")
    logger.info(f"Source languages: {source_locales}")

    # Create locale-specific output directory
    locale_output_dir = os.path.join(output_base_dir, f"iterative_{mode}_{target_lang}")
    os.makedirs(locale_output_dir, exist_ok=True)

    # Build command for iterative training
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "merginguriel", "run_iterative_training.py"),
        "--mode", mode,
        "--target-lang", target_lang,
        "--locales", ",".join(source_locales),
        "--output-dir", locale_output_dir,
        "--experiment-name", f"iterative_{mode}_{target_lang}",
        "--sequential-training",  # Always use sequential to prevent OOM
        "--enable-auto-recovery",  # Enable recovery for long-running experiments
        "--enable-wandb",  # Enable wandb for large-scale experiments
        "--wandb-mode", "offline",  # Keep wandb files local for large-scale experiments
        "--save-config"  # Save configuration for reproducibility
    ]

    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Log the command for debugging
    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run the iterative training process
        start_time = time.time()

        # Use subprocess with real-time output logging
        # Set environment variable to enable log files for large-scale experiments
        env = os.environ.copy()
        env["LARGE_SCALE_EXPERIMENT"] = "true"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )

        # Log output in real-time
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[{target_lang}] {line.strip()}")

        # Wait for completion and get return code
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        if return_code == 0:
            logger.info(f"✓ Iterative training completed for {target_lang} in {elapsed_time/3600:.1f} hours")
            return True
        else:
            logger.error(f"✗ Iterative training failed for {target_lang} with return code {return_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Error running iterative training for {target_lang}: {e}")
        return False


def extract_iterative_results(target_lang: str, mode: str, locale_output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Extract results from iterative training output for compatibility with aggregate_results.py.

    Args:
        target_lang: Target language
        mode: Training mode
        locale_output_dir: Output directory for this locale

    Returns:
        Dictionary with results in expected format, or None if extraction failed
    """
    logger.info(f"Extracting results for {target_lang} from {locale_output_dir}")

    try:
        # Look for the best performing merged model
        # The iterative training system should have evaluated merged models
        merged_models_dir = os.path.join(locale_output_dir, "merged_models")

        if not os.path.exists(merged_models_dir):
            logger.warning(f"No merged models directory found for {target_lang}")
            return None

        # Find evaluation results for merged models
        best_accuracy = 0.0
        best_model_info = None

        # Look for results files or evaluation logs
        for item in os.listdir(merged_models_dir):
            if item.startswith("merged_model_") and os.path.isdir(os.path.join(merged_models_dir, item)):
                model_dir = os.path.join(merged_models_dir, item)

                # Look for evaluation results
                results_file = os.path.join(model_dir, "evaluation_results.json")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            eval_results = json.load(f)

                        accuracy = eval_results.get('accuracy', 0.0)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_info = {
                                'model_name': item,
                                'accuracy': accuracy,
                                'eval_results': eval_results
                            }
                    except Exception as e:
                        logger.warning(f"Error reading evaluation results for {item}: {e}")

        if best_model_info is None:
            logger.warning(f"No valid evaluation results found for {target_lang}")
            return None

        # Get training configuration
        config_file = os.path.join(locale_output_dir, "experiment_config.json")
        training_config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    training_config = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading training config: {e}")

        # Extract source languages from training config or logs
        source_languages = training_config.get('training_configs', [])
        source_locales = [config.get('locale', '') for config in source_languages if config.get('locale')]

        # Get merge details from best model
        merge_details = best_model_info['eval_results'].get('merge_details', {})

        # Build results in expected format for aggregate_results.py
        results = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": f"iterative_{mode}_{target_lang}",
                "locale": target_lang,
                "massive_locale": target_lang,
                "dataset": "AmazonScience/massive",
                "split": "test",
                "experiment_type": f"iterative_{mode}",
                "mode": mode,
                "training_type": "iterative"
            },
            "model_info": {
                "num_models": len(source_locales),
                "model_names": source_locales,
                "training_mode": mode,
                "best_merged_model": best_model_info['model_name'],
                "training_config": training_config
            },
            "dataset_info": {
                "total_examples": best_model_info['eval_results'].get('total_examples', 0),
                "training_type": "iterative"
            },
            "performance": {
                "accuracy": best_accuracy,
                "correct_predictions": best_model_info['eval_results'].get('correct_predictions'),
                "total_predictions": best_model_info['eval_results'].get('total_predictions'),
                "error_rate": 1.0 - best_accuracy
            },
            "metadata": {
                "source_languages": source_locales,
                "merge_details": merge_details,
                "training_output_dir": locale_output_dir,
                "best_model_info": best_model_info
            }
        }

        logger.info(f"✓ Extracted results for {target_lang}: accuracy = {best_accuracy:.4f}")
        return results

    except Exception as e:
        logger.error(f"✗ Error extracting results for {target_lang}: {e}")
        return None


def save_iterative_results(target_lang: str, mode: str, results: Dict[str, Any],
                         results_dir: str = "results") -> bool:
    """
    Save iterative training results in format compatible with aggregate_results.py.

    Args:
        target_lang: Target language
        mode: Training mode
        results: Results dictionary
        results_dir: Base results directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create folder name in expected format
        folder_name = f"iterative_{mode}_{target_lang}"
        results_folder = os.path.join(REPO_ROOT, results_dir, folder_name)
        os.makedirs(results_folder, exist_ok=True)

        # Save results.json
        results_file = os.path.join(results_folder, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to {results_file}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to save iterative results for {target_lang} ({mode}): {e}")
        return False


def monitor_system_resources():
    """Monitor system resources for logging and optimization."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage(REPO_ROOT)
        disk_percent = (disk.used / disk.total) * 100

        # GPU usage (if nvidia-smi is available)
        gpu_info = {}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info[f'gpu_{i}'] = {
                            'utilization': parts[0],
                            'memory_used': parts[1],
                            'memory_total': parts[2]
                        }
        except Exception:
            pass

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'gpu_info': gpu_info
        }
    except Exception as e:
        logger.warning(f"Error monitoring system resources: {e}")
        return {}


def run_experiment_for_locale(
    locale: str,
    mode: str,
    training_extra_args: List[str],
    output_base_dir: str,
    max_models: int,
    top_k: int,
    sinkhorn_iters: int
) -> Dict[str, bool]:
    """
    Run the iterative training experiment for a single locale.

    Args:
        locale: Target locale
        mode: Training mode
        training_extra_args: Additional arguments for training
        output_base_dir: Base output directory
        max_models: Maximum number of source models
        top_k: Top-K for similarity selection
        sinkhorn_iters: Sinkhorn iterations

    Returns:
        Dictionary with results status
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running iterative training experiment for locale: {locale}")
    logger.info(f"{'='*60}")

    # Check if target model exists (for validation)
    target_model_path = os.path.join(REPO_ROOT, f"haryos_model/xlm-roberta-base_massive_k_{locale}")
    if not os.path.exists(target_model_path):
        logger.warning(f"Target model not found for {locale}: {target_model_path}")
        # Don't skip - iterative training creates its own models

    results = {}

    # Auto-select source languages
    source_locales = auto_select_iterative_sources(locale, max_models, top_k, sinkhorn_iters)

    if not source_locales:
        logger.error(f"No source languages found for {locale}")
        return {'success': False, 'error': 'No source languages available'}

    # Monitor system resources before starting
    resources_before = monitor_system_resources()
    logger.info(f"System resources before training: CPU={resources_before.get('cpu_percent', 0):.1f}%, "
               f"Memory={resources_before.get('memory_percent', 0):.1f}%, "
               f"Disk={resources_before.get('disk_percent', 0):.1f}%")

    # Run iterative training
    training_success = run_iterative_training(
        locale, source_locales, mode, training_extra_args, output_base_dir
    )

    if training_success:
        # Extract results from training output
        locale_output_dir = os.path.join(output_base_dir, f"iterative_{mode}_{locale}")
        training_results = extract_iterative_results(locale, mode, locale_output_dir)

        if training_results:
            # Save results in compatible format
            save_success = save_iterative_results(locale, mode, training_results)
            results['success'] = save_success
            results['accuracy'] = training_results['performance']['accuracy']
        else:
            results['success'] = False
            results['error'] = 'Failed to extract training results'
    else:
        results['success'] = False
        results['error'] = 'Iterative training failed'

    # Monitor system resources after completion
    resources_after = monitor_system_resources()
    logger.info(f"System resources after training: CPU={resources_after.get('cpu_percent', 0):.1f}%, "
               f"Memory={resources_after.get('memory_percent', 0):.1f}%, "
               f"Disk={resources_after.get('disk_percent', 0):.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run large-scale iterative training experiments")
    parser.add_argument("--target-languages", nargs="+", default=None,
                       help="Specific target languages to run (default: all available locales)")
    parser.add_argument("--mode", type=str, default="similarity",
                       choices=["similarity", "average", "fisher", "fisher_simple", "fisher_dataset",
                               "ties", "task_arithmetic", "slerp", "regmean"],
                       help="Iterative training mode to use")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       help="Maximum number of locales to process")
    parser.add_argument("--max-models", type=int, default=5,
                       help="Maximum number of source models to include in training")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    parser.add_argument("--cleanup-intermediate", action="store_true",
                       help="Clean up intermediate artifacts after each locale")

    # Pass-through options for run_iterative_training.py
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Training batch size")
    parser.add_argument("--max-seq-length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--merge-frequency", type=int, default=3,
                       help="Number of epochs between merges")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bf16 precision for training (default: enabled)")
    parser.add_argument("--enable-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--output-dir", type=str, default="iterative_large_scale_results",
                       help="Base output directory for all results")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Number of top similar languages to consider")
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       help="Number of Sinkhorn normalization iterations")
    parser.add_argument("--timeout-hours", type=float, default=12.0,
                       help="Maximum hours to wait per locale")
    parser.add_argument("--sequential-training", action="store_true", default=True,
                       help="Train models sequentially to prevent OOM")

    args = parser.parse_args()

    # Get all available locales (those with models)
    available_locales = get_available_locales()

    if args.list_locales:
        print("Available locales (with models):")
        for i, locale in enumerate(available_locales):
            print(f"  {i:2d}. {locale}")
        print(f"\nTotal: {len(available_locales)} locales")
        return

    # Filter locales
    if args.target_languages:
        locales = [loc for loc in available_locales if loc in args.target_languages]
    else:
        locales = available_locales

    # Apply start index and max limit
    if args.start_from > 0:
        locales = locales[args.start_from:]
    if args.max_locales:
        locales = locales[:args.max_locales]

    logger.info(f"Starting large-scale iterative training experiment for {len(locales)} locales")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Start from index: {args.start_from}")
    logger.info(f"Max models per locale: {args.max_models}")

    # Track overall results
    overall_results = {}

    # Build pass-through args for iterative training
    training_extra_args = [
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--max-seq-length", str(args.max_seq_length),
        "--merge-frequency", str(args.merge_frequency),
        "--top-k", str(args.top_k),
        "--sinkhorn-iters", str(args.sinkhorn_iters)
    ]

    # Add optional flags
    # fp16 removed - bf16 is now the standard precision
    if args.enable_wandb:
        training_extra_args.append("--enable-wandb")

    for i, locale in enumerate(locales):
        logger.info(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")

        start_time = time.time()
        results = run_experiment_for_locale(
            locale, args.mode, training_extra_args, args.output_dir,
            args.max_models, args.top_k, args.sinkhorn_iters
        )
        elapsed_time = time.time() - start_time

        overall_results[locale] = results
        overall_results[locale]['elapsed_time'] = elapsed_time

        # Save progress after each locale
        progress_file = os.path.join(REPO_ROOT, "iterative_large_scale_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_locales': len(available_locales),
                'processed_locales': i + 1,
                'current_locale': locale,
                'results': overall_results,
                'config': {
                    'mode': args.mode,
                    'max_models': args.max_models,
                    'epochs': args.epochs,
                    'merge_frequency': args.merge_frequency
                }
            }, f, indent=2)

        logger.info(f"Progress saved to {progress_file}")
        logger.info(f"Completed {locale} in {elapsed_time/3600:.1f} hours")

    logger.info(f"\n{'='*60}")
    logger.info("Large-scale iterative training experiment completed!")
    logger.info(f"{'='*60}")

    # Summary
    total_locales = len(overall_results)
    successful_locales = sum(1 for r in overall_results.values() if r.get('success', False))
    logger.info(f"Total locales processed: {total_locales}")
    logger.info(f"Successful experiments: {successful_locales}/{total_locales}")

    if successful_locales > 0:
        accuracies = [r['accuracy'] for r in overall_results.values() if r.get('accuracy') is not None]
        if accuracies:
            logger.info(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
            logger.info(f"Mean accuracy: {sum(accuracies)/len(accuracies):.4f}")

    # Save final results
    final_results_file = os.path.join(REPO_ROOT, "iterative_large_scale_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': args.mode,
                'start_from': args.start_from,
                'max_locales': args.max_locales,
                'max_models': args.max_models,
                'epochs': args.epochs,
                'merge_frequency': args.merge_frequency,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'top_k': args.top_k,
                'sinkhorn_iters': args.sinkhorn_iters,
                'output_dir': args.output_dir
            },
            'summary': {
                'total_locales': len(overall_results),
                'successful_locales': successful_locales,
                'success_rate': successful_locales / total_locales if total_locales > 0 else 0
            },
            'detailed_results': overall_results
        }, f, indent=2)

    logger.info(f"Final results saved to {final_results_file}")


if __name__ == "__main__":
    main()