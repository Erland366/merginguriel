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

# Import centralized naming system
from merginguriel.naming_config import naming_manager

# Import config system
from merginguriel.config import (
    IterativeExperimentConfig,
    SimilarityConfig,
    TargetConfig,
    ModelConfig,
    DatasetConfig,
    OutputConfig,
    TrainingConfig,
    MergeConfig as MergeConfigBase,
    ConfigLoader,
    TrackProvidedArgsAction,
    ITERATIVE_EXPERIMENT_ARG_MAP,
)

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
    locales = sorted(set(df.index.tolist()))
    return locales


def get_available_locales(models_root="haryos_model"):
    """Get list of locales that have available models."""
    models_dir = os.path.join(REPO_ROOT, models_root)
    available_locales = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, item)):
                # Use model-agnostic detection to find massive_k pattern
                if "_massive_k_" in item:
                    try:
                        # Extract locale from pattern like "model-family_massive_k_locale"
                        parts = item.split("_massive_k_")
                        if len(parts) == 2:
                            locale = parts[1]
                            available_locales.append(locale)
                    except Exception:
                        pass
    return sorted(available_locales)


def auto_select_iterative_sources(target_lang: str, max_models: int, similarity_type: str = "URIEL",
                                models_root: str = "haryos_model", top_k: int = 20,
                                sinkhorn_iters: int = 20, include_target: bool = False) -> List[str]:
    """
    Auto-select source languages for iterative training based on similarity to target.

    Args:
        target_lang: Target language code
        max_models: Maximum number of source models to include
        similarity_type: Type of similarity matrix to use (URIEL or REAL)
        models_root: Root directory containing models
        top_k: Number of similar languages to consider from similarity matrix
        sinkhorn_iters: Sinkhorn normalization iterations

    Returns:
        List of selected source locale codes
    """
    logger.info(f"Auto-selecting source languages for target: {target_lang}")

    # Get similarity matrix path
    if similarity_type == "URIEL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "language_similarity_matrix_unified.csv")
    elif similarity_type == "REAL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "nxn_results", "nxn_eval_20251027_103544", "evaluation_matrix.csv")
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    try:
        # Get similar languages using existing similarity processing
        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang, max_models,
            top_k, sinkhorn_iters, include_target, verbose=False
        )

        # Extract just the locale codes
        source_locales = [locale for locale, weight in similar_languages]

        # Get available locales to filter results
        available_locales = get_available_locales(models_root)
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
        available_locales = get_available_locales(models_root)
        final_fallback = [loc for loc in fallback_locales if loc in available_locales][:max_models]
        logger.warning(f"Using fallback source languages: {final_fallback}")
        return final_fallback


def results_already_exist(
    experiment_type: str,
    method: str,
    similarity_type: str,
    locale: str,
    model_family: str,
    results_dir: str,
    num_languages: Optional[int],
    include_target: Optional[bool],
) -> Optional[str]:
    """Return an existing results directory if it already holds results.json."""
    base_dir = os.path.join(REPO_ROOT, results_dir)
    if not os.path.exists(base_dir):
        return None

    existing = naming_manager.find_results_directory(
        base_dir,
        experiment_type=experiment_type,
        method=method,
        similarity_type=similarity_type,
        locale=locale,
        model_family=model_family,
        num_languages=num_languages,
        include_target=include_target,
    )
    if existing:
        results_file = os.path.join(existing, "results.json")
        if os.path.exists(results_file):
            return existing
    return None


def load_accuracy_from_results(results_dir: str) -> Optional[float]:
    """Extract accuracy from an existing results.json file if available."""
    results_file = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_file):
        return None
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        return data.get("performance", {}).get("accuracy")
    except Exception:
        return None


def run_iterative_training(target_lang: str, source_locales: List[str], mode: str,
                         extra_args: List[str], output_base_dir: str, include_target: bool = False) -> bool:
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

    # Create locale-specific output directory with IT/ET naming
    itet_suffix = "IT" if include_target else "ET"
    locale_output_dir = os.path.join(output_base_dir, f"iterative_{mode}_{target_lang}_{itet_suffix}")
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
                         results_dir: str = "results", model_family: str = None,
                         similarity_type: str = "URIEL", num_languages: int = None,
                         include_target: bool = False) -> bool:
    """
    Save iterative training results in format compatible with aggregate_results.py.

    Args:
        target_lang: Target language
        mode: Training mode
        results: Results dictionary
        results_dir: Base results directory
        model_family: Model family name
        similarity_type: Similarity type used
        num_languages: Number of languages

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create folder name using centralized naming system
        folder_name = naming_manager.get_results_dir_name(
            experiment_type='iterative',
            method=mode,
            similarity_type=similarity_type,
            locale=target_lang,
            model_family=model_family,
            num_languages=num_languages,
            include_target=include_target
        )
        results_folder = os.path.join(REPO_ROOT, results_dir, folder_name)
        os.makedirs(results_folder, exist_ok=True)

        # Update results with missing information
        results['evaluation_info']['model_family_name'] = model_family
        results['evaluation_info']['similarity_type'] = similarity_type

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
    include_target_modes: List[bool],
    similarity_type: str = "URIEL",
    models_root: str = "haryos_model",
    top_k: int = 20,
    sinkhorn_iters: int = 20,
    results_dir: str = "results",
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run the iterative training experiment for a single locale.

    Args:
        locale: Target locale
        mode: Training mode
        training_extra_args: Additional arguments for training
        output_base_dir: Base output directory
        max_models: Maximum number of source models
        similarity_type: Type of similarity matrix to use
        models_root: Root directory containing models
        top_k: Top-K for similarity selection
        sinkhorn_iters: Sinkhorn iterations

    Returns:
        Dictionary with results status
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running iterative training experiment for locale: {locale} (models: {models_root}, similarity: {similarity_type})")
    logger.info(f"{'='*60}")

    # Check if target model exists (for validation)
    target_model_path = os.path.join(REPO_ROOT, f"{models_root}/xlm-roberta-base_massive_k_{locale}")
    if not os.path.exists(target_model_path):
        logger.warning(f"Target model not found for {locale}: {target_model_path}")
        # Don't skip - iterative training creates its own models

    # Get model family name using centralized naming system
    try:
        model_family = naming_manager.detect_model_family_from_path(target_model_path)
        logger.info(f"✓ Detected model family: {model_family}")
    except Exception as e:
        logger.error(f"❌ Failed to detect model family: {e}")
        return {'success': False, 'error': f'Failed to detect model family: {e}'}

    # Validate required components
    try:
        naming_manager.validate_required_components(
            experiment_type='iterative',
            method=mode,
            similarity_type=similarity_type,
            locale=locale,
            model_family=model_family,
            model_path=target_model_path
        )
    except ValueError as e:
        logger.error(f"❌ Validation failed: {e}")
        return {'success': False, 'error': f'Validation failed: {e}'}

    variant_results: Dict[str, Any] = {}
    aggregate_success = True
    accuracy_map: Dict[str, float] = {}

    for include_target in include_target_modes:
        variant_label = "IncTar" if include_target else "ExcTar"
        logger.info(f"\n--- Iterative training variant: {variant_label} ---")

        variant_result: Dict[str, Any] = {}

        existing_results = None
        if resume:
            existing_results = results_already_exist(
                experiment_type="iterative",
                method=mode,
                similarity_type=similarity_type,
                locale=locale,
                model_family=model_family,
                results_dir=results_dir,
                num_languages=max_models,
                include_target=include_target,
            )
        if existing_results:
            logger.info(f"↩️  Skipping iterative training for {locale} ({variant_label}); results already at {existing_results}")
            variant_result["success"] = True
            existing_accuracy = load_accuracy_from_results(existing_results)
            if existing_accuracy is not None:
                variant_result["accuracy"] = existing_accuracy
                accuracy_map[variant_label] = existing_accuracy
            variant_results[variant_label] = variant_result
            continue

        # Auto-select source languages for this variant
        source_locales = auto_select_iterative_sources(
            locale, max_models, similarity_type, models_root, top_k, sinkhorn_iters, include_target
        )

        if not source_locales:
            logger.error(f"No source languages found for {locale} ({variant_label})")
            variant_result['success'] = False
            variant_result['error'] = 'No source languages available'
            aggregate_success = False
            variant_results[variant_label] = variant_result
            continue

        resources_before = monitor_system_resources()
        logger.info(f"System resources before training ({variant_label}): "
                    f"CPU={resources_before.get('cpu_percent', 0):.1f}%, "
                    f"Memory={resources_before.get('memory_percent', 0):.1f}%, "
                    f"Disk={resources_before.get('disk_percent', 0):.1f}%")

        training_success = run_iterative_training(
            locale, source_locales, mode, training_extra_args, output_base_dir, include_target
        )

        if training_success:
            itet_suffix = "IT" if include_target else "ET"
            locale_output_dir = os.path.join(output_base_dir, f"iterative_{mode}_{locale}_{itet_suffix}")
            training_results = extract_iterative_results(locale, mode, locale_output_dir)

            if training_results:
                save_success = save_iterative_results(
                    locale,
                    mode,
                    training_results,
                    results_dir=results_dir,
                    model_family=model_family,
                    similarity_type=similarity_type,
                    num_languages=max_models,
                    include_target=include_target
                )
                variant_result['success'] = save_success
                accuracy = training_results['performance']['accuracy']
                variant_result['accuracy'] = accuracy
                accuracy_map[variant_label] = accuracy
            else:
                variant_result['success'] = False
                variant_result['error'] = 'Failed to extract training results'
        else:
            variant_result['success'] = False
            variant_result['error'] = 'Iterative training failed'

        resources_after = monitor_system_resources()
        logger.info(f"System resources after training ({variant_label}): "
                    f"CPU={resources_after.get('cpu_percent', 0):.1f}%, "
                    f"Memory={resources_after.get('memory_percent', 0):.1f}%, "
                    f"Disk={resources_after.get('disk_percent', 0):.1f}%")

        aggregate_success = aggregate_success and variant_result.get('success', False)
        variant_results[variant_label] = variant_result

    results: Dict[str, Any] = {
        'variants': variant_results,
        'success': aggregate_success,
    }
    if accuracy_map:
        results['accuracy'] = accuracy_map

    return results


def create_config_from_args(args) -> IterativeExperimentConfig:
    """Create an IterativeExperimentConfig from parsed CLI arguments."""
    # Determine target inclusion
    inclusion = args.target_inclusion or "ExcTar"
    if inclusion in ("include", "IncTar"):
        inclusion = "IncTar"
    elif inclusion in ("exclude", "ExcTar"):
        inclusion = "ExcTar"

    return IterativeExperimentConfig(
        target_languages=args.target_languages,
        mode=args.mode,
        similarity=SimilarityConfig(
            type=args.similarity_type,
            top_k=args.top_k,
            sinkhorn_iters=args.sinkhorn_iters,
        ),
        target=TargetConfig(
            locale="",  # Will be set per-locale
            inclusion=inclusion,
        ),
        model=ModelConfig(
            models_root=args.models_root,
            num_languages=args.max_models,
        ),
        output=OutputConfig(
            results_dir=args.results_dir,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            bf16=args.bf16,
        ),
        merge=MergeConfigBase(
            frequency=args.merge_frequency,
        ),
        sequential_training=args.sequential_training,
        enable_wandb=args.enable_wandb,
        wandb_mode="online" if args.enable_wandb else "disabled",
        resume=args.resume,
        start_from=args.start_from,
        max_locales=args.max_locales,
        max_models=args.max_models,
        timeout_hours=args.timeout_hours,
        cleanup_intermediate=args.cleanup_intermediate,
    )


def main():
    parser = argparse.ArgumentParser(description="Run large-scale iterative training experiments")

    # Config file argument (new, preferred)
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to YAML config file. When provided, CLI args override config values with deprecation warnings.")

    parser.add_argument("--target-languages", nargs="+", default=None,
                       action=TrackProvidedArgsAction,
                       help="Specific target languages to run (default: all available locales)")
    parser.add_argument("--mode", type=str, default="similarity",
                       choices=["similarity", "average", "fisher", "fisher_simple", "fisher_dataset",
                               "ties", "task_arithmetic", "slerp", "regmean"],
                       action=TrackProvidedArgsAction,
                       help="Iterative training mode to use")
    parser.add_argument("--start-from", type=int, default=0,
                       action=TrackProvidedArgsAction,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       action=TrackProvidedArgsAction,
                       help="Maximum number of locales to process")
    parser.add_argument("--max-models", type=int, default=5,
                       action=TrackProvidedArgsAction,
                       help="Maximum number of source models to include in training")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    parser.add_argument("--cleanup-intermediate", action="store_true",
                       help="Clean up intermediate artifacts after each locale")

    # Pass-through options for run_iterative_training.py
    parser.add_argument("--epochs", type=int, default=15,
                       action=TrackProvidedArgsAction,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       action=TrackProvidedArgsAction,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128,
                       action=TrackProvidedArgsAction,
                       help="Training batch size")
    parser.add_argument("--max-seq-length", type=int, default=128,
                       action=TrackProvidedArgsAction,
                       help="Maximum sequence length")
    parser.add_argument("--merge-frequency", type=int, default=3,
                       action=TrackProvidedArgsAction,
                       help="Number of epochs between merges")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bf16 precision for training (default: enabled)")
    parser.add_argument("--enable-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--output-dir", type=str, default="iterative_large_scale_results",
                       action=TrackProvidedArgsAction,
                       help="Base output directory for all results")
    parser.add_argument("--top-k", type=int, default=20,
                       action=TrackProvidedArgsAction,
                       help="Number of top similar languages to consider")
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       action=TrackProvidedArgsAction,
                       help="Number of Sinkhorn normalization iterations")
    parser.add_argument("--timeout-hours", type=float, default=12.0,
                       action=TrackProvidedArgsAction,
                       help="Maximum hours to wait per locale")
    parser.add_argument("--sequential-training", action="store_true", default=True,
                       help="Train models sequentially to prevent OOM")

    # Additional options for consistency with merging/ensemble experiments
    parser.add_argument("--similarity-type", type=str, choices=["URIEL","REAL"], default="URIEL",
                       action=TrackProvidedArgsAction,
                       help="Type of similarity matrix to use: URIEL (linguistic features) or REAL (empirical evaluation results)")
    parser.add_argument("--target-inclusion", type=str,
                       choices=["IncTar", "ExcTar", "include", "exclude", "both"],
                       default=None,
                       action=TrackProvidedArgsAction,
                       help="Target inclusion mode for iterative training. Default runs both variants.")
    parser.add_argument("--include-target", action="store_true", dest="legacy_include_target",
                       help="Deprecated alias for --target-inclusion IncTar.")
    parser.add_argument("--exclude-target", action="store_true", dest="legacy_exclude_target",
                       help="Deprecated alias for --target-inclusion ExcTar.")
    parser.add_argument("--models-root", type=str, default="haryos_model",
                       action=TrackProvidedArgsAction,
                       help="Root directory containing models (default: haryos_model)")
    parser.add_argument("--results-dir", type=str, default="results",
                       action=TrackProvidedArgsAction,
                       help="Directory for experiment results (default: results)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                       help="Reuse existing results when present (default: enabled). Use --no-resume to force rerun.")

    args = parser.parse_args()

    # Load config from file or create from CLI args
    if args.config is not None:
        logger.info(f"Loading config from: {args.config}")
        config = IterativeExperimentConfig.from_yaml(args.config)

        # Extended arg map for iterative experiment specific args
        extended_arg_map = {
            **ITERATIVE_EXPERIMENT_ARG_MAP,
            "target_languages": "target_languages",
            "mode": "mode",
            "start_from": "start_from",
            "max_locales": "max_locales",
            "max_models": "max_models",
            "output_dir": "output.results_dir",
        }

        # Merge CLI args with deprecation warnings
        config = ConfigLoader.merge_cli_args(config, args, emit_warnings=True, arg_to_config_map=extended_arg_map)
    else:
        # Legacy path: create config from CLI args (no warnings)
        config = create_config_from_args(args)

    # Handle legacy target inclusion flags (only relevant when not using config file)
    if args.config is None:
        if args.target_inclusion is None:
            if args.legacy_include_target and args.legacy_exclude_target:
                parser.error("Cannot specify both --include-target and --exclude-target.")
            if args.legacy_include_target:
                config.target.inclusion = "IncTar"
            elif args.legacy_exclude_target:
                config.target.inclusion = "ExcTar"
            else:
                config.target.inclusion = "both"
        else:
            if args.legacy_include_target or args.legacy_exclude_target:
                parser.error("Do not mix --target-inclusion with legacy include/exclude flags.")
            config.target.inclusion = args.target_inclusion

    # Get all available locales (those with models)
    available_locales = get_available_locales(config.model.models_root)

    if args.list_locales:
        print("Available locales (with models):")
        for i, locale in enumerate(available_locales):
            print(f"  {i:2d}. {locale}")
        print(f"\nTotal: {len(available_locales)} locales")
        return

    # Filter locales
    if config.target_languages:
        locales = [loc for loc in available_locales if loc in config.target_languages]
    else:
        locales = available_locales

    # Apply start index and max limit
    if config.start_from > 0:
        locales = locales[config.start_from:]
    if config.max_locales:
        locales = locales[:config.max_locales]

    logger.info(f"Starting large-scale iterative training experiment for {len(locales)} locales")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Similarity Type: {config.similarity.type}")
    logger.info(f"Model Root: {config.model.models_root}")
    logger.info(f"Start from index: {config.start_from}")
    logger.info(f"Max models per locale: {config.max_models}")
    logger.info(f"Results directory: {config.output.results_dir}")
    logger.info(f"Resume using existing outputs: {config.resume}")

    # Track overall results
    overall_results = {}

    # Resolve target inclusion modes
    inclusion_map = {
        "IncTar": [True],
        "include": [True],
        "ExcTar": [False],
        "exclude": [False],
        "both": [False, True]
    }
    include_target_modes = inclusion_map.get(config.target.inclusion, [False])
    logger.info(f"Target inclusion modes: {config.target.inclusion}")

    # Build pass-through args for iterative training
    training_extra_args = [
        "--epochs", str(config.training.epochs),
        "--learning-rate", str(config.training.learning_rate),
        "--batch-size", str(config.training.batch_size),
        "--max-seq-length", str(config.training.max_seq_length),
        "--merge-frequency", str(config.merge.frequency),
        "--top-k", str(config.similarity.top_k),
        "--sinkhorn-iters", str(config.similarity.sinkhorn_iters)
    ]

    # Add optional flags
    if config.enable_wandb:
        training_extra_args.append("--enable-wandb")

    for i, locale in enumerate(locales):
        logger.info(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")

        start_time = time.time()
        results = run_experiment_for_locale(
            locale,
            config.mode,
            training_extra_args,
            config.output.results_dir,
            config.max_models,
            include_target_modes,
            config.similarity.type,
            config.model.models_root,
            config.similarity.top_k,
            config.similarity.sinkhorn_iters,
            config.output.results_dir,
            config.resume,
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
                'similarity_type': config.similarity.type,
                'models_root': config.model.models_root,
                'results': overall_results,
                'config': config.to_dict()
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
    logger.info(f"Successful experiments (all variants): {successful_locales}/{total_locales}")

    total_variants = 0
    successful_variants = 0
    accuracies: List[float] = []

    for result in overall_results.values():
        variant_map = result.get('variants')
        if isinstance(variant_map, dict):
            total_variants += len(variant_map)
            for variant_result in variant_map.values():
                if variant_result.get('success'):
                    successful_variants += 1
                accuracy_val = variant_result.get('accuracy')
                if isinstance(accuracy_val, (int, float)):
                    accuracies.append(accuracy_val)
        else:
            accuracy_val = result.get('accuracy')
            if isinstance(accuracy_val, (int, float)):
                accuracies.append(accuracy_val)

    if total_variants:
        logger.info(f"Successful variant runs: {successful_variants}/{total_variants}")

    if accuracies:
        logger.info(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
        logger.info(f"Mean accuracy: {sum(accuracies)/len(accuracies):.4f}")

    # Save final results with config as dict
    final_results_file = os.path.join(REPO_ROOT, "iterative_large_scale_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
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
