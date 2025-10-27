#!/usr/bin/env python3
"""
NxN cross-lingual evaluation that evaluates each local language model against all other locales.
Models are discovered under a local root (default: ./haryos_model) using the
folder naming pattern: xlm-roberta-base_massive_k_{locale} (e.g., ..._fr-FR).

Outputs a cross-lingual accuracy matrix and summary stats.
"""

import os
import sys
import pandas as pd
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from merginguriel import logger
from tqdm import tqdm

# Import evaluation function (supports local paths)
from merginguriel.evaluate_specific_model import evaluate_specific_model


def discover_locales(models_root: str) -> list[str]:
    """Scan the models root for locale-named folders and return sorted MASSIVE locales."""
    root = Path(models_root)
    if not root.exists():
        logger.warning(f"Models root does not exist: {models_root}")
        return []
    locales = []
    prefix = "xlm-roberta-base_massive_k_"
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith(prefix):
            locale = entry.name[len(prefix):]
            if isinstance(locale, str) and len(locale) >= 4 and "-" in locale:
                locales.append(locale)
    return sorted(locales)

def get_local_models_for_locales(locales: list[str], models_root: str) -> dict:
    """Resolve local model paths under haryos_model for given locales."""
    result = {}
    for locale in locales:
        model_path = os.path.join(models_root, f"xlm-roberta-base_massive_k_{locale}")
        if os.path.isdir(model_path):
            result[locale] = {
                "locale": locale,
                "path": model_path,
            }
        else:
            logger.warning(f"Missing local model for {locale}: {model_path}")
    return result

def evaluate_single_model_target(model_info, target_locale, results_base_dir):
    """Evaluate a single model on a target locale."""
    model_locale = model_info['locale']
    model_name = model_info['path']

    # Create results folder for this combination
    eval_folder = os.path.join(results_base_dir, f"{model_locale}_on_{target_locale}")
    os.makedirs(eval_folder, exist_ok=True)

    try:
        logger.info(f"Evaluating {model_locale} model on {target_locale}...")

        results = evaluate_specific_model(
            model_name=model_name,
            locale=target_locale,
            eval_folder=eval_folder,
        )

        if results:
            # Save results
            results_path = os.path.join(eval_folder, "results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Extract key metrics
            accuracy = results['performance']['accuracy']
            correct = results['performance']['correct_predictions']
            total = results['performance']['total_predictions']

            logger.info(f"✓ {model_locale} → {target_locale}: {accuracy:.4f} ({correct}/{total})")

            return {
                "model_locale": model_locale,
                "target_locale": target_locale,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results_path': results_path
            }
        else:
            logger.error(f"✗ {model_locale} → {target_locale}: Evaluation failed")
            return {
                "model_locale": model_locale,
                "target_locale": target_locale,
                'accuracy': None,
                'correct': None,
                'total': None,
                'results_path': None,
                'error': 'Evaluation failed'
            }

    except Exception as e:
        logger.error(f"✗ {model_locale} → {target_locale}: {str(e)}")
        return {
            "model_locale": model_locale,
            "target_locale": target_locale,
            'accuracy': None,
            'correct': None,
            'total': None,
            'results_path': None,
            'error': str(e)
        }

def run_nxn_evaluation(models_root: str, locales=None, max_workers=4, results_dir="nxn_results"):
    """Run NxN cross-lingual evaluation."""

    # Discover locales from disk and optionally filter
    discovered = discover_locales(models_root)
    if locales:
        all_locales = [loc for loc in discovered if loc in locales]
    else:
        all_locales = discovered

    # Resolve model directories
    models = get_local_models_for_locales(all_locales, models_root)

    print(f"Found {len(models)} models for {len(all_locales)} locales")
    print(f"Models: {list(models.keys())}")
    print(f"Target locales: {all_locales}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = os.path.join(results_dir, f"nxn_eval_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)

    # Prepare evaluation tasks
    evaluation_tasks = []
    for model_locale, model_info in models.items():
        for target_locale in all_locales:
            task_model_info = model_info.copy()
            task_model_info['locale'] = model_locale
            evaluation_tasks.append((task_model_info, target_locale, results_base_dir))

    total_tasks = len(evaluation_tasks)
    print(f"Total evaluation tasks: {total_tasks}")

    # Run evaluations
    results = []
    if max_workers == 1:
        # Sequential execution
        for task_model_info, target_locale, results_base_dir in tqdm(evaluation_tasks, desc="Evaluating"):
            result = evaluate_single_model_target(task_model_info, target_locale, results_base_dir)
            results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(evaluate_single_model_target, task_model_info, target_locale, results_base_dir): (task_model_info['locale'], target_locale)
                for task_model_info, target_locale, results_base_dir in evaluation_tasks
            }

            # Process completed tasks with progress bar
            with tqdm(total=total_tasks, desc="Evaluating") as pbar:
                for future in as_completed(future_to_task):
                    model_locale, target_locale = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task {model_locale} → {target_locale} failed: {e}")
                        results.append({
                            "model_locale": model_locale,
                            "target_locale": target_locale,
                            'accuracy': None,
                            'correct': None,
                            'total': None,
                            'results_path': None,
                            'error': str(e)
                        })
                    finally:
                        pbar.update(1)

    # Create evaluation matrix
    create_evaluation_matrix(results, all_locales, results_base_dir)

    return results, results_base_dir

def create_evaluation_matrix(results, locales, results_base_dir):
    """Create and save evaluation matrix."""

    # Initialize matrix with NaN values
    matrix = pd.DataFrame(index=locales, columns=locales, dtype=float)

    # Fill matrix with results
    for result in results:
        model_locale = result['model_locale']
        target_locale = result['target_locale']
        accuracy = result['accuracy']

        if accuracy is not None:
            matrix.loc[model_locale, target_locale] = accuracy
        else:
            matrix.loc[model_locale, target_locale] = float('nan')

    # Save matrix
    matrix_path = os.path.join(results_base_dir, "evaluation_matrix.csv")
    matrix.to_csv(matrix_path)
    print(f"✓ Evaluation matrix saved to: {matrix_path}")

    # Create summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'locales': locales,
        'total_evaluations': len(results),
        'successful_evaluations': len([r for r in results if r['accuracy'] is not None]),
        'failed_evaluations': len([r for r in results if r['accuracy'] is None]),
        'matrix_stats': {
            'mean_accuracy': matrix.stack().mean(),
            'median_accuracy': matrix.stack().median(),
            'std_accuracy': matrix.stack().std(),
            'min_accuracy': matrix.stack().min(),
            'max_accuracy': matrix.stack().max()
        }
    }

    # Save summary
    summary_path = os.path.join(results_base_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Evaluation summary saved to: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("NxN Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Successful: {summary['successful_evaluations']}")
    print(f"Failed: {summary['failed_evaluations']}")
    print(f"Mean accuracy: {summary['matrix_stats']['mean_accuracy']:.4f}")
    print(f"Median accuracy: {summary['matrix_stats']['median_accuracy']:.4f}")
    print(f"Std accuracy: {summary['matrix_stats']['std_accuracy']:.4f}")

    # Print matrix
    print(f"\n{'='*60}")
    print("Evaluation Matrix (Accuracy)")
    print(f"{'='*60}")
    print(matrix.to_string(float_format="%.4f", na_rep="N/A"))

    return matrix

def main():
    parser = argparse.ArgumentParser(description="Run NxN cross-lingual evaluation (local models)")
    parser.add_argument("--models-root", type=str, default="haryos_model",
                       help="Root folder containing local models (default: haryos_model)")
    parser.add_argument("--locales", nargs="+", default=None,
                       help="Specific locales to evaluate (default: all locales)")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1 - sequential)")
    parser.add_argument("--results-dir", type=str, default="nxn_results",
                       help="Base directory for results (default: nxn_results)")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")

    args = parser.parse_args()

    # List locales if requested
    if args.list_locales:
        locales = discover_locales(args.models_root)
        print("Available locales:")
        for i, locale in enumerate(locales):
            print(f"  {i:2d}. {locale}")
        print(f"\nTotal: {len(locales)} locales")
        return

    print(f"Starting NxN cross-lingual evaluation...")
    print(f"Models root: {args.models_root}")
    print(f"Max workers: {args.max_workers}")
    print(f"Results directory: {args.results_dir}")
    if args.locales:
        print(f"Locales: {args.locales}")

    # Run evaluation
    results, results_dir = run_nxn_evaluation(
        models_root=args.models_root,
        locales=args.locales,
        max_workers=args.max_workers,
        results_dir=args.results_dir
    )

    print(f"\n🎉 NxN evaluation completed!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
