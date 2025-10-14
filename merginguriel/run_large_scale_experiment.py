#!/usr/bin/env python3
# Moved into merginguriel/ package
"""
Large-scale experiment runner that orchestrates merging + evaluation across locales.

Refactor highlights:
- Specify what to run via --modes (e.g., baseline similarity average fisher_dataset),
  instead of skip flags.
- Uses run_merging_pipeline_refactored.py and passes through relevant options.
- Extensible: discovers available merge methods; adding a new method shouldn't require
  editing this script—just include it in --modes.
"""

import os
import sys
import pandas as pd
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Ensure project root on sys.path for method discovery
# When this script is inside merginguriel/, repo root is one level up
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBMODULES_DIR = os.path.join(REPO_ROOT, "submodules", "auto_merge_llm")
if SUBMODULES_DIR not in sys.path:
    sys.path.insert(0, SUBMODULES_DIR)

try:
    from auto_merge_llm.methods import merging_methods_dict
except Exception:
    merging_methods_dict = {}

def get_all_locales_from_similarity_matrix():
    """Extract all unique locales from the similarity matrix."""
    similarity_matrix_path = os.path.join(REPO_ROOT, "sparsed_language_similarity_matrix_unified.csv")
    df = pd.read_csv(similarity_matrix_path, index_col=0)
    # Return all locale codes from the index (and columns, they should be the same)
    locales = sorted(df.index.tolist())
    return locales

def get_model_for_locale(locale):
    """Get the model path for a specific locale using consolidated directory structure."""
    # Use the consolidated model directory structure
    model_path = os.path.join(REPO_ROOT, f"haryos_model/xlm-roberta-base_massive_k_{locale}")

    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"Warning: Model directory not found for locale {locale}: {model_path}")
        return None

    return model_path

def run_merge(mode: str, target_lang: str, extra_args: List[str]) -> bool:
    """Run the merging pipeline for a specific mode and target language."""
    print(f"Running {mode} merge for {target_lang}...")

    cmd = [sys.executable, os.path.join(REPO_ROOT, "merginguriel", "run_merging_pipeline_refactored.py"),
           "--mode", mode,
           "--target-lang", target_lang]
    if extra_args:
        cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {mode} merge completed for {target_lang}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode} merge failed for {target_lang}: {e}")
        return False

def run_evaluation(model_path, locale, prefix):
    """Run evaluation for a specific model using the consolidated model path."""
    print(f"Running evaluation for {model_path} with locale {locale}...")

    cmd = [sys.executable, os.path.join(REPO_ROOT, "merginguriel", "evaluate_specific_model.py"),
           "--base-model", model_path,
           "--locale", locale,
           "--prefix", prefix]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Evaluation completed for {locale} ({prefix})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for {locale} ({prefix}): {e}")
        return False

def run_experiment_for_locale(
    locale: str,
    modes: List[str],
    merge_extra_args: List[str],
):
    """Run the requested experiment modes for a single locale."""
    print(f"\n{'='*60}")
    print(f"Running experiment for locale: {locale}")
    print(f"{'='*60}")

    # Get model path for this locale
    base_model_path = get_model_for_locale(locale)

    if not base_model_path:
        print(f"Skipping {locale} - no model found")
        return False

    results: Dict[str, bool] = {}

    for mode in modes:
        if mode == "baseline":
            print(f"\n--- Baseline Evaluation for {locale} ---")
            success = run_evaluation(base_model_path, locale, "baseline")
            results['baseline'] = success
            continue

        print(f"\n--- {mode} Merge for {locale} ---")
        merge_success = run_merge(mode, locale, merge_extra_args)
        if merge_success:
            merged_model_path = os.path.join(REPO_ROOT, "merged_models", f"{mode}_merge_{locale}")
            eval_success = run_evaluation(merged_model_path, locale, mode)
            results[mode] = eval_success
        else:
            results[mode] = False

    return results

def main():
    parser = argparse.ArgumentParser(description="Run large-scale merging experiments")
    parser.add_argument("--locales", nargs="+", default=None,
                       help="Specific locales to run (default: all locales)")
    parser.add_argument("--modes", nargs="+",
                       default=["baseline", "similarity", "average", "fisher", "ties", "task_arithmetic", "slerp", "regmean", "linear"],
                       help="Which modes to run per locale (include 'baseline' to evaluate base model)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       help="Maximum number of locales to process")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    parser.add_argument("--list-modes", action="store_true",
                       help="List available merge method keys and exit")

    # Pass-through options for run_merging_pipeline_refactored.py
    parser.add_argument("--num-languages", type=int, default=5,
                       help="Top-K languages to include in merges that auto-select sources")
    parser.add_argument("--similarity-source", type=str, choices=["sparse","dense"], default="dense",
                       help="Use precomputed sparse CSV or compute dense similarities on-the-fly")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Top-K neighbors per language for on-the-fly similarity")
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       help="Sinkhorn iterations for on-the-fly similarity")
    parser.add_argument("--dataset-name", type=str, default="AmazonScience/massive",
                       help="Dataset name for dataset-enabled Fisher")
    parser.add_argument("--dataset-split", type=str, default="train",
                       help="Dataset split for Fisher ('train' or 'validation')")
    parser.add_argument("--text-column", type=str, default="utt",
                       help="Text column to use (MASSIVE uses 'utt')")
    parser.add_argument("--num-fisher-examples", type=int, default=1000,
                       help="Total examples for Fisher computation")
    parser.add_argument("--fisher-data-mode", type=str, choices=["target","sources","both"], default="target",
                       help="Distribution to compute Fisher on")
    parser.add_argument("--preweight", type=str, choices=["equal","uriel"], default="equal",
                       help="Pre-weighting for models before Fisher")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for Fisher computation")
    parser.add_argument("--max-seq-length", type=int, default=128,
                       help="Max sequence length for Fisher tokenization")
    parser.add_argument("--preset", type=str, choices=["none", "fairness", "target"], default="none",
                        help="Convenience presets for Fisher config: fairness = sources-only + equal preweights; target = target-only + URIEL preweights")
    
    args = parser.parse_args()
    
    all_locales = get_all_locales_from_similarity_matrix()

    if args.list_modes:
        if merging_methods_dict:
            print("Available merge method keys:")
            for k in sorted(merging_methods_dict.keys()):
                print(f"  - {k}")
        else:
            print("Could not import merging_methods_dict; ensure submodule path is correct.")
        return

    if args.list_locales:
        print("Available locales:")
        for i, locale in enumerate(all_locales):
            print(f"  {i:2d}. {locale}")
        print(f"\nTotal: {len(all_locales)} locales")
        return
    
    # Filter locales
    if args.locales:
        locales = [loc for loc in all_locales if loc in args.locales]
    else:
        locales = all_locales
    
    # Apply start index and max limit
    if args.start_from > 0:
        locales = locales[args.start_from:]
    if args.max_locales:
        locales = locales[:args.max_locales]
    
    print(f"Starting experiment for {len(locales)} locales")
    print(f"Modes: {args.modes}")
    print(f"Start from index: {args.start_from}")
    
    # Track overall results
    overall_results = {}
    
    # Apply preset defaults if requested (explicit CLI flags override presets)
    argv = sys.argv[1:]
    if args.preset != "none":
        if args.preset == "fairness":
            if "--fisher-data-mode" not in argv:
                args.fisher_data_mode = "sources"
            if "--preweight" not in argv:
                args.preweight = "equal"
        elif args.preset == "target":
            if "--fisher-data-mode" not in argv:
                args.fisher_data_mode = "target"
            if "--preweight" not in argv:
                args.preweight = "uriel"

    # Build pass-through args once
    merge_extra_args = [
        "--num-languages", str(args.num_languages),
        "--similarity-source", args.similarity_source,
        "--top-k", str(args.top_k),
        "--sinkhorn-iters", str(args.sinkhorn_iters),
        "--dataset-name", args.dataset_name,
        "--dataset-split", args.dataset_split,
        "--text-column", args.text_column,
        "--num-fisher-examples", str(args.num_fisher_examples),
        "--fisher-data-mode", args.fisher_data_mode,
        "--preweight", args.preweight,
        "--batch-size", str(args.batch_size),
        "--max-seq-length", str(args.max_seq_length),
    ]

    for i, locale in enumerate(locales):
        print(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")
        results = run_experiment_for_locale(locale, args.modes, merge_extra_args)
        
        overall_results[locale] = results
        
        # Save progress after each locale
        progress_file = os.path.join(REPO_ROOT, "experiment_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_locales': len(all_locales),
                'processed_locales': i + 1,
                'current_locale': locale,
                'results': overall_results
            }, f, indent=2)
        
        print(f"Progress saved to {progress_file}")
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")
    
    # Summary
    total_locales = len(overall_results)
    print(f"Total locales processed: {len(overall_results)}")
    # Dynamic summary per mode
    mode_success_counts: Dict[str, int] = {}
    for locale, res in overall_results.items():
        for mode, ok in res.items():
            if ok is True:
                mode_success_counts[mode] = mode_success_counts.get(mode, 0) + 1
    if mode_success_counts:
        print("Successful runs per mode:")
        for mode, cnt in sorted(mode_success_counts.items()):
            print(f"  - {mode}: {cnt}")
    
    # Save final results
    final_results_file = os.path.join(REPO_ROOT, "experiment_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'modes': args.modes,
                'start_from': args.start_from,
                'max_locales': args.max_locales,
                'num_languages': args.num_languages,
                'dataset_name': args.dataset_name,
                'dataset_split': args.dataset_split,
                'text_column': args.text_column,
                'num_fisher_examples': args.num_fisher_examples,
                'fisher_data_mode': args.fisher_data_mode,
                'preweight': args.preweight,
                'batch_size': args.batch_size,
                'max_seq_length': args.max_seq_length,
            },
            'summary': {
                'total_locales': len(overall_results),
                'mode_success_counts': mode_success_counts
            },
            'detailed_results': overall_results
        }, f, indent=2)
    
    print(f"Final results saved to {final_results_file}")

if __name__ == "__main__":
    main()
