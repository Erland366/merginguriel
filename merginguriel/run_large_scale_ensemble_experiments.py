#!/usr/bin/env python3
"""
Large-scale ensemble inference experiment runner that orchestrates ensemble evaluation across locales.

This script mirrors run_large_scale_experiment.py but focuses on ensemble inference methods
instead of model merging. It runs multiple voting methods across multiple target languages
and generates results compatible with the existing aggregation system.

Key features:
- Supports multiple voting methods: majority, weighted_majority, soft, uriel_logits
- Progress tracking and resumption capability
- Results compatible with aggregate_results.py
- Leverages existing similarity processing and model discovery
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
import shutil

# Ensure project root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def get_all_locales_from_similarity_matrix():
    """Extract all unique locales from the similarity matrix."""
    similarity_matrix_path = os.path.join(REPO_ROOT, "sparsed_language_similarity_matrix_unified.csv")
    df = pd.read_csv(similarity_matrix_path, index_col=0)
    locales = sorted(df.index.tolist())
    return locales

def get_model_for_locale(locale):
    """Get the model path for a specific locale using consolidated directory structure."""
    model_path = os.path.join(REPO_ROOT, f"haryos_model/xlm-roberta-base_massive_k_{locale}")
    if not os.path.exists(model_path):
        print(f"Warning: Model directory not found for locale {locale}: {model_path}")
        return None
    return model_path

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

def run_ensemble_inference(voting_method: str, target_lang: str, extra_args: List[str]) -> bool:
    """Run ensemble inference for a specific voting method and target language."""
    print(f"Running {voting_method} ensemble for {target_lang}...")

    cmd = [sys.executable, os.path.join(REPO_ROOT, "merginguriel", "uriel_ensemble_inference.py"),
           "--target-lang", target_lang,
           "--voting-method", voting_method]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {voting_method} ensemble completed for {target_lang}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {voting_method} ensemble failed for {target_lang}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def save_ensemble_results(target_lang: str, voting_method: str, ensemble_data: Dict,
                        results_dir: str = "results") -> bool:
    """Save ensemble results in format compatible with aggregate_results.py."""
    try:
        # Create folder name in expected format
        folder_name = f"ensemble_{voting_method}_{target_lang}"
        results_folder = os.path.join(REPO_ROOT, results_dir, folder_name)
        os.makedirs(results_folder, exist_ok=True)

        # Convert ensemble data to results.json format
        results = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": f"ensemble_{voting_method}_{target_lang}",
                "locale": target_lang,
                "massive_locale": target_lang,
                "dataset": "AmazonScience/massive",
                "split": "test",
                "experiment_type": f"ensemble_{voting_method}",
                "voting_method": voting_method
            },
            "model_info": {
                "num_models": ensemble_data.get("experiment_info", {}).get("num_models", 0),
                "model_names": ensemble_data.get("metadata", {}).get("model_names", []),
                "ensemble_method": voting_method,
                "similarity_weights": ensemble_data.get("models", {})
            },
            "dataset_info": {
                "total_examples": ensemble_data.get("experiment_info", {}).get("num_examples", 0),
                "voting_method": voting_method
            },
            "performance": ensemble_data.get("performance", {}),
            "metadata": ensemble_data.get("metadata", {}),
            "examples": ensemble_data.get("examples", [])
        }

        # Save results.json
        results_file = os.path.join(results_folder, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {results_file}")
        return True

    except Exception as e:
        print(f"✗ Failed to save ensemble results for {target_lang} ({voting_method}): {e}")
        return False

def load_ensemble_results(target_lang: str, voting_method: str, output_dir: str = "urie_ensemble_results") -> Dict:
    """Load ensemble results from the output directory."""
    # Expected folder structure from uriel_ensemble_inference.py
    source_folder = os.path.join(REPO_ROOT, output_dir, f"urie_{voting_method}_{target_lang}")
    results_file = os.path.join(source_folder, "results.json")

    if not os.path.exists(results_file):
        print(f"Warning: Ensemble results not found at {results_file}")
        return {}

    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ensemble results: {e}")
        return {}

def run_experiment_for_locale(
    locale: str,
    voting_methods: List[str],
    ensemble_extra_args: List[str],
):
    """Run the requested ensemble experiments for a single locale."""
    print(f"\n{'='*60}")
    print(f"Running ensemble experiments for locale: {locale}")
    print(f"{'='*60}")

    # Check if model exists for this locale
    base_model_path = get_model_for_locale(locale)
    if not base_model_path:
        print(f"Skipping {locale} - no model found")
        return {}

    results: Dict[str, bool] = {}

    for voting_method in voting_methods:
        print(f"\n--- {voting_method} Ensemble for {locale} ---")

        # Run ensemble inference
        ensemble_success = run_ensemble_inference(voting_method, locale, ensemble_extra_args)

        if ensemble_success:
            # Load results and save in compatible format
            ensemble_data = load_ensemble_results(locale, voting_method)
            if ensemble_data:
                save_success = save_ensemble_results(locale, voting_method, ensemble_data)
                results[voting_method] = save_success
            else:
                results[voting_method] = False
        else:
            results[voting_method] = False

    return results

def main():
    parser = argparse.ArgumentParser(description="Run large-scale ensemble inference experiments")
    parser.add_argument("--target-languages", nargs="+", default=None,
                       help="Specific target languages to run (default: all available locales)")
    parser.add_argument("--voting-methods", nargs="+",
                       default=["majority", "weighted_majority", "soft", "uriel_logits"],
                       help="Voting methods to test for each locale")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       help="Maximum number of locales to process")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    parser.add_argument("--list-voting-methods", action="store_true",
                       help="List available voting methods and exit")

    # Pass-through options for uriel_ensemble_inference.py
    parser.add_argument("--num-languages", type=int, default=5,
                       help="Number of models to include in ensemble")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of test examples to evaluate (default: all available examples)")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Number of top similar languages to consider")
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       help="Number of Sinkhorn normalization iterations")
    parser.add_argument("--output-dir", type=str, default="urie_ensemble_results",
                       help="Output directory for ensemble results")

    args = parser.parse_args()

    # Get all available locales (those with models)
    available_locales = get_available_locales()

    if args.list_voting_methods:
        print("Available voting methods:")
        for method in args.voting_methods:
            print(f"  - {method}")
        return

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

    print(f"Starting ensemble experiment for {len(locales)} locales")
    print(f"Voting methods: {args.voting_methods}")
    print(f"Start from index: {args.start_from}")

    # Track overall results
    overall_results = {}

    # Build pass-through args for ensemble inference
    ensemble_extra_args = [
        "--num-languages", str(args.num_languages),
        "--top-k", str(args.top_k),
        "--sinkhorn-iters", str(args.sinkhorn_iters),
        "--output-dir", args.output_dir
    ]

    # Only add num-examples if specified (for full test set evaluation)
    if args.num_examples is not None:
        ensemble_extra_args.extend(["--num-examples", str(args.num_examples)])

    for i, locale in enumerate(locales):
        print(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")
        results = run_experiment_for_locale(locale, args.voting_methods, ensemble_extra_args)

        overall_results[locale] = results

        # Save progress after each locale
        progress_file = os.path.join(REPO_ROOT, "ensemble_experiment_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_locales': len(available_locales),
                'processed_locales': i + 1,
                'current_locale': locale,
                'results': overall_results
            }, f, indent=2)

        print(f"Progress saved to {progress_file}")

    print(f"\n{'='*60}")
    print("Ensemble experiment completed!")
    print(f"{'='*60}")

    # Summary
    total_locales = len(overall_results)
    print(f"Total locales processed: {len(overall_results)}")

    # Dynamic summary per voting method
    method_success_counts: Dict[str, int] = {}
    for locale, res in overall_results.items():
        for method, ok in res.items():
            if ok is True:
                method_success_counts[method] = method_success_counts.get(method, 0) + 1

    if method_success_counts:
        print("Successful runs per voting method:")
        for method, cnt in sorted(method_success_counts.items()):
            print(f"  - {method}: {cnt}")

    # Save final results
    final_results_file = os.path.join(REPO_ROOT, "ensemble_experiment_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'voting_methods': args.voting_methods,
                'start_from': args.start_from,
                'max_locales': args.max_locales,
                'num_languages': args.num_languages,
                'num_examples': args.num_examples,
                'top_k': args.top_k,
                'sinkhorn_iters': args.sinkhorn_iters,
                'output_dir': args.output_dir
            },
            'summary': {
                'total_locales': len(overall_results),
                'method_success_counts': method_success_counts
            },
            'detailed_results': overall_results
        }, f, indent=2)

    print(f"Final results saved to {final_results_file}")

if __name__ == "__main__":
    main()