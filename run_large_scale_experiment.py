#!/usr/bin/env python3
"""
Large-scale experiment script to run model merging and evaluation for all locales.
"""

import os
import sys
import pandas as pd
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path

def get_all_locales(csv_path):
    """Extract all unique locales from the CSV file."""
    df = pd.read_csv(csv_path)
    # Filter out 'unknown' locale
    locales = sorted([locale for locale in df['locale'].unique() if locale != 'unknown'])
    return locales

def get_model_for_locale(csv_path, locale):
    """Get the base model and model directory for a specific locale."""
    df = pd.read_csv(csv_path)
    
    # Find the model with '_k_' in the name (k-models)
    k_models = df[df['model_name'].str.contains('_k_', na=False)]
    locale_models = k_models[k_models['locale'] == locale]
    
    if locale_models.empty:
        print(f"Warning: No k-model found for locale {locale}")
        return None, None
    
    # Get the first k-model for this locale
    model_info = locale_models.iloc[0]
    base_model = model_info['model_name']
    
    return base_model, "alpha_0.5_{locale}_epoch-9".format(locale=locale)

def run_merge(mode, target_lang):
    """Run the merging pipeline for a specific mode and target language."""
    print(f"Running {mode} merge for {target_lang}...")

    cmd = [sys.executable, "run_merging_pipeline.py",
           "--mode", mode,
           "--target-lang", target_lang]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {mode} merge completed for {target_lang}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode} merge failed for {target_lang}: {e}")
        return False

def run_evaluation(base_model, model_dir, locale, prefix):
    """Run evaluation for a specific model."""
    print(f"Running evaluation for {base_model} with locale {locale}...")
    
    cmd = [sys.executable, "evaluate_specific_model.py",
           "--base-model", base_model,
           "--locale", locale,
           "--prefix", prefix]
    
    if model_dir:
        cmd.extend(["--model-dir", model_dir])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Evaluation completed for {locale} ({prefix})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for {locale} ({prefix}): {e}")
        return False

def run_experiment_for_locale(locale, skip_baseline=False, skip_similarity=False, skip_average=False, skip_fisher=False, skip_fisher_simple=False):
    """Run complete experiment for a single locale."""
    print(f"\n{'='*60}")
    print(f"Running experiment for locale: {locale}")
    print(f"{'='*60}")
    
    # Get model info for this locale
    base_model, model_dir = get_model_for_locale("haryoaw_massive_models.csv", locale)
    
    if not base_model:
        print(f"Skipping {locale} - no model found")
        return False
    
    results = {}
    
    # 1. Run baseline evaluation
    if not skip_baseline:
        print(f"\n--- Baseline Evaluation for {locale} ---")
        success = run_evaluation(base_model, model_dir, locale, "baseline")
        results['baseline'] = success
    else:
        results['baseline'] = "skipped"
    
    # 2. Run similarity merge and evaluation
    if not skip_similarity:
        print(f"\n--- Similarity Merge for {locale} ---")
        merge_success = run_merge("similarity", locale)
        if merge_success:
            # Evaluate the merged model
            eval_success = run_evaluation("merged_models/similarity_merge_{locale}".format(locale=locale), 
                                       None, locale, "similarity")
            results['similarity'] = eval_success
        else:
            results['similarity'] = False
    else:
        results['similarity'] = "skipped"
    
    # 3. Run average merge and evaluation
    if not skip_average:
        print(f"\n--- Average Merge for {locale} ---")
        merge_success = run_merge("average", locale)
        if merge_success:
            # Evaluate the merged model
            eval_success = run_evaluation("merged_models/average_merge_{locale}".format(locale=locale), 
                                       None, locale, "average")
            results['average'] = eval_success
        else:
            results['average'] = False
    else:
        results['average'] = "skipped"

    # 4. Run Fisher merge and evaluation
    if not skip_fisher:
        print(f"\n--- Fisher Merge for {locale} ---")
        merge_success = run_merge("fisher_simple", locale)  # Use fisher_simple for now
        if merge_success:
            # Evaluate the merged model
            eval_success = run_evaluation("merged_models/fisher_simple_merge_{locale}".format(locale=locale),
                                       None, locale, "fisher_simple")
            results['fisher_simple'] = eval_success
        else:
            results['fisher_simple'] = False
    else:
        results['fisher_simple'] = "skipped"

    return results

def main():
    parser = argparse.ArgumentParser(description="Run large-scale merging experiments")
    parser.add_argument("--locales", nargs="+", default=None, 
                       help="Specific locales to run (default: all locales)")
    parser.add_argument("--skip-baseline", action="store_true", 
                       help="Skip baseline evaluation")
    parser.add_argument("--skip-similarity", action="store_true", 
                       help="Skip similarity merging")
    parser.add_argument("--skip-average", action="store_true",
                       help="Skip average merging")
    parser.add_argument("--skip-fisher", action="store_true",
                       help="Skip Fisher merging")
      parser.add_argument("--skip-fisher-simple", action="store_true",
                       help="Skip simplified Fisher merging")
    parser.add_argument("--fisher-mode", type=str, default="fisher_simple",
                       choices=["fisher", "fisher_simple", "fisher_dataset"],
                       help="Fisher merging mode to use (default: fisher_simple)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       help="Maximum number of locales to process")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    
    args = parser.parse_args()
    
    # Get all locales
    all_locales = get_all_locales("haryoaw_massive_models.csv")
    
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
    print(f"Skip baseline: {args.skip_baseline}")
    print(f"Skip similarity: {args.skip_similarity}")
    print(f"Skip average: {args.skip_average}")
    print(f"Skip Fisher: {args.skip_fisher}")
    print(f"Skip Fisher Simple: {args.skip_fisher_simple}")
    print(f"Start from index: {args.start_from}")
    
    # Track overall results
    overall_results = {}
    
    for i, locale in enumerate(locales):
        print(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")
        
        results = run_experiment_for_locale(
            locale,
            skip_baseline=args.skip_baseline,
            skip_similarity=args.skip_similarity,
            skip_average=args.skip_average,
            skip_fisher=args.skip_fisher,
            skip_fisher_simple=args.skip_fisher_simple
        )
        
        overall_results[locale] = results
        
        # Save progress after each locale
        progress_file = "experiment_progress.json"
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
    successful_baseline = sum(1 for r in overall_results.values() if r.get('baseline') == True)
    successful_similarity = sum(1 for r in overall_results.values() if r.get('similarity') == True)
    successful_average = sum(1 for r in overall_results.values() if r.get('average') == True)
    successful_fisher_simple = sum(1 for r in overall_results.values() if r.get('fisher_simple') == True)

    print(f"Total locales processed: {total_locales}")
    print(f"Successful baseline evaluations: {successful_baseline}")
    print(f"Successful similarity merges: {successful_similarity}")
    print(f"Successful average merges: {successful_average}")
    print(f"Successful Fisher Simple merges: {successful_fisher_simple}")
    
    # Save final results
    final_results_file = "experiment_final_results.json"
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'skip_baseline': args.skip_baseline,
                'skip_similarity': args.skip_similarity,
                'skip_average': args.skip_average,
                'skip_fisher': args.skip_fisher,
                'skip_fisher_simple': args.skip_fisher_simple,
                'start_from': args.start_from,
                'max_locales': args.max_locales
            },
            'summary': {
                'total_locales': total_locales,
                'successful_baseline': successful_baseline,
                'successful_similarity': successful_similarity,
                'successful_average': successful_average,
                'successful_fisher_simple': successful_fisher_simple
            },
            'detailed_results': overall_results
        }, f, indent=2)
    
    print(f"Final results saved to {final_results_file}")

if __name__ == "__main__":
    main()