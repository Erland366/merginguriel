#!/usr/bin/env python3
"""
Extract True Best Zero-Shot Baseline Performance (FIXED)

This script extracts the best zero-shot performance from ONLY the models
that were actually used in the merging process, excluding the native model.
Uses locale codes directly for NxN matrix lookup.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

class TrueBestBaselineExtractorFixed:
    def __init__(self):
        self.nxn_results_path = "nxn_results/nxn_eval_20250915_101911/evaluation_matrix.csv"
        self.merged_models_path = "merged_models"
        self.load_data()

    def load_data(self):
        """Load all necessary data files"""
        try:
            self.nxn_df = pd.read_csv(self.nxn_results_path, index_col=0)
            self.nxn_df = self.nxn_df.dropna()
            print(f"Loaded NxN matrix with {len(self.nxn_df)} source languages and {len(self.nxn_df.columns)} target languages")
            print(f"Available source locales: {list(self.nxn_df.index)[:10]}...")
            print(f"Available target locales: {list(self.nxn_df.columns)[:10]}...")
        except FileNotFoundError:
            print(f"Error: Could not find {self.nxn_results_path}")
            self.nxn_df = None

    def get_models_used_for_merging(self, target_locale, merge_type="average"):
        """Extract the actual models used for merging from merge_details.txt"""
        merge_path = os.path.join(self.merged_models_path, f"{merge_type}_merge_{target_locale}", "merge_details.txt")

        if not os.path.exists(merge_path):
            print(f"Warning: No merge details found for {target_locale} with {merge_type}")
            return []

        with open(merge_path, 'r') as f:
            content = f.read()

        # Extract model locales using regex
        locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})$", re.MULTILINE)
        model_locales = locale_pattern.findall(content)

        return model_locales

    def calculate_true_best_zero_shot(self, target_locale):
        """Calculate the true best zero-shot performance from available merging models"""
        if self.nxn_df is None:
            return None

        # Get models actually used for merging (try average first, then other types)
        merge_types = ["average", "similarity", "fisher_simple"]
        models_used = []

        for merge_type in merge_types:
            models_used = self.get_models_used_for_merging(target_locale, merge_type)
            if models_used:
                print(f"Using {merge_type} merge models for {target_locale}: {models_used}")
                break

        if not models_used:
            print(f"No merge details found for {target_locale}")
            return None

        # Extract zero-shot scores from available merging models (excluding native model)
        zero_shot_scores = []
        source_info = []

        for source_locale in models_used:
            # Skip if this is the native model (not zero-shot)
            if source_locale == target_locale:
                print(f"  Skipping native model {source_locale} for {target_locale}")
                continue

            # Check if both source and target exist in NxN matrix
            if source_locale in self.nxn_df.index and target_locale in self.nxn_df.columns:
                score = self.nxn_df.loc[source_locale, target_locale]
                if pd.notna(score):
                    zero_shot_scores.append(score)
                    source_info.append({
                        'source_locale': source_locale,
                        'zero_shot_score': score
                    })
                    print(f"  Found zero-shot score: {source_locale} → {target_locale} = {score:.4f}")
                else:
                    print(f"  No score found: {source_locale} → {target_locale}")
            else:
                print(f"  Missing in NxN matrix: {source_locale} or {target_locale}")

        if not zero_shot_scores:
            print(f"No zero-shot scores found for {target_locale}")
            return None

        # Find the best performing source model
        best_performance = max(source_info, key=lambda x: x['zero_shot_score'])

        # Calculate statistics
        scores_only = [item['zero_shot_score'] for item in source_info]
        avg_score = np.mean(scores_only)
        std_score = np.std(scores_only)
        median_score = np.median(scores_only)

        result = {
            'target_locale': target_locale,
            'models_used_for_merging': models_used,
            'available_source_models': [item['source_locale'] for item in source_info],
            'best_source_locale': best_performance['source_locale'],
            'best_zero_shot_score': best_performance['zero_shot_score'],
            'average_zero_shot_score': avg_score,
            'std_zero_shot_score': std_score,
            'median_zero_shot_score': median_score,
            'total_available_models': len(source_info),
            'all_zero_shot_scores': source_info
        }

        print(f"{target_locale}: Best zero-shot = {best_performance['zero_shot_score']:.4f} "
              f"from {best_performance['source_locale']} (avg: {avg_score:.4f}, "
              f"models available: {len(source_info)})")

        return result

    def extract_true_baseline_for_all_languages(self):
        """Extract true best zero-shot baseline for all languages that have merge results"""
        if self.nxn_df is None:
            return None

        all_results = []

        # Get all target languages that have merge results
        merged_models_dir = Path(self.merged_models_path)
        if not merged_models_dir.exists():
            print(f"No merged models directory found at {self.merged_models_path}")
            return []

        # Find all locales that have been processed
        processed_locales = set()
        for item in merged_models_dir.iterdir():
            if item.is_dir() and item.name.startswith("average_merge_"):
                locale = item.name.replace("average_merge_", "")
                processed_locales.add(locale)

        print(f"Found {len(processed_locales)} locales with merge results")

        for target_locale in sorted(processed_locales):
            result = self.calculate_true_best_zero_shot(target_locale)
            if result:
                all_results.append(result)

        return all_results

    def create_true_baseline_csv(self, all_results):
        """Create CSV with true baseline information"""
        if not all_results:
            print("No results to process")
            return

        # Create main results DataFrame
        main_data = []
        for result in all_results:
            main_data.append({
                'target_locale': result['target_locale'],
                'models_used_for_merging': ', '.join(result['models_used_for_merging']),
                'available_source_models': ', '.join(result['available_source_models']),
                'best_source_locale': result['best_source_locale'],
                'best_zero_shot_score': result['best_zero_shot_score'],
                'average_zero_shot_score': result['average_zero_shot_score'],
                'std_zero_shot_score': result['std_zero_shot_score'],
                'median_zero_shot_score': result['median_zero_shot_score'],
                'total_available_models': result['total_available_models']
            })

        main_df = pd.DataFrame(main_data)
        main_df = main_df.sort_values('target_locale')

        # Save main CSV
        main_csv_path = "true_best_zero_shot_baseline_fixed.csv"
        main_df.to_csv(main_csv_path, index=False)
        print(f"\nSaved true baseline to: {main_csv_path}")

        # Create detailed CSV with all individual scores
        detailed_data = []
        for result in all_results:
            for score_info in result['all_zero_shot_scores']:
                detailed_data.append({
                    'target_locale': result['target_locale'],
                    'source_locale': score_info['source_locale'],
                    'zero_shot_score': score_info['zero_shot_score'],
                    'is_best': score_info['source_locale'] == result['best_source_locale']
                })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df = detailed_df.sort_values(['target_locale', 'zero_shot_score'], ascending=[True, False])
        detailed_csv_path = "detailed_true_zero_shot_scores_fixed.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Saved detailed scores to: {detailed_csv_path}")

        # Calculate summary statistics
        summary_stats = {
            'total_target_languages': len(main_df),
            'languages_with_available_models': len(main_df[main_df['total_available_models'] > 0]),
            'average_best_zero_shot': main_df['best_zero_shot_score'].mean(),
            'std_best_zero_shot': main_df['best_zero_shot_score'].std(),
            'min_best_zero_shot': main_df['best_zero_shot_score'].min(),
            'max_best_zero_shot': main_df['best_zero_shot_score'].max(),
            'median_best_zero_shot': main_df['best_zero_shot_score'].median(),
            'average_available_models': main_df['total_available_models'].mean()
        }

        print(f"\n=== True Zero-Shot Baseline Statistics ===")
        print(f"Total target languages: {summary_stats['total_target_languages']}")
        print(f"Languages with available models: {summary_stats['languages_with_available_models']}")
        print(f"Average best zero-shot performance: {summary_stats['average_best_zero_shot']:.4f}")
        print(f"Best possible zero-shot: {summary_stats['max_best_zero_shot']:.4f}")
        print(f"Worst best zero-shot: {summary_stats['min_best_zero_shot']:.4f}")
        print(f"Median best zero-shot: {summary_stats['median_best_zero_shot']:.4f}")
        print(f"Average available models per target: {summary_stats['average_available_models']:.1f}")

        # Save summary
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv("true_baseline_summary_statistics_fixed.csv", index=False)
        print(f"Saved summary statistics to: true_baseline_summary_statistics_fixed.csv")

        return main_df, detailed_df, summary_stats

def main():
    print("Extracting True Best Zero-Shot Baseline Performance (FIXED)")
    print("=" * 60)
    print("This calculates the best zero-shot performance from ONLY the models")
    print("that were actually used in the merging process (excluding native model)")
    print("=" * 60)

    extractor = TrueBestBaselineExtractorFixed()

    # Extract true best zero-shot baseline for all languages
    print("\n1. Extracting true best zero-shot baseline...")
    all_results = extractor.extract_true_baseline_for_all_languages()

    if all_results:
        # Create CSV files
        print("\n2. Creating CSV files...")
        main_df, detailed_df, summary_stats = extractor.create_true_baseline_csv(all_results)

        print(f"\nProcessed {len(all_results)} languages successfully")
        print("\nGenerated files:")
        print("- true_best_zero_shot_baseline_fixed.csv (main results)")
        print("- detailed_true_zero_shot_scores_fixed.csv (all source-target pairs)")
        print("- true_baseline_summary_statistics_fixed.csv (summary stats)")

    else:
        print("No results found!")

if __name__ == "__main__":
    main()