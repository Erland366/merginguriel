#!/usr/bin/env python3
"""
Extract Best Possible Baseline Performance from NxN Matrix

This script extracts the best zero-shot performance for each target language
from all available source languages, representing the theoretical maximum
performance you could achieve by picking the best single model for each language.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BestBaselineExtractor:
    def __init__(self):
        self.nxn_results_path = "nxn_results/nxn_eval_20250915_101911/evaluation_matrix.csv"
        self.load_data()

    def load_data(self):
        """Load the NxN evaluation matrix"""
        try:
            self.nxn_df = pd.read_csv(self.nxn_results_path, index_col=0)
            self.nxn_df = self.nxn_df.dropna()
            print(f"Loaded NxN matrix with {len(self.nxn_df)} source languages and {len(self.nxn_df.columns)} target languages")
            print(f"Available source languages: {list(self.nxn_df.index)}")
            print(f"Available target languages: {list(self.nxn_df.columns)}")
        except FileNotFoundError:
            print(f"Error: Could not find {self.nxn_results_path}")
            self.nxn_df = None

    def extract_best_baseline_for_all_languages(self):
        """Extract best zero-shot performance for all target languages"""
        if self.nxn_df is None:
            return None

        best_baseline_data = []

        for target_locale in self.nxn_df.columns:
            # Get all zero-shot performances for this target from all source languages
            zero_shot_scores = []

            for source_locale in self.nxn_df.index:
                score = self.nxn_df.loc[source_locale, target_locale]
                if pd.notna(score):
                    zero_shot_scores.append({
                        'source_locale': source_locale,
                        'target_locale': target_locale,
                        'zero_shot_score': score
                    })

            if zero_shot_scores:
                # Find the best performing source language
                best_performance = max(zero_shot_scores, key=lambda x: x['zero_shot_score'])

                # Calculate statistics
                scores_only = [item['zero_shot_score'] for item in zero_shot_scores]
                avg_score = np.mean(scores_only)
                std_score = np.std(scores_only)
                median_score = np.median(scores_only)

                best_baseline_data.append({
                    'target_locale': target_locale,
                    'best_source_locale': best_performance['source_locale'],
                    'best_zero_shot_score': best_performance['zero_shot_score'],
                    'average_zero_shot_score': avg_score,
                    'std_zero_shot_score': std_score,
                    'median_zero_shot_score': median_score,
                    'total_source_languages': len(zero_shot_scores),
                    'num_available_models': len(zero_shot_scores)
                })

                print(f"{target_locale}: Best = {best_performance['zero_shot_score']:.4f} "
                      f"from {best_performance['source_locale']} (avg: {avg_score:.4f})")

        return best_baseline_data

    def create_comprehensive_baseline_csv(self, best_baseline_data):
        """Create a comprehensive CSV with all baseline information"""
        if not best_baseline_data:
            print("No data to process")
            return

        # Convert to DataFrame
        df = pd.DataFrame(best_baseline_data)

        # Sort by target locale
        df = df.sort_values('target_locale')

        # Save main baseline CSV
        baseline_csv_path = "best_baseline_performance.csv"
        df.to_csv(baseline_csv_path, index=False)
        print(f"\nSaved baseline performance to: {baseline_csv_path}")

        # Create a more detailed version with all source-target pairs
        detailed_data = []

        for target_locale in self.nxn_df.columns:
            for source_locale in self.nxn_df.index:
                score = self.nxn_df.loc[source_locale, target_locale]
                if pd.notna(score):
                    detailed_data.append({
                        'source_locale': source_locale,
                        'target_locale': target_locale,
                        'zero_shot_score': score
                    })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df = detailed_df.sort_values(['target_locale', 'zero_shot_score'], ascending=[True, False])
        detailed_csv_path = "detailed_zero_shot_matrix.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Saved detailed zero-shot matrix to: {detailed_csv_path}")

        # Create summary statistics
        summary_stats = {
            'total_target_languages': len(df),
            'target_languages_with_models': len(df[df['num_available_models'] > 0]),
            'average_best_performance': df['best_zero_shot_score'].mean(),
            'std_best_performance': df['best_zero_shot_score'].std(),
            'min_best_performance': df['best_zero_shot_score'].min(),
            'max_best_performance': df['best_zero_shot_score'].max(),
            'median_best_performance': df['best_zero_shot_score'].median()
        }

        print(f"\n=== Summary Statistics ===")
        print(f"Total target languages: {summary_stats['total_target_languages']}")
        print(f"Languages with available models: {summary_stats['target_languages_with_models']}")
        print(f"Average best performance: {summary_stats['average_best_performance']:.4f}")
        print(f"Best possible performance: {summary_stats['max_best_performance']:.4f}")
        print(f"Worst best performance: {summary_stats['min_best_performance']:.4f}")
        print(f"Median best performance: {summary_stats['median_best_performance']:.4f}")

        # Save summary
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv("baseline_summary_statistics.csv", index=False)
        print(f"Saved summary statistics to: baseline_summary_statistics.csv")

        return df, detailed_df, summary_stats

    def get_best_baseline_for_specific_languages(self, target_languages):
        """Get best baseline for specific list of languages"""
        if self.nxn_df is None:
            return None

        results = []
        for target_locale in target_languages:
            if target_locale not in self.nxn_df.columns:
                print(f"Warning: {target_locale} not found in target languages")
                results.append({
                    'target_locale': target_locale,
                    'best_source_locale': None,
                    'best_zero_shot_score': None,
                    'status': 'not_found'
                })
                continue

            zero_shot_scores = []
            for source_locale in self.nxn_df.index:
                score = self.nxn_df.loc[source_locale, target_locale]
                if pd.notna(score):
                    zero_shot_scores.append({
                        'source_locale': source_locale,
                        'zero_shot_score': score
                    })

            if zero_shot_scores:
                best_performance = max(zero_shot_scores, key=lambda x: x['zero_shot_score'])
                results.append({
                    'target_locale': target_locale,
                    'best_source_locale': best_performance['source_locale'],
                    'best_zero_shot_score': best_performance['zero_shot_score'],
                    'status': 'found'
                })
            else:
                results.append({
                    'target_locale': target_locale,
                    'best_source_locale': None,
                    'best_zero_shot_score': None,
                    'status': 'no_data'
                })

        return pd.DataFrame(results)

def main():
    print("Extracting Best Possible Baseline Performance")
    print("=" * 50)

    extractor = BestBaselineExtractor()

    # Extract best baseline for all languages
    print("\n1. Extracting best baseline for all languages...")
    best_baseline_data = extractor.extract_best_baseline_for_all_languages()

    if best_baseline_data:
        # Create comprehensive CSV files
        print("\n2. Creating CSV files...")
        baseline_df, detailed_df, summary_stats = extractor.create_comprehensive_baseline_csv(best_baseline_data)

        # Example: Get baseline for specific languages (like the ones in your experiment)
        print("\n3. Example: Extracting for experimental languages...")
        experimental_languages = [
            'af-ZA', 'am-ET', 'ar-SA', 'az-AZ', 'bn-BD', 'ca-ES', 'cy-GB',
            'da-DK', 'de-DE', 'el-GR', 'en-US', 'es-ES', 'et-EE', 'fa-IR',
            'fi-FI', 'fr-FR', 'he-IL', 'hi-IN', 'hu-HU', 'id-ID', 'is-IS',
            'it-IT', 'ja-JP', 'jv-ID', 'ka-GE', 'km-KH', 'kn-IN', 'ko-KR',
            'lt-LT', 'lv-LV', 'ml-IN', 'mn-MN', 'mr-IN', 'ms-MY', 'my-MM',
            'nb-NO', 'nl-NL', 'pa-IN', 'pl-PL', 'pt-PT', 'ro-RO', 'ru-RU',
            'sl-SL', 'sq-AL', 'sv-SE', 'sw-KE', 'ta-IN', 'te-IN', 'th-TH',
            'tl-PH', 'tr-TR', 'uk-UA', 'ur-PK', 'vi-VN', 'yo-NG', 'zh-CN',
            'zh-TW', 'zu-ZA'
        ]

        experimental_baseline = extractor.get_best_baseline_for_specific_languages(experimental_languages)
        if experimental_baseline is not None:
            experimental_baseline.to_csv("experimental_best_baseline.csv", index=False)
            print(f"Saved experimental languages baseline to: experimental_best_baseline.csv")

            # Show some examples
            print("\n4. Sample results for experimental languages:")
            sample_languages = ['af-ZA', 'ar-SA', 'de-DE', 'fr-FR', 'zh-CN']
            for _, row in experimental_baseline[experimental_baseline['target_locale'].isin(sample_languages)].iterrows():
                if row['status'] == 'found':
                    print(f"  {row['target_locale']}: {row['best_zero_shot_score']:.4f} "
                          f"(from {row['best_source_locale']})")
                else:
                    print(f"  {row['target_locale']}: {row['status']}")

    print("\nExtraction complete!")
    print("Generated files:")
    print("- best_baseline_performance.csv (main results)")
    print("- detailed_zero_shot_matrix.csv (all source-target pairs)")
    print("- baseline_summary_statistics.csv (summary stats)")
    print("- experimental_best_baseline.csv (for your experimental languages)")

if __name__ == "__main__":
    main()