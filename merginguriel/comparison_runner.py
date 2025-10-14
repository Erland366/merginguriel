#!/usr/bin/env python3
"""
Ensemble Methods Comparison Runner

This script runs comprehensive comparisons between different ensemble methods
including traditional voting approaches and the URIEL-guided logits weighting.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from ensemble_runner import run_single_ensemble_experiment

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_comprehensive_comparison(target_languages: List[str],
                                voting_methods: List[str],
                                num_examples: int = 50,
                                num_languages: int = 5,
                                output_dir: str = "comparison_results") -> Dict[str, Any]:
    """
    Run comprehensive comparison across multiple target languages and voting methods.

    Args:
        target_languages: List of target language locales
        voting_methods: List of voting methods to compare
        num_examples: Number of test examples per language
        num_languages: Number of source models to include in ensemble
        output_dir: Base output directory

    Returns:
        Comprehensive results dictionary
    """
    print("=" * 80)
    print("COMPREHENSIVE ENSEMBLE METHODS COMPARISON")
    print("=" * 80)
    print(f"Target languages: {target_languages}")
    print(f"Voting methods: {voting_methods}")
    print(f"Examples per language: {num_examples}")
    print(f"Source models per ensemble: {num_languages}")
    print("=" * 80)

    all_results = {}
    comparison_data = []

    for target_lang in target_languages:
        print(f"\n🎯 Processing target language: {target_lang}")
        print("-" * 50)

        all_results[target_lang] = {}

        for voting_method in voting_methods:
            print(f"\n📊 Running {voting_method} ensemble...")

            try:
                # Run single experiment
                result = run_single_ensemble_experiment(
                    target_lang=target_lang,
                    voting_method=voting_method,
                    num_examples=num_examples,
                    num_languages=num_languages,
                    output_dir=output_dir
                )

                # Store results
                all_results[target_lang][voting_method] = result

                # Extract key metrics for comparison
                comparison_entry = {
                    'target_language': target_lang,
                    'voting_method': voting_method,
                    'accuracy': result['performance']['accuracy'],
                    'correct_predictions': result['performance']['correct_predictions'],
                    'total_predictions': result['performance']['total_predictions'],
                    'error_rate': result['performance']['error_rate'],
                    'num_models': result['experiment_info']['num_models'],
                    'top_languages': list(result['models'].keys())[:3],  # Top 3 languages
                    'top_weights': [info['weight'] for info in list(result['models'].values())[:3]]
                }
                comparison_data.append(comparison_entry)

                print(f"✅ {voting_method}: Accuracy = {result['performance']['accuracy']:.4f}")

            except Exception as e:
                print(f"❌ {voting_method}: Failed with error: {e}")
                # Add failed entry
                comparison_entry = {
                    'target_language': target_lang,
                    'voting_method': voting_method,
                    'accuracy': 0.0,
                    'correct_predictions': 0,
                    'total_predictions': 0,
                    'error_rate': 1.0,
                    'num_models': 0,
                    'top_languages': [],
                    'top_weights': [],
                    'error': str(e)
                }
                comparison_data.append(comparison_entry)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Generate summary statistics
    summary_stats = generate_summary_statistics(comparison_df, voting_methods)

    # Prepare final results
    final_results = {
        'experiment_info': {
            'timestamp': datetime.utcnow().isoformat(),
            'target_languages': target_languages,
            'voting_methods': voting_methods,
            'num_examples': num_examples,
            'num_languages': num_languages,
            'total_experiments': len(target_languages) * len(voting_methods),
            'successful_experiments': len([r for r in comparison_data if 'error' not in r])
        },
        'detailed_results': all_results,
        'comparison_table': comparison_df.to_dict('records'),
        'summary_statistics': summary_stats
    }

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(project_root, output_dir, f"comprehensive_comparison_{timestamp}.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save comparison table as CSV
    csv_file = os.path.join(project_root, output_dir, f"comparison_table_{timestamp}.csv")
    comparison_df.to_csv(csv_file, index=False)

    print(f"\n" + "=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {results_file}")
    print(f"Comparison table saved to: {csv_file}")

    # Print summary
    print_summary_statistics(summary_stats)

    return final_results


def generate_summary_statistics(comparison_df: pd.DataFrame, voting_methods: List[str]) -> Dict[str, Any]:
    """Generate summary statistics for the comparison."""
    stats = {}

    # Overall statistics
    successful_experiments = comparison_df[comparison_df['accuracy'] > 0]
    stats['overall'] = {
        'total_experiments': len(comparison_df),
        'successful_experiments': len(successful_experiments),
        'average_accuracy': successful_experiments['accuracy'].mean(),
        'best_accuracy': successful_experiments['accuracy'].max(),
        'worst_accuracy': successful_experiments['accuracy'].min(),
        'accuracy_std': successful_experiments['accuracy'].std()
    }

    # Per-method statistics
    stats['by_method'] = {}
    for method in voting_methods:
        method_data = successful_experiments[successful_experiments['voting_method'] == method]
        if len(method_data) > 0:
            stats['by_method'][method] = {
                'count': len(method_data),
                'average_accuracy': method_data['accuracy'].mean(),
                'best_accuracy': method_data['accuracy'].max(),
                'worst_accuracy': method_data['accuracy'].min(),
                'accuracy_std': method_data['accuracy'].std()
            }
        else:
            stats['by_method'][method] = {
                'count': 0,
                'average_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 0.0,
                'accuracy_std': 0.0
            }

    # Per-language statistics
    stats['by_language'] = {}
    for lang in comparison_df['target_language'].unique():
        lang_data = successful_experiments[successful_experiments['target_language'] == lang]
        if len(lang_data) > 0:
            best_method = lang_data.loc[lang_data['accuracy'].idxmax(), 'voting_method']
            best_accuracy = lang_data['accuracy'].max()

            stats['by_language'][lang] = {
                'count': len(lang_data),
                'average_accuracy': lang_data['accuracy'].mean(),
                'best_method': best_method,
                'best_accuracy': best_accuracy
            }
        else:
            stats['by_language'][lang] = {
                'count': 0,
                'average_accuracy': 0.0,
                'best_method': 'None',
                'best_accuracy': 0.0
            }

    # Method ranking
    method_performance = [(method, stats['by_method'][method]['average_accuracy'])
                         for method in voting_methods]
    method_performance.sort(key=lambda x: x[1], reverse=True)
    stats['method_ranking'] = method_performance

    return stats


def print_summary_statistics(stats: Dict[str, Any]):
    """Print a formatted summary of the comparison results."""
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)

    # Overall performance
    overall = stats['overall']
    print(f"Total experiments: {overall['total_experiments']}")
    print(f"Successful experiments: {overall['successful_experiments']}")
    print(f"Average accuracy: {overall['average_accuracy']:.4f}")
    print(f"Best accuracy: {overall['best_accuracy']:.4f}")
    print(f"Worst accuracy: {overall['worst_accuracy']:.4f}")

    print(f"\n🏆 METHOD RANKING (by average accuracy):")
    print("-" * 40)
    for i, (method, accuracy) in enumerate(stats['method_ranking'], 1):
        print(f"{i}. {method}: {accuracy:.4f}")

    print(f"\n🌍 BEST METHOD PER LANGUAGE:")
    print("-" * 40)
    for lang, lang_stats in stats['by_language'].items():
        if lang_stats['count'] > 0:
            print(f"{lang}: {lang_stats['best_method']} ({lang_stats['best_accuracy']:.4f})")

    print(f"\n📊 DETAILED METHOD PERFORMANCE:")
    print("-" * 40)
    for method, method_stats in stats['by_method'].items():
        if method_stats['count'] > 0:
            print(f"{method}:")
            print(f"  Average: {method_stats['average_accuracy']:.4f}")
            print(f"  Range: {method_stats['worst_accuracy']:.4f} - {method_stats['best_accuracy']:.4f}")
            print(f"  Std: {method_stats['accuracy_std']:.4f}")


def main():
    """Main function to run the comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Ensemble Methods Comparison")
    parser.add_argument(
        "--target-languages",
        type=str,
        nargs="+",
        default=["en-US", "sq-AL", "sw-KE", "th-TH"],
        help="Target languages to test"
    )
    parser.add_argument(
        "--voting-methods",
        type=str,
        nargs="+",
        default=["majority", "weighted_majority", "soft", "uriel_logits"],
        choices=["majority", "weighted_majority", "soft", "uriel_logits"],
        help="Voting methods to compare"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of test examples per language"
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=5,
        help="Number of source models in ensemble"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Run comprehensive comparison
    results = run_comprehensive_comparison(
        target_languages=args.target_languages,
        voting_methods=args.voting_methods,
        num_examples=args.num_examples,
        num_languages=args.num_languages,
        output_dir=args.output_dir
    )

    return results


if __name__ == "__main__":
    main()