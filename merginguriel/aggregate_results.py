#!/usr/bin/env python3
"""
Script to aggregate and compare results from large-scale experiments.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

def load_results_from_folder(folder_path):
    """Load results.json from a folder."""
    results_file = os.path.join(folder_path, "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            return None
    return None

def extract_accuracy(results):
    """Extract accuracy from results."""
    if results and 'performance' in results:
        return results['performance']['accuracy']
    return None

def get_experiment_folders():
    """Get all experiment folders from the results directory."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return []
    
    folders = []
    for item in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, item)
        if os.path.isdir(folder_path):
            folders.append(item)
    
    return sorted(folders)

def parse_folder_name(folder_name):
    """Parse folder name to extract experiment type and locale."""
    parts = folder_name.split('_')
    
    # Handle different patterns:
    # 1. "baseline_sq-AL" -> type="baseline", locale="sq-AL"
    # 2. "similarity_sq-AL" -> type="similarity", locale="sq-AL"
    # 3. "average_sq-AL" -> type="average", locale="sq-AL"
    # 4. "xlm-roberta-base_massive_k_sq-AL_alpha_0.5_sq-AL_epoch-9_sq-AL" -> type="baseline", locale="sq-AL"
    
    if parts[0] in ['baseline', 'similarity', 'average']:
        # Simple case: prefix_locale
        exp_type = parts[0]
        locale = '_'.join(parts[1:])
    elif 'massive_k_' in folder_name:
        # Complex case: baseline with full model name
        exp_type = 'baseline'
        # Find the locale part (usually the last part before the final underscore)
        if len(parts) >= 2:
            locale = parts[-1]  # Last part is usually the locale
        else:
            locale = 'unknown'
    else:
        # Fallback
        exp_type = 'unknown'
        locale = folder_name
    
    return exp_type, locale

def aggregate_results():
    """Aggregate results from all experiment folders."""
    folders = get_experiment_folders()
    
    data = []
    
    for folder in folders:
        folder_path = os.path.join("results", folder)
        results = load_results_from_folder(folder_path)
        
        if results:
            exp_type, locale = parse_folder_name(folder)
            accuracy = extract_accuracy(results)
            
            # Extract additional info
            model_info = results.get('evaluation_info', {})
            perf_info = results.get('performance', {})
            
            data.append({
                'locale': locale,
                'experiment_type': exp_type,
                'folder_name': folder,
                'accuracy': accuracy,
                'correct_predictions': perf_info.get('correct_predictions'),
                'total_predictions': perf_info.get('total_predictions'),
                'error_rate': perf_info.get('error_rate'),
                'model_name': model_info.get('model_name'),
                'subfolder': model_info.get('subfolder'),
                'timestamp': model_info.get('timestamp')
            })
    
    return pd.DataFrame(data)

def create_comparison_table(df):
    """Create a comparison table with baseline, similarity, and average results."""
    # Pivot the data to have one row per locale
    pivot_df = df.pivot_table(
        index='locale', 
        columns='experiment_type', 
        values='accuracy', 
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename_axis(None, axis=1)
    
    # Ensure we have the expected columns
    expected_columns = ['locale', 'baseline', 'similarity', 'average']
    for col in expected_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = None
    
    # Reorder columns
    pivot_df = pivot_df[expected_columns]
    
    # Calculate improvements
    if 'baseline' in pivot_df.columns and 'similarity' in pivot_df.columns:
        pivot_df['similarity_improvement'] = pivot_df.apply(
            lambda row: row['similarity'] - row['baseline'] 
            if row['baseline'] is not None and row['similarity'] is not None 
            else None, axis=1
        )
    
    if 'baseline' in pivot_df.columns and 'average' in pivot_df.columns:
        pivot_df['average_improvement'] = pivot_df.apply(
            lambda row: row['average'] - row['baseline'] 
            if row['baseline'] is not None and row['average'] is not None 
            else None, axis=1
        )
    
    return pivot_df

def generate_summary_stats(df):
    """Generate summary statistics."""
    summary = {}
    
    # Overall statistics by experiment type
    for exp_type in ['baseline', 'similarity', 'average']:
        if exp_type in df['experiment_type'].values:
            type_df = df[df['experiment_type'] == exp_type]
            valid_acc = type_df[type_df['accuracy'].notna()]
            
            if len(valid_acc) > 0:
                summary[exp_type] = {
                    'count': len(valid_acc),
                    'mean_accuracy': valid_acc['accuracy'].mean(),
                    'std_accuracy': valid_acc['accuracy'].std(),
                    'min_accuracy': valid_acc['accuracy'].min(),
                    'max_accuracy': valid_acc['accuracy'].max()
                }
    
    return summary

def save_results(df, comparison_df, summary):
    """Save all results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    df.to_csv(f'results_aggregated_{timestamp}.csv', index=False)
    print(f"Raw results saved to results_aggregated_{timestamp}.csv")
    
    # Save comparison table
    comparison_df.to_csv(f'results_comparison_{timestamp}.csv', index=False)
    print(f"Comparison table saved to results_comparison_{timestamp}.csv")
    
    # Save summary
    with open(f'results_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved to results_summary_{timestamp}.json")
    
    # Save comparison table as markdown for easy reading
    markdown_content = create_markdown_report(comparison_df, summary)
    with open(f'results_report_{timestamp}.md', 'w') as f:
        f.write(markdown_content)
    print(f"Report saved to results_report_{timestamp}.md")

def create_markdown_report(comparison_df, summary):
    """Create a markdown report."""
    report = "# Large-Scale Experiment Results\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Summary statistics
    report += "## Summary Statistics\n\n"
    for exp_type, stats in summary.items():
        report += f"### {exp_type.upper()}\n"
        report += f"- Number of experiments: {stats['count']}\n"
        report += f"- Mean accuracy: {stats['mean_accuracy']:.4f}\n"
        report += f"- Standard deviation: {stats['std_accuracy']:.4f}\n"
        report += f"- Min accuracy: {stats['min_accuracy']:.4f}\n"
        report += f"- Max accuracy: {stats['max_accuracy']:.4f}\n\n"
    
    # Comparison table
    report += "## Detailed Comparison\n\n"
    report += "| Locale | Baseline | Similarity | Average | Sim. Improvement | Avg. Improvement |\n"
    report += "|--------|----------|------------|---------|------------------|-----------------|\n"
    
    for _, row in comparison_df.iterrows():
        baseline = f"{row['baseline']:.4f}" if row['baseline'] is not None else "N/A"
        similarity = f"{row['similarity']:.4f}" if row['similarity'] is not None else "N/A"
        average = f"{row['average']:.4f}" if row['average'] is not None else "N/A"
        sim_imp = f"{row['similarity_improvement']:+.4f}" if row['similarity_improvement'] is not None else "N/A"
        avg_imp = f"{row['average_improvement']:+.4f}" if row['average_improvement'] is not None else "N/A"
        
        report += f"| {row['locale']} | {baseline} | {similarity} | {average} | {sim_imp} | {avg_imp} |\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Aggregate and compare experiment results")
    parser.add_argument("--output-prefix", type=str, default=None,
                       help="Custom prefix for output files")
    parser.add_argument("--format", choices=['csv', 'json', 'markdown', 'all'], default='all',
                       help="Output format (default: all)")
    parser.add_argument("--show-missing", action="store_true",
                       help="Show missing/failed experiments")
    
    args = parser.parse_args()
    
    print("Aggregating results...")
    
    # Load and aggregate results
    df = aggregate_results()
    
    if len(df) == 0:
        print("No results found in the results directory.")
        return
    
    print(f"Found {len(df)} experiment results")
    
    # Create comparison table
    comparison_df = create_comparison_table(df)
    
    # Generate summary statistics
    summary = generate_summary_stats(df)
    
    # Show missing experiments
    if args.show_missing:
        print("\nMissing/Failed experiments:")
        for exp_type in ['baseline', 'similarity', 'average']:
            missing = comparison_df[comparison_df[exp_type].isna()]
            if len(missing) > 0:
                print(f"  {exp_type}: {len(missing)} locales")
                for locale in missing['locale']:
                    print(f"    - {locale}")
    
    # Print summary
    print("\nSummary Statistics:")
    for exp_type, stats in summary.items():
        print(f"  {exp_type}: {stats['count']} experiments, "
              f"mean accuracy = {stats['mean_accuracy']:.4f}")
    
    # Best performing experiments
    if 'similarity' in comparison_df.columns and not comparison_df['similarity'].isna().all():
        best_sim_idx = comparison_df['similarity'].idxmax()
        best_sim = comparison_df.loc[best_sim_idx]
        print(f"\nBest similarity result: {best_sim['locale']} "
              f"({best_sim['similarity']:.4f})")
    
    if 'average' in comparison_df.columns and not comparison_df['average'].isna().all():
        best_avg_idx = comparison_df['average'].idxmax()
        best_avg = comparison_df.loc[best_avg_idx]
        print(f"Best average result: {best_avg['locale']} "
              f"({best_avg['average']:.4f})")
    
    # Save results
    save_results(df, comparison_df, summary)
    
    print("\nAggregation completed!")

if __name__ == "__main__":
    main()