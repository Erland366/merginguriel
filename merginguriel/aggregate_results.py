#!/usr/bin/env python3
"""
Script to aggregate and compare results from large-scale experiments.
Enhanced version with comprehensive automated evaluation reports including N-vs-N baselines.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from merginguriel import logger
from merginguriel.aggregation import (
    aggregate_results,
    create_comparison_table,
    find_evaluation_matrices,
    generate_summary_stats,
    generate_win_rate_analysis,
    load_evaluation_matrix,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare experiment results with comprehensive baseline analysis"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Custom prefix for output files",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "markdown", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Show missing/failed experiments",
    )
    parser.add_argument(
        "--nxn-results-dir",
        type=str,
        default="nxn_results",
        help="Directory containing N-x-N evaluation results (default: nxn_results)",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline integration (for faster processing)",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=None,
        help="Filter results to specific locales only",
    )
    parser.add_argument(
        "--experiment-types",
        nargs="+",
        default=None,
        help="Filter results to specific experiment types only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results (default: results)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting comprehensive results aggregation...")

    evaluation_matrix = _load_evaluation_matrices(args)
    df = _aggregate_and_filter(args, evaluation_matrix)

    if len(df) == 0:
        logger.error("No results found in the results directory.")
        return

    logger.info(f"Processing {len(df)} experiment results")

    logger.info("Creating comparison table...")
    comparison_df = create_comparison_table(df)

    logger.info("Generating summary statistics...")
    summary = generate_summary_stats(df)

    logger.info("Analyzing win rates...")
    win_analysis = generate_win_rate_analysis(df, comparison_df)

    if args.show_missing:
        _show_missing_experiments(df, comparison_df)

    _print_summary_info(summary, win_analysis, comparison_df)

    logger.info("Saving results...")
    saved_files = save_results(
        df, comparison_df, summary, win_analysis, output_dir=Path(args.results_dir)
    )

    logger.info("Files generated:")
    for file_type, filename in saved_files.items():
        if filename:
            logger.info(f"  {file_type}: {filename}")

    logger.info("Comprehensive aggregation completed successfully!")


def _load_evaluation_matrices(args: argparse.Namespace) -> Dict[str, pd.DataFrame] | None:
    """Load evaluation matrices for baseline comparison."""
    if args.no_baselines:
        return None

    matrix_map: Dict[str, pd.DataFrame] = {}
    matrix_paths = find_evaluation_matrices(args.nxn_results_dir)

    for fam, path in matrix_paths.items():
        mat = load_evaluation_matrix(path)
        if mat is not None:
            matrix_map[fam] = mat

    if matrix_map:
        logger.info(f"Loaded evaluation matrices for families: {list(matrix_map.keys())}")
        return matrix_map

    logger.info("No evaluation matrix found, proceeding without baseline integration")
    return None


def _aggregate_and_filter(
    args: argparse.Namespace,
    evaluation_matrix: Dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Aggregate results and apply filters."""
    logger.info("Aggregating experiment results...")
    df = aggregate_results(evaluation_matrix, args.results_dir)

    if args.locales:
        df = df[df["locale"].isin(args.locales)]
        logger.info(f"Filtered to {len(df)} results for locales: {args.locales}")

    if args.experiment_types:
        mask = df["experiment_type"].isin(args.experiment_types)
        if "experiment_variant" in df.columns:
            mask = mask | df["experiment_variant"].isin(args.experiment_types)
        df = df[mask]
        logger.info(f"Filtered to {len(df)} results for experiment types: {args.experiment_types}")

    return df


def _show_missing_experiments(df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    """Log missing/failed experiments."""
    logger.info("Missing/failed experiments analysis:")

    if "experiment_variant" in df.columns:
        experiment_types = sorted(df["experiment_variant"].dropna().unique())
    else:
        experiment_types = sorted(df["experiment_type"].unique())

    for exp_type in experiment_types:
        if exp_type in comparison_df.columns:
            missing = comparison_df[comparison_df[exp_type].isna()]
            if len(missing) > 0:
                logger.info(f"  {exp_type}: {len(missing)} locales missing")
                for locale in missing["locale"]:
                    logger.info(f"    - {locale}")


def _print_summary_info(
    summary: dict, win_analysis: dict, comparison_df: pd.DataFrame
) -> None:
    """Print summary statistics and win rate analysis."""
    logger.info("Summary Statistics:")
    for exp_type, stats in summary.items():
        logger.info(
            f"  {exp_type}: {stats['count']} experiments, "
            f"mean accuracy = {stats['mean_accuracy']:.4f}"
        )

    if win_analysis:
        logger.info("Win Rate Analysis Summary:")
        for exp_type, stats in win_analysis.items():
            if stats["vs_baseline_total"] > 0:
                win_rate = (stats["vs_baseline_wins"] / stats["vs_baseline_total"]) * 100
                logger.info(
                    f"  {exp_type}: {win_rate:.1f}% win rate vs baseline "
                    f"({stats['vs_baseline_wins']}/{stats['vs_baseline_total']})"
                )

    experiment_columns = [
        col
        for col in comparison_df.columns
        if col not in ["locale", "baseline", "best_source_accuracy", "best_overall_accuracy"]
        and not col.endswith("_improvement")
        and not col.endswith("_vs_")
    ]

    for exp_type in experiment_columns:
        if exp_type in comparison_df.columns and not comparison_df[exp_type].isna().all():
            best_idx = comparison_df[exp_type].idxmax()
            best_result = comparison_df.loc[best_idx]
            logger.info(
                f"Best {exp_type} result: {best_result['locale']} "
                f"({best_result[exp_type]:.4f})"
            )


if __name__ == "__main__":
    main()
