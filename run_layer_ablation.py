#!/usr/bin/env python3
"""
Run layer ablation experiments for selective layer merging.

This script runs leave-one-layer-out ablation with leave-one-source-out
cross-validation to identify which transformer layers contribute positively
to cross-lingual transfer vs which cause interference.

Usage:
    # Run full ablation from config
    python run_layer_ablation.py --config configs/ablations/layer_ablation_swke.yaml

    # Plan only (dry run)
    python run_layer_ablation.py --config configs/ablations/layer_ablation_swke.yaml --plan

    # Run specific ablation points
    python run_layer_ablation.py --config configs/ablations/layer_ablation_swke.yaml \
        --ablation-points baseline_all_layers exclude_layer_5

    # Run specific holdout locales
    python run_layer_ablation.py --config configs/ablations/layer_ablation_swke.yaml \
        --holdouts en-US de-DE

    # Analyze existing results
    python run_layer_ablation.py --config configs/ablations/layer_ablation_swke.yaml --analyze
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from merginguriel.selective_layer import (
    LayerAblationConfig,
    LeaveOneSourceOutCV,
    LayerAblationDB,
    analyze_ablation_results,
    print_transfer_summary,
    get_ablation_points,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run layer ablation experiments for selective layer merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to ablation config YAML file",
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Plan experiments only, don't run them",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results instead of running experiments",
    )

    parser.add_argument(
        "--ablation-points",
        nargs="+",
        help="Run only specific ablation points (e.g., exclude_layer_5 exclude_group_top)",
    )

    parser.add_argument(
        "--holdouts",
        nargs="+",
        help="Run only specific holdout locales (e.g., en-US de-DE)",
    )

    parser.add_argument(
        "--list-points",
        action="store_true",
        help="List all available ablation points and exit",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous run (skip completed experiments)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume, re-run all experiments",
    )

    return parser.parse_args()


def list_ablation_points():
    """Print all available ablation points."""
    points = get_ablation_points()
    print("\nAvailable ablation points:")
    print("-" * 40)
    for name, layers in sorted(points.items()):
        layers_str = str(layers) if layers else "[]"
        print(f"  {name:25s} -> exclude {layers_str}")
    print()


def run_plan(config: LayerAblationConfig):
    """Plan experiments without running them."""
    cv = LeaveOneSourceOutCV(config)
    records = cv.plan()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT PLAN: {config.name}")
    print(f"{'='*60}")
    print(f"Target locale: {config.target_locale}")
    print(f"CV source locales: {config.cv_source_locales}")
    print(f"Total experiments: {len(records)}")
    print(f"  - Holdout locales: {len(config.cv_source_locales)}")
    print(f"  - Ablation points: {len(get_ablation_points())}")
    print()

    # Group by holdout
    by_holdout = {}
    for r in records:
        if r.holdout_locale not in by_holdout:
            by_holdout[r.holdout_locale] = []
        by_holdout[r.holdout_locale].append(r.ablation_point)

    print("Experiments per holdout:")
    for holdout, points in sorted(by_holdout.items()):
        print(f"  {holdout}: {len(points)} ablation points")

    print(f"\nDatabase: {config.db_path}")
    print(f"Results dir: {config.results_dir}")


def run_analysis(config: LayerAblationConfig):
    """Analyze existing results."""
    db = LayerAblationDB(config.db_path)
    results_df = db.get_results_df(config.name)

    if results_df.empty:
        print(f"No completed results found for ablation '{config.name}'")
        print(f"Database: {config.db_path}")
        return

    print(f"\nLoaded {len(results_df)} completed experiments from {config.db_path}")

    # Analyze
    summary = analyze_ablation_results(results_df)
    print_transfer_summary(summary)

    # Save summary
    summary_path = Path(config.results_dir) / "transfer_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Also export full results
    full_results_path = Path(config.results_dir) / "full_results.csv"
    results_df.to_csv(full_results_path, index=False)
    print(f"Full results saved to: {full_results_path}")


def run_experiments(
    config: LayerAblationConfig,
    ablation_points: list = None,
    holdouts: list = None,
):
    """Run layer ablation experiments."""
    # Filter config if specific points/holdouts requested
    if holdouts:
        config.cv_source_locales = [
            loc for loc in config.cv_source_locales if loc in holdouts
        ]
        print(f"Filtered to holdouts: {config.cv_source_locales}")

    cv = LeaveOneSourceOutCV(config)

    # Filter ablation points if requested
    if ablation_points:
        original_points = cv.ablation_points
        cv.ablation_points = {
            k: v for k, v in original_points.items() if k in ablation_points
        }
        print(f"Filtered to ablation points: {list(cv.ablation_points.keys())}")

    # Run
    print(f"\nRunning layer ablation: {config.name}")
    print(f"Target: {config.target_locale}")
    print(f"CV sources: {config.cv_source_locales}")
    print(f"Ablation points: {len(cv.ablation_points)}")
    print(f"Total experiments: {len(config.cv_source_locales) * len(cv.ablation_points)}")
    print()

    results_df = cv.run_all(str(project_root))

    # Analyze results
    if not results_df.empty:
        summary = analyze_ablation_results(results_df)
        print_transfer_summary(summary)

        # Save results
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        summary.to_csv(results_dir / "transfer_summary.csv", index=False)
        results_df.to_csv(results_dir / "full_results.csv", index=False)
        print(f"\nResults saved to: {results_dir}")


def main():
    args = parse_args()

    if args.list_points:
        list_ablation_points()
        return

    if not args.config:
        print("Error: --config is required (unless using --list-points)")
        sys.exit(1)

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    config = LayerAblationConfig.from_yaml(args.config)

    # Handle resume flag
    if args.no_resume:
        config.resume = False

    if args.plan:
        run_plan(config)
    elif args.analyze:
        run_analysis(config)
    else:
        run_experiments(
            config,
            ablation_points=args.ablation_points,
            holdouts=args.holdouts,
        )


if __name__ == "__main__":
    main()
