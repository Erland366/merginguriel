#!/usr/bin/env python3
"""
Run best-source-base layer ablation experiments.

Use the best single source model (from NxN matrix) as base, merge ONLY
specified layer subsets from remaining sources. Tests whether merging
specific layers onto a strong finetuned base improves or hurts accuracy.

Usage:
    # Run full ablation from config
    python run_best_source_base_ablation.py --config configs/ablations/best_source_base_swke.yaml

    # Plan only (dry run)
    python run_best_source_base_ablation.py --config configs/ablations/best_source_base_swke.yaml --plan

    # Run specific ablation points
    python run_best_source_base_ablation.py --config configs/ablations/best_source_base_swke.yaml \
        --ablation-points include_only_group_middle include_only_bottom_middle

    # Analyze existing results
    python run_best_source_base_ablation.py --config configs/ablations/best_source_base_swke.yaml --analyze
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from merginguriel.selective_layer import (
    get_include_only_ablation_points,
    analyze_ablation_results,
    LayerAblationDB,
)
from merginguriel.selective_layer.best_source_base_cv import (
    BestSourceBaseCVConfig,
    LeaveOneSourceOutBestSourceBaseCV,
    print_best_source_base_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run best-source-base layer ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to best-source-base ablation config YAML file",
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
        help="Run only specific ablation points (e.g., include_only_group_middle)",
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
        "--no-resume",
        action="store_true",
        help="Don't resume, re-run all experiments",
    )

    return parser.parse_args()


def list_ablation_points():
    """Print all available ablation points."""
    points = get_include_only_ablation_points()
    print("\nAvailable ablation points (same as include-only):")
    print("-" * 50)
    for name, include_layers in sorted(points.items()):
        layers_str = str(include_layers) if include_layers else "[]"
        print(f"  {name:30s} -> include {layers_str}")
    print()


def run_plan(config: BestSourceBaseCVConfig):
    """Plan experiments without running them."""
    cv = LeaveOneSourceOutBestSourceBaseCV(config)
    records = cv.plan(str(project_root))
    points = get_include_only_ablation_points()

    print(f"\n{'='*60}")
    print(f"BEST-SOURCE-BASE EXPERIMENT PLAN: {config.name}")
    print(f"{'='*60}")
    print(f"Target locale: {config.target_locale}")
    print(f"Base model: best single source (from NxN matrix)")
    print(f"NxN matrix: {config.nxn_matrix_path}")
    print(f"CV source locales: {config.cv_source_locales}")
    print(f"Total experiments: {len(records)}")
    print(f"  - Holdout locales: {len(config.cv_source_locales)}")
    print(f"  - Ablation points: {len(points)}")
    print()

    # Show best source per holdout
    print("Best source per holdout:")
    seen_holdouts = set()
    for record in records:
        if record.holdout_locale not in seen_holdouts:
            seen_holdouts.add(record.holdout_locale)
            config_data = __import__("json").loads(record.config_json)
            print(
                f"  {record.holdout_locale:10s} -> "
                f"best source: {config_data['best_source_locale']} "
                f"(acc: {config_data['best_source_accuracy']:.4f})"
            )

    print(f"\nAblation points:")
    for name, layers in sorted(points.items()):
        print(f"  {name:30s} -> include layers {layers}")

    print(f"\nDatabase: {config.db_path}")
    print(f"Results dir: {config.results_dir}")


def run_analysis(config: BestSourceBaseCVConfig):
    """Analyze existing results."""
    db = LayerAblationDB(config.db_path)
    results_df = db.get_results_df(config.name)

    if results_df.empty:
        print(f"No completed results found for ablation '{config.name}'")
        print(f"Database: {config.db_path}")
        return

    print(f"\nLoaded {len(results_df)} completed experiments from {config.db_path}")

    summary = analyze_ablation_results(results_df)
    print_best_source_base_summary(summary)

    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(results_dir / "transfer_summary.csv", index=False)
    results_df.to_csv(results_dir / "full_results.csv", index=False)
    print(f"\nResults saved to: {results_dir}")


def run_experiments(
    config: BestSourceBaseCVConfig,
    ablation_points: list = None,
    holdouts: list = None,
):
    """Run best-source-base layer ablation experiments."""
    # Don't filter cv_source_locales — full list needed to compute
    # remaining sources per holdout. Filter at execution level instead.
    if holdouts:
        print(f"Filtered to holdouts: {holdouts}")

    cv = LeaveOneSourceOutBestSourceBaseCV(config)

    # Filter ablation points if requested
    if ablation_points:
        original_points = cv.ablation_points
        cv.ablation_points = {
            k: v for k, v in original_points.items() if k in ablation_points
        }
        print(f"Filtered to ablation points: {list(cv.ablation_points.keys())}")

    effective_holdouts = holdouts if holdouts else config.cv_source_locales
    print(f"\nRunning best-source-base layer ablation: {config.name}")
    print(f"Target: {config.target_locale}")
    print(f"Base model: best single source (from NxN matrix)")
    print(f"CV sources: {config.cv_source_locales}")
    print(f"Holdouts to run: {effective_holdouts}")
    print(f"Ablation points: {len(cv.ablation_points)}")
    print(f"Total experiments: {len(effective_holdouts) * len(cv.ablation_points)}")
    print()

    results_df = cv.run_all(str(project_root), holdout_filter=holdouts)

    if not results_df.empty:
        summary = analyze_ablation_results(results_df)
        print_best_source_base_summary(summary)

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

    config = BestSourceBaseCVConfig.from_yaml(args.config)

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
