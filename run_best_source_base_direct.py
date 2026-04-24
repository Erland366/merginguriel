#!/usr/bin/env python3
"""
Run direct best-source-base layer ablation (no cross-validation).

Finds best source for the target locale, merges layer subsets from
remaining sources onto it, evaluates directly on the target.

Usage:
    # Run full ablation
    python run_best_source_base_direct.py --config configs/ablations/best_source_base_direct_swke.yaml

    # Plan only
    python run_best_source_base_direct.py --config configs/ablations/best_source_base_direct_swke.yaml --plan

    # Run specific ablation points
    python run_best_source_base_direct.py --config configs/ablations/best_source_base_direct_swke.yaml \
        --ablation-points include_only_group_middle include_only_bottom_middle

    # Analyze existing results
    python run_best_source_base_direct.py --config configs/ablations/best_source_base_direct_swke.yaml --analyze
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from merginguriel.selective_layer import (
    get_include_only_ablation_points,
    LayerAblationDB,
)
from merginguriel.selective_layer.best_source_base_direct import (
    DirectBestSourceBaseConfig,
    DirectBestSourceBaseAblation,
    analyze_direct_results,
    print_direct_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run direct best-source-base layer ablation",
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
        help="Analyze existing results",
    )

    parser.add_argument(
        "--ablation-points",
        nargs="+",
        help="Run only specific ablation points",
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
    points = get_include_only_ablation_points()
    print("\nAvailable ablation points:")
    print("-" * 50)
    for name, include_layers in sorted(points.items()):
        layers_str = str(include_layers) if include_layers else "[]"
        print(f"  {name:30s} -> include {layers_str}")
    print()


def run_plan(config: DirectBestSourceBaseConfig):
    ablation = DirectBestSourceBaseAblation(config)
    records = ablation.plan(str(project_root))

    best_source = records[0].best_source_locale if records else "?"
    best_acc = records[0].best_source_accuracy if records else 0.0

    print(f"\n{'='*60}")
    print(f"DIRECT BEST-SOURCE-BASE PLAN: {config.name}")
    print(f"{'='*60}")
    print(f"Target locale: {config.target_locale}")
    print(f"Best source: {best_source} (acc: {best_acc:.4f})")
    print(f"Source locales: {config.source_locales}")
    print(f"Remaining sources (for merge): {json.loads(records[0].remaining_sources) if records else []}")
    print(f"Total experiments: {len(records)}")
    print()

    points = ablation.ablation_points
    print("Ablation points:")
    for name, layers in sorted(points.items()):
        print(f"  {name:30s} -> include layers {layers}")

    print(f"\nDatabase: {config.db_path}")
    print(f"Results dir: {config.results_dir}")


def run_analysis(config: DirectBestSourceBaseConfig):
    db = LayerAblationDB(config.db_path)
    results_df = db.get_results_df(config.name)

    if results_df.empty:
        print(f"No completed results found for '{config.name}'")
        print(f"Database: {config.db_path}")
        return

    print(f"\nLoaded {len(results_df)} completed experiments from {config.db_path}")

    summary = analyze_direct_results(results_df)
    print_direct_summary(summary, config.target_locale)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(results_dir / "transfer_summary.csv", index=False)
    results_df.to_csv(results_dir / "full_results.csv", index=False)
    print(f"\nResults saved to: {results_dir}")


def run_experiments(
    config: DirectBestSourceBaseConfig,
    ablation_points: list = None,
):
    ablation = DirectBestSourceBaseAblation(config)

    if ablation_points:
        original_points = ablation.ablation_points
        ablation.ablation_points = {
            k: v for k, v in original_points.items() if k in ablation_points
        }
        print(f"Filtered to ablation points: {list(ablation.ablation_points.keys())}")

    print(f"\nRunning direct best-source-base ablation: {config.name}")
    print(f"Target: {config.target_locale}")
    print(f"Sources: {config.source_locales}")
    print(f"Ablation points: {len(ablation.ablation_points)}")
    print()

    results_df = ablation.run_all(str(project_root))

    if not results_df.empty:
        summary = analyze_direct_results(results_df)
        print_direct_summary(summary, config.target_locale)

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

    config = DirectBestSourceBaseConfig.from_yaml(args.config)

    if args.no_resume:
        config.resume = False

    if args.plan:
        run_plan(config)
    elif args.analyze:
        run_analysis(config)
    else:
        run_experiments(config, ablation_points=args.ablation_points)


if __name__ == "__main__":
    import json
    main()
