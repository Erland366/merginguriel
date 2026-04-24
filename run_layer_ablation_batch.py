#!/usr/bin/env python3
"""
Batch runner for layer ablation experiments across multiple locales.

Runs layer ablation configs sequentially, one locale at a time.
Each locale runs 7 holdouts x 16 ablation points = 112 experiments.

Usage:
    # Run all layer ablation configs
    python run_layer_ablation_batch.py

    # Run specific configs
    python run_layer_ablation_batch.py --configs configs/ablations/layer_ablation_swke.yaml configs/ablations/layer_ablation_cygb.yaml

    # Plan only (dry run)
    python run_layer_ablation_batch.py --plan

    # Analyze all completed results
    python run_layer_ablation_batch.py --analyze
"""

import argparse
import glob
import sys
import time
from pathlib import Path

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


def find_all_configs() -> list[Path]:
    """Find all layer ablation config files."""
    pattern = str(project_root / "configs" / "ablations" / "layer_ablation_*.yaml")
    configs = sorted(glob.glob(pattern))
    return [Path(c) for c in configs]


def run_plan_all(configs: list[Path]):
    """Plan all experiments across all configs."""
    total = 0
    for config_path in configs:
        config = LayerAblationConfig.from_yaml(config_path)
        n_holdouts = len(config.cv_source_locales)
        n_points = len(get_ablation_points())
        n_experiments = n_holdouts * n_points
        total += n_experiments
        print(f"  {config.target_locale:8s} | {config.name:30s} | {n_holdouts} holdouts x {n_points} points = {n_experiments} experiments")

    print(f"\n  TOTAL: {total} experiments across {len(configs)} locales")


def run_all(configs: list[Path]):
    """Run all experiments sequentially."""
    results = {}

    for i, config_path in enumerate(configs):
        config = LayerAblationConfig.from_yaml(config_path)
        print(f"\n{'#'*70}")
        print(f"# LOCALE {i+1}/{len(configs)}: {config.target_locale}")
        print(f"# Config: {config_path}")
        print(f"{'#'*70}")

        start_time = time.time()
        cv = LeaveOneSourceOutCV(config)
        results_df = cv.run_all(str(project_root))
        elapsed = time.time() - start_time

        if not results_df.empty:
            summary = analyze_ablation_results(results_df)
            print_transfer_summary(summary)

            # Save per-locale results
            results_dir = Path(config.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            summary.to_csv(results_dir / "transfer_summary.csv", index=False)
            results_df.to_csv(results_dir / "full_results.csv", index=False)

            results[config.target_locale] = summary

        print(f"\n  Completed {config.target_locale} in {elapsed/60:.1f} minutes")

    return results


def run_analyze_all(configs: list[Path]):
    """Analyze all completed results."""
    for config_path in configs:
        config = LayerAblationConfig.from_yaml(config_path)
        db = LayerAblationDB(config.db_path)
        stats = db.stats()

        completed = stats["by_status"].get("completed", 0)
        total = stats["total"]

        if completed == 0:
            print(f"  {config.target_locale:8s} | No completed experiments")
            continue

        print(f"\n{'='*60}")
        print(f"  {config.target_locale} | {completed}/{total} completed")
        print(f"{'='*60}")

        results_df = db.get_results_df(config.name)
        if not results_df.empty:
            summary = analyze_ablation_results(results_df)
            print_transfer_summary(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for layer ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        help="Specific config files to run (default: all layer_ablation_*.yaml)",
    )
    parser.add_argument("--plan", action="store_true", help="Plan only")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")

    args = parser.parse_args()
    configs = args.configs if args.configs else find_all_configs()

    if not configs:
        print("No layer ablation configs found")
        sys.exit(1)

    print(f"Found {len(configs)} layer ablation configs:")
    for c in configs:
        print(f"  - {c}")

    if args.plan:
        run_plan_all(configs)
    elif args.analyze:
        run_analyze_all(configs)
    else:
        run_all(configs)


if __name__ == "__main__":
    main()
