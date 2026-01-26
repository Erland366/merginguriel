#!/usr/bin/env python3
"""
Run ablation sweep from YAML config file.

Usage:
    python run_ablation_sweep.py configs/ablations/paper_exploration_sweep.yaml
    python run_ablation_sweep.py configs/ablations/paper_exploration_sweep.yaml --dry-run
    python run_ablation_sweep.py configs/ablations/paper_exploration_sweep.yaml --plan-only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from merginguriel.experiments import AblationRunner, AblationConfig


def main():
    parser = argparse.ArgumentParser(description="Run ablation sweep from YAML config")
    parser.add_argument("config", type=Path, help="Path to ablation YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without running")
    parser.add_argument("--plan-only", action="store_true", help="Only register plans, don't run")
    parser.add_argument("--no-resume", action="store_true", help="Re-run completed experiments")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    config = AblationConfig.from_yaml(args.config)

    # Apply command-line overrides
    if args.dry_run:
        config.dry_run = True
    if args.no_resume:
        config.resume = False

    print(f"Loaded config: {config.name}")
    print(f"Description: {config.description}")
    print(f"Estimated runs: {config.estimate_runs()}")
    print()

    # Create runner
    runner = AblationRunner(config)

    if args.plan_only:
        # Just register plans and show them
        print("Registering plans...")
        exp_ids = runner.register_plans()
        print(f"Registered {len(exp_ids)} experiments")

        # Show planned experiments
        print("\nPlanned experiments:")
        for exp_id in exp_ids:
            record = runner.db.get(exp_id)
            print(f"  [{record.status}] {record.locale} / {record.method} / sim={record.similarity_type} / include_target={record.include_target}")
    else:
        # Run all experiments
        stats = runner.run_all()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Completed: {stats['completed']}")
        print(f"Failed:    {stats['failed']}")
        print(f"Skipped:   {stats['skipped']}")


if __name__ == "__main__":
    main()
