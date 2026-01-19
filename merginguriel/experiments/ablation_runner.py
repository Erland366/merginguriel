"""
Ablation runner for systematic experiment sweeps.

Reads ablation configuration files and runs experiments while tracking
results in the SQLite database.
"""

import itertools
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from merginguriel.experiments.db import ExperimentDB, ExperimentRecord


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""

    name: str
    description: str = ""

    # Fixed parameters (same for all runs)
    fixed: Dict[str, Any] = field(default_factory=dict)

    # Parameters to sweep (cartesian product)
    sweep: Dict[str, List[Any]] = field(default_factory=dict)

    # Output configuration
    results_base_dir: str = "results"
    db_path: str = "experiments.db"

    # Execution options
    dry_run: bool = False
    resume: bool = True  # Skip already-completed experiments

    @classmethod
    def from_yaml(cls, path: Path) -> "AblationConfig":
        """Load ablation config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        ablation_data = data.get("ablation", data)
        return cls(
            name=ablation_data.get("name", path.stem),
            description=ablation_data.get("description", ""),
            fixed=ablation_data.get("fixed", {}),
            sweep=ablation_data.get("sweep", {}),
            results_base_dir=ablation_data.get("results_base_dir", "results"),
            db_path=ablation_data.get("db_path", "experiments.db"),
            dry_run=ablation_data.get("dry_run", False),
            resume=ablation_data.get("resume", True),
        )

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from the sweep."""
        if not self.sweep:
            # No sweep, just return fixed config
            return [self.fixed.copy()]

        # Get all sweep parameter names and their values
        sweep_keys = list(self.sweep.keys())
        sweep_values = [self.sweep[k] for k in sweep_keys]

        # Generate cartesian product
        configs = []
        for combo in itertools.product(*sweep_values):
            config = self.fixed.copy()
            for key, value in zip(sweep_keys, combo):
                config[key] = value
            configs.append(config)

        return configs

    def estimate_runs(self) -> int:
        """Estimate total number of runs."""
        if not self.sweep:
            return 1

        total = 1
        for values in self.sweep.values():
            total *= len(values)

        # If locales is in fixed, multiply by number of locales
        locales = self.fixed.get("locales", [None])
        if isinstance(locales, list) and len(locales) > 0:
            total *= len(locales)

        return total


class AblationRunner:
    """Runs ablation experiments and tracks results."""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.db = ExperimentDB(config.db_path)

    def plan(self) -> List[ExperimentRecord]:
        """
        Plan all experiments without running them.
        Returns list of planned experiment records.
        """
        records = []
        configs = self.config.generate_configs()

        # Get locales from fixed config (if not in sweep)
        fixed_locales = self.config.fixed.get("locales", None)

        for exp_config in configs:
            # Locales can come from sweep (in exp_config) or from fixed
            if "locales" in exp_config:
                # Single locale from sweep
                locales = [exp_config["locales"]]
            elif fixed_locales:
                # Multiple locales from fixed
                locales = fixed_locales if isinstance(fixed_locales, list) else [fixed_locales]
            else:
                # Fallback: try locale (singular) from exp_config
                locales = [exp_config.get("locale", "")]

            for locale in locales:
                record = ExperimentRecord(
                    ablation_name=self.config.name,
                    locale=locale,
                    method=exp_config.get("method", "similarity"),
                    similarity_type=exp_config.get("similarity_type", "URIEL"),
                    num_languages=exp_config.get("num_languages", 3),
                    include_target=exp_config.get("include_target", False),
                    model_family=exp_config.get("model_family", "xlm-roberta-base"),
                    status="planned",
                    config_json=json.dumps(exp_config),
                )
                records.append(record)

        return records

    def register_plans(self) -> List[int]:
        """Register all planned experiments in the database."""
        records = self.plan()
        ids = []
        for record in records:
            # Check if already exists
            existing = self.db.find(
                ablation_name=self.config.name,
                locale=record.locale,
                method=record.method,
                similarity_type=record.similarity_type,
            )
            if existing and self.config.resume:
                # Skip if already completed
                if existing[0].status == "completed":
                    print(f"  Skipping (completed): {record.locale} / {record.method} / {record.similarity_type}")
                    ids.append(existing[0].id)
                    continue
                # Update existing planned/failed record
                record.id = existing[0].id
                self.db.update(record)
                ids.append(record.id)
            else:
                exp_id = self.db.insert(record)
                ids.append(exp_id)

        return ids

    def run_single(self, exp_id: int) -> bool:
        """
        Run a single experiment by ID.
        Returns True if successful, False otherwise.
        """
        record = self.db.get(exp_id)
        if not record:
            print(f"Experiment {exp_id} not found")
            return False

        if record.status == "completed" and self.config.resume:
            print(f"Experiment {exp_id} already completed, skipping")
            return True

        print(f"\n{'='*60}")
        print(f"Running: {record.locale} / {record.method} / {record.similarity_type}")
        print(f"{'='*60}")

        if self.config.dry_run:
            print("  [DRY RUN] Would run experiment")
            return True

        # Mark as running
        self.db.mark_running(exp_id)

        try:
            # Import here to avoid circular imports
            from merginguriel.run_merging_pipeline_refactored import MergingPipeline, MergeConfig

            # Build MergeConfig from experiment config
            exp_config = json.loads(record.config_json) if record.config_json else {}

            merged_models_dir = exp_config.get("merged_models_dir", "merged_models")

            merge_config = MergeConfig(
                mode=record.method,
                target_lang=record.locale,
                similarity_type=record.similarity_type,
                num_languages=record.num_languages,
                include_target=record.include_target,
                base_model=exp_config.get("model_family", "xlm-roberta-base"),
                base_model_dir=exp_config.get("models_root", "haryos_model"),
                batch_size=exp_config.get("batch_size", 16),
                max_seq_length=exp_config.get("max_seq_length", 128),
            )

            # Run the pipeline (merge + STS-B sanity check)
            pipeline = MergingPipeline(merge_config, merged_models_dir=merged_models_dir)
            pipeline.run()

            # Find the merged model directory
            from merginguriel.naming_config import naming_manager
            # num_merged is the number of models actually merged (num_languages - 1 since one is the base)
            num_merged = record.num_languages - 1
            merged_dir_name = naming_manager.get_merged_model_dir_name(
                experiment_type="merging",
                method=record.method,
                similarity_type=record.similarity_type,
                locale=record.locale,
                model_family=exp_config.get("model_family", "xlm-roberta-base"),
                num_merged=num_merged,
                include_target=record.include_target,
            )
            merged_model_path = Path(merged_models_dir) / merged_dir_name

            # Now run the actual MASSIVE intent classification evaluation
            from merginguriel.evaluate_specific_model import evaluate_specific_model

            # Create results directory for this experiment
            results_dir_name = naming_manager.get_results_dir_name(
                experiment_type="merging",
                method=record.method,
                similarity_type=record.similarity_type,
                locale=record.locale,
                model_family=exp_config.get("model_family", "xlm-roberta-base"),
                num_languages=record.num_languages,
                include_target=record.include_target,
            )
            results_dir = Path("results") / results_dir_name
            results_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Evaluating on MASSIVE intent classification ({record.locale})...")
            eval_results = evaluate_specific_model(
                model_name=str(merged_model_path),
                locale=record.locale,
                eval_folder=str(results_dir),
            )

            if eval_results and "performance" in eval_results:
                accuracy = eval_results["performance"]["accuracy"]
                self.db.mark_completed(exp_id, accuracy, str(results_dir))
                print(f"  Completed: accuracy = {accuracy:.4f}")
                return True
            else:
                self.db.mark_failed(exp_id, "No accuracy found in MASSIVE evaluation results")
                print("  Failed: No accuracy found in MASSIVE evaluation results")
                return False

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.db.mark_failed(exp_id, error_msg)
            print(f"  Failed: {e}")
            return False

    def run_all(self) -> Dict[str, int]:
        """
        Run all planned experiments.
        Returns dict with counts of completed, failed, skipped.
        """
        print(f"\n{'#'*60}")
        print(f"# Ablation: {self.config.name}")
        print(f"# {self.config.description}")
        print(f"{'#'*60}")

        # Register plans
        print("\nRegistering experiment plans...")
        exp_ids = self.register_plans()
        print(f"  Total experiments: {len(exp_ids)}")

        # Run each experiment
        stats = {"completed": 0, "failed": 0, "skipped": 0}

        for i, exp_id in enumerate(exp_ids, 1):
            record = self.db.get(exp_id)
            if record.status == "completed" and self.config.resume:
                stats["skipped"] += 1
                continue

            print(f"\n[{i}/{len(exp_ids)}]", end="")
            success = self.run_single(exp_id)
            if success:
                stats["completed"] += 1
            else:
                stats["failed"] += 1

        # Print summary
        print(f"\n{'='*60}")
        print("Ablation Summary")
        print(f"{'='*60}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed:    {stats['failed']}")
        print(f"  Skipped:   {stats['skipped']}")

        return stats

    def show_plan(self) -> None:
        """Print the planned experiments without running them."""
        records = self.plan()

        print(f"\nAblation: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Total runs: {len(records)}")
        print()

        print(f"{'Locale':<10} {'Method':<12} {'Similarity':<10} {'NumLang':<8} {'IncTar':<8}")
        print("-" * 50)

        for r in records:
            print(f"{r.locale:<10} {r.method:<12} {r.similarity_type:<10} {r.num_languages:<8} {str(r.include_target):<8}")


def main():
    """CLI entry point for ablation runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("config", type=Path, help="Path to ablation config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Plan but don't run")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip completed experiments")
    parser.add_argument("--plan-only", action="store_true", help="Just show the plan")
    parser.add_argument("--db", type=str, default=None, help="Override database path")

    args = parser.parse_args()

    # Load config
    config = AblationConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.dry_run:
        config.dry_run = True
    if args.no_resume:
        config.resume = False
    if args.db:
        config.db_path = args.db

    # Create runner
    runner = AblationRunner(config)

    if args.plan_only:
        runner.show_plan()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
