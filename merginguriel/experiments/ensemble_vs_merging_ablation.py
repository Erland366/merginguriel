"""
Unified ablation runner for Ensemble vs Merging experiments.

This runner handles both ensemble inference and parameter merging experiments,
enabling direct comparison between output averaging (ensemble) and
parameter averaging (merging) approaches.

Usage:
    python ensemble_vs_merging_ablation.py configs/ablations/ensemble_vs_merging.yaml
    python ensemble_vs_merging_ablation.py configs/ablations/ensemble_vs_merging.yaml --plan-only
    python ensemble_vs_merging_ablation.py configs/ablations/ensemble_vs_merging.yaml --max-experiments 4
"""

import itertools
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure project root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from merginguriel.experiments.db import ExperimentDB, ExperimentRecord


@dataclass
class UnifiedAblationConfig:
    """Configuration for ensemble vs merging ablation study."""

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
    resume: bool = True
    max_experiments: Optional[int] = None  # For validation runs

    @classmethod
    def from_yaml(cls, path: Path) -> "UnifiedAblationConfig":
        """Load config from YAML file."""
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
            return [self.fixed.copy()]

        sweep_keys = list(self.sweep.keys())
        sweep_values = [self.sweep[k] for k in sweep_keys]

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

        locales = self.fixed.get("locales", [None])
        if isinstance(locales, list) and len(locales) > 0:
            total *= len(locales)

        return total


class UnifiedAblationRunner:
    """Runs both ensemble and merging experiments for comparison."""

    def __init__(self, config: UnifiedAblationConfig):
        self.config = config
        self.db = ExperimentDB(config.db_path)

    def plan(self) -> List[ExperimentRecord]:
        """Plan all experiments without running them."""
        records = []
        configs = self.config.generate_configs()

        for exp_config in configs:
            # Get locales
            if "locales" in exp_config:
                loc = exp_config["locales"]
                locales = loc if isinstance(loc, list) else [loc]
            elif "locale" in exp_config:
                locales = [exp_config["locale"]]
            else:
                # Also check fixed config
                loc = self.config.fixed.get("locales", [""])
                locales = loc if isinstance(loc, list) else [loc]

            # Extract experiment_type dict (from sweep)
            exp_type_info = exp_config.get("experiment_type", {})
            if isinstance(exp_type_info, dict):
                experiment_type = exp_type_info.get("type", "merging")
                method = exp_type_info.get("method", "similarity")
            else:
                experiment_type = "merging"
                method = exp_config.get("method", "similarity")

            for locale in locales:
                record = ExperimentRecord(
                    ablation_name=self.config.name,
                    locale=locale,
                    method=method,
                    experiment_type=experiment_type,
                    similarity_type=exp_config.get("similarity_type",
                                                   self.config.fixed.get("similarity_type", "URIEL")),
                    num_languages=exp_config.get("num_languages", 3),
                    include_target=exp_config.get("include_target", False),
                    model_family=exp_config.get("model_family",
                                                self.config.fixed.get("model_family", "xlm-roberta-base")),
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
            # Check if already exists (now including experiment_type)
            existing = self.db.find(
                ablation_name=self.config.name,
                locale=record.locale,
                method=record.method,
                experiment_type=record.experiment_type,
                similarity_type=record.similarity_type,
                num_languages=record.num_languages,
            )

            # Also need to check include_target since db.find doesn't filter on it
            existing = [e for e in existing if e.include_target == record.include_target]

            if existing and self.config.resume:
                if existing[0].status == "completed":
                    print(f"  Skipping (completed): {record.experiment_type}/{record.method} on {record.locale}")
                    ids.append(existing[0].id)
                    continue
                record.id = existing[0].id
                self.db.update(record)
                ids.append(record.id)
            else:
                exp_id = self.db.insert(record)
                ids.append(exp_id)

        return ids

    def run_ensemble_experiment(self, record: ExperimentRecord) -> Dict[str, Any]:
        """Run an ensemble inference experiment."""
        from merginguriel.uriel_ensemble_inference import (
            load_similarity_weights,
            run_ensemble_inference,
            evaluate_ensemble,
        )
        from merginguriel.naming_config import naming_manager
        from datasets import load_dataset

        exp_config = json.loads(record.config_json) if record.config_json else {}

        # Get paths
        similarity_matrix_path = os.path.join(REPO_ROOT, "language_similarity_matrix_unified.csv")
        models_root = exp_config.get("models_root", self.config.fixed.get("models_root", "haryos_model"))
        local_models_dir = os.path.join(REPO_ROOT, models_root)

        # Detect available locales
        model_prefix = f"{record.model_family}_massive_k_"
        available_locales = []
        for item in os.listdir(local_models_dir):
            if os.path.isdir(os.path.join(local_models_dir, item)) and item.startswith(model_prefix):
                locale = item.replace(model_prefix, "")
                available_locales.append(locale)

        print(f"    Found {len(available_locales)} available models")

        # Load similarity weights
        models_and_weights = load_similarity_weights(
            similarity_matrix_path,
            available_locales,
            record.locale,
            num_languages=record.num_languages,
            include_target=record.include_target,
        )

        if not models_and_weights:
            raise ValueError(f"No models found for {record.locale}")

        # Load test data
        print(f"    Loading test data for {record.locale}...")
        dataset = load_dataset("AmazonScience/massive", record.locale, split="test", trust_remote_code=True)
        test_texts = dataset["utt"]
        test_labels = dataset["intent"]
        print(f"    Loaded {len(test_texts)} test examples")

        # Run ensemble inference
        predictions, metadata = run_ensemble_inference(
            models_and_weights, test_texts, record.method, record.model_family,
            models_root=local_models_dir
        )

        # Evaluate
        results = evaluate_ensemble(predictions, test_labels)

        # Create results directory
        mode_suffix = "IncTar" if record.include_target else "ExcTar"
        results_dir_name = f"ensemble_{record.method}_{record.similarity_type}_{record.locale}_{record.model_family}_{record.num_languages}lang_{mode_suffix}"
        results_dir = Path(self.config.results_base_dir) / results_dir_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        detailed_results = {
            "experiment_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "experiment_type": "ensemble",
                "target_language": record.locale,
                "voting_method": record.method,
                "num_models": len(models_and_weights),
                "num_examples": len(test_texts),
                "include_target": record.include_target,
                "mode": mode_suffix,
            },
            "models": {k: {"weight": v["weight"], "locale": v["locale"]}
                      for k, v in models_and_weights.items()},
            "performance": results,
        }

        results_file = results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        return {
            "accuracy": results["accuracy"],
            "results_dir": str(results_dir),
        }

    def run_merging_experiment(self, record: ExperimentRecord) -> Dict[str, Any]:
        """Run a parameter merging experiment."""
        from merginguriel.run_merging_pipeline_refactored import MergingPipeline, MergeConfig
        from merginguriel.evaluate_specific_model import evaluate_specific_model
        from merginguriel.naming_config import naming_manager

        exp_config = json.loads(record.config_json) if record.config_json else {}
        merged_models_dir = exp_config.get("merged_models_dir",
                                           self.config.fixed.get("merged_models_dir", "merged_models"))

        # Build MergeConfig
        merge_config = MergeConfig(
            mode=record.method,
            target_lang=record.locale,
            similarity_type=record.similarity_type,
            num_languages=record.num_languages,
            include_target=record.include_target,
            base_model=record.model_family,
            base_model_dir=exp_config.get("models_root", self.config.fixed.get("models_root", "haryos_model")),
            batch_size=exp_config.get("batch_size", 16),
            max_seq_length=exp_config.get("max_seq_length", 128),
            dataset_name=exp_config.get("dataset_name"),
            text_column=exp_config.get("text_column", "text"),
            num_fisher_examples=exp_config.get("num_fisher_examples", 1000),
            fisher_data_mode=exp_config.get("fisher_data_mode", "target"),
        )

        # Run merge pipeline
        pipeline = MergingPipeline(merge_config, merged_models_dir=merged_models_dir)
        pipeline.run()

        # Find merged model directory
        num_merged = record.num_languages - 1
        merged_dir_name = naming_manager.get_merged_model_dir_name(
            experiment_type="merging",
            method=record.method,
            similarity_type=record.similarity_type,
            locale=record.locale,
            model_family=record.model_family,
            num_merged=num_merged,
            include_target=record.include_target,
        )
        merged_model_path = Path(merged_models_dir) / merged_dir_name

        # Create results directory
        mode_suffix = "IncTar" if record.include_target else "ExcTar"
        results_dir_name = f"merging_{record.method}_{record.similarity_type}_{record.locale}_{record.model_family}_{record.num_languages}lang_{mode_suffix}"
        results_dir = Path(self.config.results_base_dir) / results_dir_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate on MASSIVE
        print(f"    Evaluating merged model on MASSIVE ({record.locale})...")
        eval_results = evaluate_specific_model(
            model_name=str(merged_model_path),
            locale=record.locale,
            eval_folder=str(results_dir),
        )

        if eval_results and "performance" in eval_results:
            return {
                "accuracy": eval_results["performance"]["accuracy"],
                "results_dir": str(results_dir),
            }
        else:
            raise ValueError("No accuracy found in evaluation results")

    def run_single(self, exp_id: int) -> bool:
        """Run a single experiment by ID."""
        record = self.db.get(exp_id)
        if not record:
            print(f"Experiment {exp_id} not found")
            return False

        if record.status == "completed" and self.config.resume:
            print(f"Experiment {exp_id} already completed, skipping")
            return True

        mode = "IncTar" if record.include_target else "ExcTar"
        print(f"\n{'='*70}")
        print(f"Running: [{record.experiment_type}] {record.method} on {record.locale} ({mode}, k={record.num_languages})")
        print(f"{'='*70}")

        if self.config.dry_run:
            print("  [DRY RUN] Would run experiment")
            return True

        self.db.mark_running(exp_id)

        try:
            if record.experiment_type == "ensemble":
                result = self.run_ensemble_experiment(record)
            else:  # merging
                result = self.run_merging_experiment(record)

            accuracy = result["accuracy"]
            results_dir = result["results_dir"]
            self.db.mark_completed(exp_id, accuracy, results_dir)
            print(f"  Completed: accuracy = {accuracy:.4f}")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.db.mark_failed(exp_id, error_msg)
            print(f"  Failed: {e}")
            return False

    def run_all(self) -> Dict[str, int]:
        """Run all planned experiments."""
        print(f"\n{'#'*70}")
        print(f"# Ablation: {self.config.name}")
        print(f"# {self.config.description}")
        print(f"{'#'*70}")

        print("\nRegistering experiment plans...")
        exp_ids = self.register_plans()
        print(f"  Total experiments: {len(exp_ids)}")

        if self.config.max_experiments:
            print(f"  Limited to: {self.config.max_experiments} experiments")
            exp_ids = exp_ids[:self.config.max_experiments]

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

        print(f"\n{'='*70}")
        print("Ablation Summary")
        print(f"{'='*70}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed:    {stats['failed']}")
        print(f"  Skipped:   {stats['skipped']}")

        return stats

    def show_plan(self) -> None:
        """Print the planned experiments."""
        records = self.plan()

        print(f"\nAblation: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Total runs: {len(records)}")
        print()

        print(f"{'Type':<10} {'Locale':<8} {'Method':<18} {'Sim':<8} {'K':<4} {'IncTar':<8}")
        print("-" * 60)

        for r in records:
            print(f"{r.experiment_type:<10} {r.locale:<8} {r.method:<18} {r.similarity_type:<8} {r.num_languages:<4} {str(r.include_target):<8}")

    def show_results(self) -> None:
        """Show results comparison table."""
        records = self.db.find(ablation_name=self.config.name, status="completed")

        if not records:
            print("No completed experiments found")
            return

        print(f"\n{'='*80}")
        print(f"Results: {self.config.name}")
        print(f"{'='*80}")

        # Group by locale and mode
        from collections import defaultdict
        results_table = defaultdict(dict)

        for r in records:
            mode = "IncTar" if r.include_target else "ExcTar"
            key = (r.locale, mode, r.num_languages)
            method_key = f"{r.experiment_type}_{r.method}"
            results_table[key][method_key] = r.accuracy

        # Print header
        methods = sorted(set(
            f"{r.experiment_type}_{r.method}" for r in records
        ))
        header = f"{'Locale':<8} {'Mode':<8} {'K':<4} " + " ".join(f"{m:<20}" for m in methods)
        print(header)
        print("-" * len(header))

        # Print rows
        for (locale, mode, k), method_results in sorted(results_table.items()):
            row = f"{locale:<8} {mode:<8} {k:<4} "
            row += " ".join(f"{method_results.get(m, 0):<20.4f}" for m in methods)
            print(row)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ensemble vs merging ablation")
    parser.add_argument("config", type=Path, help="Path to ablation config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Plan but don't run")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip completed experiments")
    parser.add_argument("--plan-only", action="store_true", help="Just show the plan")
    parser.add_argument("--show-results", action="store_true", help="Show results table")
    parser.add_argument("--db", type=str, default=None, help="Override database path")
    parser.add_argument("--max-experiments", type=int, default=None,
                       help="Limit number of experiments (for validation)")

    args = parser.parse_args()

    config = UnifiedAblationConfig.from_yaml(args.config)

    if args.dry_run:
        config.dry_run = True
    if args.no_resume:
        config.resume = False
    if args.db:
        config.db_path = args.db
    if args.max_experiments:
        config.max_experiments = args.max_experiments

    runner = UnifiedAblationRunner(config)

    if args.plan_only:
        runner.show_plan()
    elif args.show_results:
        runner.show_results()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
