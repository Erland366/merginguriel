"""
Leave-one-source-out cross-validation for best-source-base layer ablation.

For each source language:
1. Hold it out as the "pseudo-target"
2. Find the best single source for the holdout (from NxN matrix)
3. Merge remaining sources (excluding best source) using ONLY specified layers
   onto the best source as base model
4. Evaluate merged model on holdout
5. Record which layer subsets improve over best-source-alone baseline

This extends include-only merging by using a finetuned best source as the base
instead of pretrained XLM-RoBERTa-base, providing a much stronger baseline.
"""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from merginguriel.selective_layer.layer_masking import (
    get_include_only_ablation_points,
    NUM_LAYERS,
)
from merginguriel.selective_layer.leave_one_out_cv import (
    LayerAblationResult,
    LayerAblationDB,
    interpret_transfer,
)
from merginguriel.selective_layer.selective_merge import (
    find_best_source,
    run_best_source_base_merge_experiment,
)


@dataclass
class BestSourceBaseCVConfig:
    """Configuration for a best-source-base layer ablation study."""

    name: str
    description: str = ""

    # Target locale (for context, not used in CV directly)
    target_locale: str = ""

    # Source locales for leave-one-out CV
    cv_source_locales: List[str] = field(default_factory=list)

    # Model configuration
    model_family: str = "xlm-roberta-base"
    models_root: str = "haryos_model"

    # Merge configuration
    merge_method: str = "linear"
    similarity_type: str = "REAL"
    num_languages: int = 5

    # Paths
    nxn_matrix_path: str = ""
    db_path: str = "best_source_base_ablation.db"
    results_dir: str = "results/best_source_base"

    # Execution
    dry_run: bool = False
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "BestSourceBaseCVConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        ablation_data = data.get("ablation", data)
        fixed = ablation_data.get("fixed", {})

        return cls(
            name=ablation_data.get("name", path.stem),
            description=ablation_data.get("description", ""),
            target_locale=fixed.get("target_locale", ""),
            cv_source_locales=ablation_data.get("cv_source_locales", []),
            model_family=fixed.get("model_family", "xlm-roberta-base"),
            models_root=fixed.get("models_root", "haryos_model"),
            merge_method=fixed.get("method", "linear"),
            similarity_type=fixed.get("similarity_type", "REAL"),
            num_languages=fixed.get("num_languages", 5),
            nxn_matrix_path=fixed.get("nxn_matrix_path", ""),
            db_path=ablation_data.get("db_path", "best_source_base_ablation.db"),
            results_dir=ablation_data.get("results_dir", "results/best_source_base"),
            dry_run=ablation_data.get("dry_run", False),
            resume=ablation_data.get("resume", True),
        )


class LeaveOneSourceOutBestSourceBaseCV:
    """
    Leave-one-source-out cross-validation for best-source-base layer merging.

    Unlike include-only CV (pretrained base), this:
    - Finds the best single source for each holdout (via NxN matrix)
    - Uses that best source as the base model
    - Merges ONLY specified layers from remaining sources (excluding best source)
    - Non-merged layers retain best source's finetuned weights
    """

    def __init__(self, config: BestSourceBaseCVConfig):
        self.config = config
        self.db = LayerAblationDB(config.db_path)
        self.ablation_points = get_include_only_ablation_points()

        # Cache for baseline accuracies (holdout -> accuracy)
        self._baseline_cache: Dict[str, float] = {}

    def _find_best_source_for_holdout(
        self,
        holdout: str,
        remaining: List[str],
        project_root: str,
    ) -> tuple:
        """Find best source for a holdout locale using NxN matrix.

        Returns:
            (best_source_locale, best_source_accuracy, best_source_path,
             remaining_without_best)
        """
        nxn_path = self.config.nxn_matrix_path
        if not Path(nxn_path).is_absolute():
            nxn_path = f"{project_root}/{nxn_path}"

        best_source, best_acc = find_best_source(
            holdout_locale=holdout,
            source_locales=remaining,
            nxn_matrix_path=nxn_path,
        )

        remaining_without_best = [s for s in remaining if s != best_source]

        best_source_path = (
            f"{project_root}/{self.config.models_root}/"
            f"{self.config.model_family}_massive_k_{best_source}"
        )

        return best_source, best_acc, best_source_path, remaining_without_best

    def plan(self, project_root: str = ".") -> List[LayerAblationResult]:
        """Plan all experiments without running them."""
        records = []

        for holdout in self.config.cv_source_locales:
            remaining = [
                s for s in self.config.cv_source_locales if s != holdout
            ]

            # Find best source for this holdout
            best_source, best_acc, _, remaining_without_best = (
                self._find_best_source_for_holdout(holdout, remaining, project_root)
            )

            for point_name, include_layers in self.ablation_points.items():
                exclude_layers = [
                    i for i in range(NUM_LAYERS) if i not in include_layers
                ]

                record = LayerAblationResult(
                    ablation_name=self.config.name,
                    holdout_locale=holdout,
                    remaining_sources=json.dumps(remaining_without_best),
                    ablation_point=point_name,
                    exclude_layers=json.dumps(exclude_layers),
                    best_source_locale=best_source,
                    best_source_accuracy=best_acc,
                    status="planned",
                    config_json=json.dumps({
                        "mode": "best_source_base",
                        "include_layers": include_layers,
                        "best_source_locale": best_source,
                        "best_source_accuracy": best_acc,
                        "merge_method": self.config.merge_method,
                        "model_family": self.config.model_family,
                    }),
                )
                records.append(record)

        return records

    def register_plans(self, project_root: str = ".") -> List[int]:
        """Register all planned experiments in the database."""
        records = self.plan(project_root)
        ids = []

        for record in records:
            existing = self.db.find(
                ablation_name=self.config.name,
                holdout_locale=record.holdout_locale,
                ablation_point=record.ablation_point,
            )

            if existing and self.config.resume:
                ids.append(existing[0].id)
            else:
                exp_id = self.db.insert(record)
                ids.append(exp_id)

        return ids

    def run_single(
        self,
        exp_id: int,
        project_root: str,
    ) -> Optional[LayerAblationResult]:
        """Run a single best-source-base ablation experiment."""
        record = self.db.get(exp_id)
        if not record:
            print(f"Experiment {exp_id} not found")
            return None

        if record.status == "completed" and self.config.resume:
            print(f"Experiment {exp_id} already completed, skipping")
            return record

        self.db.mark_running(exp_id)
        print(f"\n{'='*60}")
        print(f"Running: {record.ablation_point} | Holdout: {record.holdout_locale}")
        print(f"{'='*60}")

        try:
            config_data = json.loads(record.config_json)
            include_layers = config_data["include_layers"]
            best_source_locale = config_data["best_source_locale"]
            remaining_without_best = json.loads(record.remaining_sources)

            if not remaining_without_best:
                raise ValueError(
                    f"No remaining sources after holding out {record.holdout_locale} "
                    f"and removing best source {best_source_locale}. "
                    f"Need at least 4 locales in cv_source_locales."
                )

            # Build model paths
            best_source_path = (
                f"{project_root}/{self.config.models_root}/"
                f"{self.config.model_family}_massive_k_{best_source_locale}"
            )
            models_to_merge = [
                f"{project_root}/{self.config.models_root}/"
                f"{self.config.model_family}_massive_k_{loc}"
                for loc in remaining_without_best
            ]

            # Equal weights for remaining sources
            weights = [1.0 / len(models_to_merge)] * len(models_to_merge)

            print(f"  Best source: {best_source_locale} (base model)")
            print(f"  Merging {len(models_to_merge)} remaining sources")
            print(f"  Include layers: {include_layers}")

            # Run best-source-base merge
            merge_result = run_best_source_base_merge_experiment(
                best_source_path=best_source_path,
                models_to_merge=models_to_merge,
                weights=weights,
                include_layers=include_layers,
                merge_method=self.config.merge_method,
            )

            # Evaluate on holdout
            from merginguriel.evaluate_specific_model import evaluate_specific_model

            with tempfile.TemporaryDirectory() as tmp_dir:
                merge_result.model.save_pretrained(tmp_dir)
                merge_result.tokenizer.save_pretrained(tmp_dir)

                eval_results = evaluate_specific_model(
                    model_name=tmp_dir,
                    locale=record.holdout_locale,
                )

            accuracy = (
                eval_results["performance"]["accuracy"] if eval_results else 0.0
            )
            print(f"Accuracy on {record.holdout_locale}: {accuracy:.4f}")

            # Get baseline (all layers included)
            baseline_accuracy = self._get_baseline(
                holdout_locale=record.holdout_locale,
            )

            # Mark completed
            self.db.mark_completed(
                exp_id=exp_id,
                accuracy=accuracy,
                baseline_accuracy=baseline_accuracy,
                best_source_locale=best_source_locale,
                best_source_accuracy=config_data["best_source_accuracy"],
            )

            return self.db.get(exp_id)

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            self.db.mark_failed(exp_id, error_msg)
            return None

    def _get_baseline(
        self,
        holdout_locale: str,
    ) -> float:
        """Get baseline accuracy (all layers included) for a holdout."""
        cache_key = holdout_locale

        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Check if baseline already computed in DB
        baseline_records = self.db.find(
            ablation_name=self.config.name,
            holdout_locale=holdout_locale,
            ablation_point="include_all_layers",
            status="completed",
        )

        if baseline_records:
            baseline = baseline_records[0].accuracy
            self._baseline_cache[cache_key] = baseline
            return baseline

        # Baseline not yet computed
        return 0.0

    def run_all(
        self,
        project_root: str,
        holdout_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run all planned experiments.

        Args:
            project_root: Path to project root
            holdout_filter: If provided, only run experiments for these holdouts.
                Plans are still registered for all holdouts (needed for
                remaining source computation).
        """
        exp_ids = self.register_plans(project_root)
        print(f"Registered {len(exp_ids)} experiments")

        # Filter to requested holdouts
        if holdout_filter:
            exp_ids = [
                eid for eid in exp_ids
                if self.db.get(eid).holdout_locale in holdout_filter
            ]
            print(f"Filtered to {len(exp_ids)} experiments for holdouts: {holdout_filter}")

        # Run baselines first (include_all_layers and include_no_layers)
        print("\n=== Running baselines first ===")
        baseline_points = {"include_all_layers", "include_no_layers"}
        baseline_ids = [
            eid for eid in exp_ids
            if self.db.get(eid).ablation_point in baseline_points
        ]
        for exp_id in baseline_ids:
            self.run_single(exp_id, project_root)

        # Run remaining experiments
        print("\n=== Running best-source-base ablations ===")
        for exp_id in exp_ids:
            record = self.db.get(exp_id)
            if record.ablation_point not in baseline_points:
                self.run_single(exp_id, project_root)

        return self.db.get_results_df(self.config.name)


def print_best_source_base_summary(summary_df: pd.DataFrame) -> None:
    """Print a human-readable best-source-base transfer summary."""
    if summary_df.empty:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("BEST-SOURCE-BASE LAYER MERGING ANALYSIS")
    print("=" * 70)
    print("\nInterpretation (delta = subset_accuracy - full_merge_accuracy):")
    print("  - Positive delta = subset BEATS full merge (some layers cause interference)")
    print("  - Negative delta = subset is worse (excluded layers were needed)")
    print("  - Compare to include_no_layers for best-source-alone baseline")
    print("  - Base model = best single source (finetuned), not pretrained")
    print()

    # Sort by delta descending (best subsets first)
    sorted_df = summary_df.sort_values("delta_mean", ascending=False)

    print(f"{'Ablation Point':<30s} {'Delta':>8s} {'Std':>8s} {'Accuracy':>10s} {'N':>4s}")
    print("-" * 64)
    for _, row in sorted_df.iterrows():
        point = row["ablation_point"]
        delta = row["delta_mean"]
        std = row["delta_std"]
        acc = row["accuracy_mean"]
        n = int(row["n_folds"])

        marker = ""
        if delta > 0.005:
            marker = " << BEATS FULL MERGE"
        elif point == "include_no_layers":
            marker = " (best-source-alone baseline)"

        print(
            f"{point:<30s} {delta:>+.4f} {std:>8.4f} {acc:>10.4f} {n:>4d}{marker}"
        )

    print("=" * 70)
