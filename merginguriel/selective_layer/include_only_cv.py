"""
Leave-one-source-out cross-validation for include-only layer ablation.

For each source language:
1. Hold it out as the "pseudo-target"
2. Merge remaining sources using ONLY specified layers (pretrained base for the rest)
3. Evaluate merged model on holdout
4. Record which layer subsets carry useful cross-lingual task knowledge

This tests the LinguaMap hypothesis: different layer phases (early/mid/late)
have different roles, and merging only the right subset can beat full merging.
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
    run_include_only_merge_experiment,
)


@dataclass
class IncludeOnlyCVConfig:
    """Configuration for an include-only layer ablation study."""

    name: str
    description: str = ""

    # Target locale (for context, not used in CV directly)
    target_locale: str = ""

    # Source locales for leave-one-out CV
    cv_source_locales: List[str] = field(default_factory=list)

    # Model configuration
    pretrained_model_name: str = "xlm-roberta-base"
    model_family: str = "xlm-roberta-base"
    models_root: str = "haryos_model"

    # Merge configuration
    merge_method: str = "linear"
    similarity_type: str = "REAL"
    num_languages: int = 5

    # Paths
    nxn_matrix_path: str = ""
    db_path: str = "include_only_ablation.db"
    results_dir: str = "results/include_only"

    # Execution
    dry_run: bool = False
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "IncludeOnlyCVConfig":
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
            pretrained_model_name=fixed.get(
                "pretrained_model_name", "xlm-roberta-base"
            ),
            model_family=fixed.get("model_family", "xlm-roberta-base"),
            models_root=fixed.get("models_root", "haryos_model"),
            merge_method=fixed.get("method", "linear"),
            similarity_type=fixed.get("similarity_type", "REAL"),
            num_languages=fixed.get("num_languages", 5),
            nxn_matrix_path=fixed.get("nxn_matrix_path", ""),
            db_path=ablation_data.get("db_path", "include_only_ablation.db"),
            results_dir=ablation_data.get("results_dir", "results/include_only"),
            dry_run=ablation_data.get("dry_run", False),
            resume=ablation_data.get("resume", True),
        )


class LeaveOneSourceOutIncludeOnlyCV:
    """
    Leave-one-source-out cross-validation for include-only layer merging.

    Unlike the exclude-based CV (LeaveOneSourceOutCV), this:
    - Uses pretrained xlm-roberta-base as the base model
    - Merges ONLY the specified layers (non-merged = pretrained)
    - Does NOT copy excluded layers from best source
    - Tests which layer subsets carry useful cross-lingual task knowledge
    """

    def __init__(self, config: IncludeOnlyCVConfig):
        self.config = config
        self.db = LayerAblationDB(config.db_path)
        self.ablation_points = get_include_only_ablation_points()

        # Cache for baseline accuracies (holdout -> accuracy)
        self._baseline_cache: Dict[str, float] = {}

    def plan(self) -> List[LayerAblationResult]:
        """Plan all experiments without running them."""
        records = []

        for holdout in self.config.cv_source_locales:
            remaining = [
                s for s in self.config.cv_source_locales if s != holdout
            ]

            for point_name, include_layers in self.ablation_points.items():
                # Convert include_layers to exclude_layers for DB storage
                exclude_layers = [
                    i for i in range(NUM_LAYERS) if i not in include_layers
                ]

                record = LayerAblationResult(
                    ablation_name=self.config.name,
                    holdout_locale=holdout,
                    remaining_sources=json.dumps(remaining),
                    ablation_point=point_name,
                    exclude_layers=json.dumps(exclude_layers),
                    best_source_locale="",  # Not used in include-only mode
                    best_source_accuracy=0.0,
                    status="planned",
                    config_json=json.dumps({
                        "mode": "include_only",
                        "include_layers": include_layers,
                        "merge_method": self.config.merge_method,
                        "model_family": self.config.model_family,
                        "pretrained_model_name": self.config.pretrained_model_name,
                    }),
                )
                records.append(record)

        return records

    def register_plans(self) -> List[int]:
        """Register all planned experiments in the database."""
        records = self.plan()
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
        """Run a single include-only ablation experiment."""
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
            remaining_sources = json.loads(record.remaining_sources)

            if not remaining_sources:
                raise ValueError(
                    f"No remaining sources after holding out {record.holdout_locale}. "
                    f"Need at least 3 locales in cv_source_locales."
                )

            # Build model paths
            models_to_merge = [
                f"{project_root}/{self.config.models_root}/"
                f"{self.config.model_family}_massive_k_{loc}"
                for loc in remaining_sources
            ]

            # Equal weights
            weights = [1.0 / len(models_to_merge)] * len(models_to_merge)

            # Run include-only merge
            merge_result = run_include_only_merge_experiment(
                pretrained_model_name=self.config.pretrained_model_name,
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
                remaining_sources=remaining_sources,
                project_root=project_root,
            )

            # Mark completed (best_source fields unused in include-only)
            self.db.mark_completed(
                exp_id=exp_id,
                accuracy=accuracy,
                baseline_accuracy=baseline_accuracy,
                best_source_locale="N/A",
                best_source_accuracy=0.0,
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
        remaining_sources: List[str],
        project_root: str,
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

    def run_all(self, project_root: str) -> pd.DataFrame:
        """Run all planned experiments."""
        exp_ids = self.register_plans()
        print(f"Registered {len(exp_ids)} experiments")

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
        print("\n=== Running include-only ablations ===")
        for exp_id in exp_ids:
            record = self.db.get(exp_id)
            if record.ablation_point not in baseline_points:
                self.run_single(exp_id, project_root)

        return self.db.get_results_df(self.config.name)


def print_include_only_summary(summary_df: pd.DataFrame) -> None:
    """Print a human-readable include-only transfer summary."""
    if summary_df.empty:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("INCLUDE-ONLY LAYER MERGING ANALYSIS")
    print("=" * 70)
    print("\nInterpretation (delta = subset_accuracy - full_merge_accuracy):")
    print("  - Positive delta = subset BEATS full merge (other layers cause interference)")
    print("  - Negative delta = subset is worse (excluded layers were needed)")
    print("  - Compare to include_no_layers for pretrained-only lower bound")
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
            marker = " (pretrained-only baseline)"

        print(
            f"{point:<30s} {delta:>+.4f} {std:>8.4f} {acc:>10.4f} {n:>4d}{marker}"
        )

    print("=" * 70)
