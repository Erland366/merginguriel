"""
Direct best-source-base layer ablation (no cross-validation).

For each target locale:
1. Find the best single source (from NxN matrix)
2. For each ablation point (layer subset):
   - Merge remaining sources onto best source using ONLY specified layers
   - Evaluate merged model directly on the target locale
3. Compare each subset to best-source-alone baseline

No CV holdouts — evaluates on the actual target.
"""

import json
import tempfile
from dataclasses import dataclass, field
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
class DirectBestSourceBaseConfig:
    """Configuration for direct best-source-base layer ablation."""

    name: str
    description: str = ""

    # Target locale to evaluate on
    target_locale: str = ""

    # Source locales (all available)
    source_locales: List[str] = field(default_factory=list)

    # Model configuration
    model_family: str = "xlm-roberta-base"
    models_root: str = "haryos_model"

    # Merge configuration
    merge_method: str = "linear"

    # Paths
    nxn_matrix_path: str = ""
    db_path: str = "direct_best_source_base.db"
    results_dir: str = "results/best_source_base_direct"

    # Execution
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "DirectBestSourceBaseConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        ablation_data = data.get("ablation", data)
        fixed = ablation_data.get("fixed", {})

        return cls(
            name=ablation_data.get("name", path.stem),
            description=ablation_data.get("description", ""),
            target_locale=fixed.get("target_locale", ""),
            source_locales=ablation_data.get("source_locales", []),
            model_family=fixed.get("model_family", "xlm-roberta-base"),
            models_root=fixed.get("models_root", "haryos_model"),
            merge_method=fixed.get("method", "linear"),
            nxn_matrix_path=fixed.get("nxn_matrix_path", ""),
            db_path=ablation_data.get("db_path", "direct_best_source_base.db"),
            results_dir=ablation_data.get("results_dir", "results/best_source_base_direct"),
            resume=ablation_data.get("resume", True),
        )


class DirectBestSourceBaseAblation:
    """
    Direct best-source-base layer ablation.

    No cross-validation. Finds best source for the target locale,
    merges layer subsets from remaining sources, evaluates on target.
    """

    def __init__(self, config: DirectBestSourceBaseConfig):
        self.config = config
        self.db = LayerAblationDB(config.db_path)
        self.ablation_points = get_include_only_ablation_points()
        self._baseline_accuracy: Optional[float] = None

    def _find_best_source(self, project_root: str) -> tuple:
        """Find best source for the target locale.

        Returns:
            (best_source_locale, best_accuracy, best_source_path, remaining_sources)
        """
        nxn_path = self.config.nxn_matrix_path
        if not Path(nxn_path).is_absolute():
            nxn_path = f"{project_root}/{nxn_path}"

        best_source, best_acc = find_best_source(
            holdout_locale=self.config.target_locale,
            source_locales=self.config.source_locales,
            nxn_matrix_path=nxn_path,
        )

        remaining = [s for s in self.config.source_locales if s != best_source]

        best_source_path = (
            f"{project_root}/{self.config.models_root}/"
            f"{self.config.model_family}_massive_k_{best_source}"
        )

        return best_source, best_acc, best_source_path, remaining

    def plan(self, project_root: str = ".") -> List[LayerAblationResult]:
        """Plan all experiments."""
        best_source, best_acc, _, remaining = self._find_best_source(project_root)

        records = []
        for point_name, include_layers in self.ablation_points.items():
            exclude_layers = [
                i for i in range(NUM_LAYERS) if i not in include_layers
            ]

            record = LayerAblationResult(
                ablation_name=self.config.name,
                holdout_locale=self.config.target_locale,
                remaining_sources=json.dumps(remaining),
                ablation_point=point_name,
                exclude_layers=json.dumps(exclude_layers),
                best_source_locale=best_source,
                best_source_accuracy=best_acc,
                status="planned",
                config_json=json.dumps({
                    "mode": "best_source_base_direct",
                    "include_layers": include_layers,
                    "best_source_locale": best_source,
                    "best_source_accuracy": best_acc,
                    "merge_method": self.config.merge_method,
                    "model_family": self.config.model_family,
                    "target_locale": self.config.target_locale,
                }),
            )
            records.append(record)

        return records

    def register_plans(self, project_root: str = ".") -> List[int]:
        """Register planned experiments in the database."""
        records = self.plan(project_root)
        ids = []

        for record in records:
            existing = self.db.find(
                ablation_name=self.config.name,
                holdout_locale=self.config.target_locale,
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
        """Run a single ablation experiment."""
        record = self.db.get(exp_id)
        if not record:
            print(f"Experiment {exp_id} not found")
            return None

        if record.status == "completed" and self.config.resume:
            print(f"Experiment {exp_id} already completed, skipping")
            return record

        self.db.mark_running(exp_id)
        print(f"\n{'='*60}")
        print(f"Running: {record.ablation_point} | Target: {self.config.target_locale}")
        print(f"{'='*60}")

        try:
            config_data = json.loads(record.config_json)
            include_layers = config_data["include_layers"]
            best_source_locale = config_data["best_source_locale"]
            remaining = json.loads(record.remaining_sources)

            # Build model paths
            best_source_path = (
                f"{project_root}/{self.config.models_root}/"
                f"{self.config.model_family}_massive_k_{best_source_locale}"
            )
            models_to_merge = [
                f"{project_root}/{self.config.models_root}/"
                f"{self.config.model_family}_massive_k_{loc}"
                for loc in remaining
            ]

            # Equal weights
            weights = [1.0 / len(models_to_merge)] * len(models_to_merge)

            print(f"  Best source: {best_source_locale} (base model)")
            print(f"  Merging {len(models_to_merge)} remaining sources")
            print(f"  Include layers: {include_layers}")

            # Run merge
            merge_result = run_best_source_base_merge_experiment(
                best_source_path=best_source_path,
                models_to_merge=models_to_merge,
                weights=weights,
                include_layers=include_layers,
                merge_method=self.config.merge_method,
            )

            # Evaluate on target
            from merginguriel.evaluate_specific_model import evaluate_specific_model

            with tempfile.TemporaryDirectory() as tmp_dir:
                merge_result.model.save_pretrained(tmp_dir)
                merge_result.tokenizer.save_pretrained(tmp_dir)

                eval_results = evaluate_specific_model(
                    model_name=tmp_dir,
                    locale=self.config.target_locale,
                )

            accuracy = (
                eval_results["performance"]["accuracy"] if eval_results else 0.0
            )
            print(f"Accuracy on {self.config.target_locale}: {accuracy:.4f}")

            # Baseline = include_no_layers (best source alone)
            baseline_accuracy = self._get_baseline()

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

    def _get_baseline(self) -> float:
        """Get baseline accuracy (include_no_layers = best source alone)."""
        if self._baseline_accuracy is not None:
            return self._baseline_accuracy

        baseline_records = self.db.find(
            ablation_name=self.config.name,
            holdout_locale=self.config.target_locale,
            ablation_point="include_no_layers",
            status="completed",
        )

        if baseline_records:
            self._baseline_accuracy = baseline_records[0].accuracy
            return self._baseline_accuracy

        return 0.0

    def run_all(self, project_root: str) -> pd.DataFrame:
        """Run all experiments."""
        exp_ids = self.register_plans(project_root)
        print(f"Registered {len(exp_ids)} experiments for {self.config.target_locale}")

        # Run include_no_layers first (baseline)
        print("\n=== Running baseline (best source alone) ===")
        baseline_id = None
        for exp_id in exp_ids:
            record = self.db.get(exp_id)
            if record.ablation_point == "include_no_layers":
                baseline_id = exp_id
                self.run_single(exp_id, project_root)
                break

        # Fix baseline's own delta to 0 (it IS the baseline)
        if baseline_id:
            baseline_record = self.db.get(baseline_id)
            if baseline_record and baseline_record.status == "completed":
                self._baseline_accuracy = baseline_record.accuracy
                self.db.mark_completed(
                    exp_id=baseline_id,
                    accuracy=baseline_record.accuracy,
                    baseline_accuracy=baseline_record.accuracy,
                    best_source_locale=baseline_record.best_source_locale,
                    best_source_accuracy=baseline_record.best_source_accuracy,
                )

        # Run remaining
        print("\n=== Running layer ablations ===")
        for exp_id in exp_ids:
            record = self.db.get(exp_id)
            if record.ablation_point != "include_no_layers":
                self.run_single(exp_id, project_root)

        return self.db.get_results_df(self.config.name)


def analyze_direct_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze direct ablation results.

    Same grouping as analyze_ablation_results but handles n=1 per point.
    """
    if results_df.empty:
        return pd.DataFrame()

    df = results_df[results_df["status"] == "completed"].copy()
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("ablation_point").agg({
        "delta": ["mean", "std", "count"],
        "accuracy": "mean",
    }).round(4)

    summary.columns = ["delta_mean", "delta_std", "n_folds", "accuracy_mean"]
    summary = summary.reset_index()

    # Classify transfer
    summary["transfer_type"] = summary["delta_mean"].apply(
        lambda d: interpret_transfer(d)
    )

    return summary


def print_direct_summary(summary_df: pd.DataFrame, target_locale: str = "") -> None:
    """Print direct best-source-base results."""
    if summary_df.empty:
        print("No results to summarize")
        return

    header = "DIRECT BEST-SOURCE-BASE LAYER ABLATION"
    if target_locale:
        header += f" — {target_locale}"

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)
    print("\nDelta = accuracy - best_source_alone_accuracy")
    print("  Positive delta = merging this layer subset HELPS over best source alone")
    print("  Negative delta = merging this layer subset HURTS (interference)")
    print()

    sorted_df = summary_df.sort_values("accuracy_mean", ascending=False)

    print(f"{'Ablation Point':<30s} {'Accuracy':>10s} {'Delta':>8s}")
    print("-" * 52)
    for _, row in sorted_df.iterrows():
        point = row["ablation_point"]
        acc = row["accuracy_mean"]
        delta = row["delta_mean"]

        marker = ""
        if point == "include_no_layers":
            marker = " (best source alone)"
        elif point == "include_all_layers":
            marker = " (full merge)"
        elif delta > 0.005:
            marker = " *"

        print(
            f"{point:<30s} {acc:>10.4f} {delta:>+.4f}{marker}"
        )

    print("=" * 70)
