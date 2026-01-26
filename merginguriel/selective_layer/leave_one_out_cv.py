"""
Leave-one-source-out cross-validation for layer ablation.

For each source language:
1. Hold it out as the "pseudo-target"
2. Merge remaining sources (with selective layer exclusion)
3. Evaluate merged model on holdout
4. Record which layers contributed positively

This enables identification of positive-transfer vs interference layers
without using any target language data.
"""

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from merginguriel.selective_layer.layer_masking import (
    get_ablation_points,
    LAYER_GROUPS,
    NUM_LAYERS,
)
from merginguriel.selective_layer.selective_merge import (
    find_best_source,
    run_selective_merge_experiment,
)


@dataclass
class LayerAblationResult:
    """Result of a single layer ablation experiment."""

    # Identity
    id: Optional[int] = None
    ablation_name: str = ""  # e.g., "layer_ablation_swke_v1"

    # CV fold info
    holdout_locale: str = ""
    remaining_sources: str = ""  # JSON list of source locales

    # Layer ablation info
    ablation_point: str = ""  # e.g., "exclude_layer_5", "exclude_group_top"
    exclude_layers: str = ""  # JSON list of layer indices

    # Best source for excluded layers
    best_source_locale: str = ""
    best_source_accuracy: float = 0.0

    # Results
    accuracy: float = 0.0
    baseline_accuracy: float = 0.0  # Accuracy with no layer exclusion
    delta: float = 0.0  # accuracy - baseline (positive = exclusion helped)

    # Transfer interpretation
    # positive delta = excluding hurt = layer has positive transfer
    # negative delta = excluding helped = layer causes interference
    transfer_interpretation: str = ""

    # Status
    status: str = "planned"  # planned | running | completed | failed
    error_message: str = ""

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    # Metadata
    config_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LayerAblationResult":
        known_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in dict(row).items() if k in known_fields}
        return cls(**data)


class LayerAblationDB:
    """SQLite database for layer ablation experiments."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS layer_ablation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ablation_name TEXT NOT NULL,
        holdout_locale TEXT NOT NULL,
        remaining_sources TEXT,
        ablation_point TEXT NOT NULL,
        exclude_layers TEXT,
        best_source_locale TEXT,
        best_source_accuracy REAL,
        accuracy REAL,
        baseline_accuracy REAL,
        delta REAL,
        transfer_interpretation TEXT,
        status TEXT DEFAULT 'planned',
        error_message TEXT,
        started_at TEXT,
        completed_at TEXT,
        config_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_layer_ablation_name ON layer_ablation(ablation_name);
    CREATE INDEX IF NOT EXISTS idx_layer_holdout ON layer_ablation(holdout_locale);
    CREATE INDEX IF NOT EXISTS idx_layer_point ON layer_ablation(ablation_point);
    CREATE INDEX IF NOT EXISTS idx_layer_status ON layer_ablation(status);
    """

    def __init__(self, db_path: str = "layer_ablation.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, record: LayerAblationResult) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO layer_ablation (
                    ablation_name, holdout_locale, remaining_sources,
                    ablation_point, exclude_layers, best_source_locale,
                    best_source_accuracy, accuracy, baseline_accuracy, delta,
                    transfer_interpretation, status, error_message,
                    started_at, completed_at, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.ablation_name, record.holdout_locale, record.remaining_sources,
                record.ablation_point, record.exclude_layers, record.best_source_locale,
                record.best_source_accuracy, record.accuracy, record.baseline_accuracy,
                record.delta, record.transfer_interpretation, record.status,
                record.error_message, record.started_at, record.completed_at,
                record.config_json
            ))
            return cursor.lastrowid

    def update(self, record: LayerAblationResult) -> None:
        if record.id is None:
            raise ValueError("Cannot update record without ID")

        with self._get_conn() as conn:
            conn.execute("""
                UPDATE layer_ablation SET
                    ablation_name = ?, holdout_locale = ?, remaining_sources = ?,
                    ablation_point = ?, exclude_layers = ?, best_source_locale = ?,
                    best_source_accuracy = ?, accuracy = ?, baseline_accuracy = ?,
                    delta = ?, transfer_interpretation = ?, status = ?,
                    error_message = ?, started_at = ?, completed_at = ?, config_json = ?
                WHERE id = ?
            """, (
                record.ablation_name, record.holdout_locale, record.remaining_sources,
                record.ablation_point, record.exclude_layers, record.best_source_locale,
                record.best_source_accuracy, record.accuracy, record.baseline_accuracy,
                record.delta, record.transfer_interpretation, record.status,
                record.error_message, record.started_at, record.completed_at,
                record.config_json, record.id
            ))

    def get(self, exp_id: int) -> Optional[LayerAblationResult]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM layer_ablation WHERE id = ?", (exp_id,)
            ).fetchone()
            return LayerAblationResult.from_row(row) if row else None

    def find(
        self,
        ablation_name: Optional[str] = None,
        holdout_locale: Optional[str] = None,
        ablation_point: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[LayerAblationResult]:
        conditions = []
        params = []

        if ablation_name:
            conditions.append("ablation_name = ?")
            params.append(ablation_name)
        if holdout_locale:
            conditions.append("holdout_locale = ?")
            params.append(holdout_locale)
        if ablation_point:
            conditions.append("ablation_point = ?")
            params.append(ablation_point)
        if status:
            conditions.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM layer_ablation"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [LayerAblationResult.from_row(row) for row in rows]

    def mark_running(self, exp_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE layer_ablation
                SET status = 'running', started_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), exp_id))

    def mark_completed(
        self,
        exp_id: int,
        accuracy: float,
        baseline_accuracy: float,
        best_source_locale: str,
        best_source_accuracy: float,
    ) -> None:
        delta = accuracy - baseline_accuracy
        interpretation = interpret_transfer(delta)

        with self._get_conn() as conn:
            conn.execute("""
                UPDATE layer_ablation
                SET status = 'completed', completed_at = ?,
                    accuracy = ?, baseline_accuracy = ?, delta = ?,
                    best_source_locale = ?, best_source_accuracy = ?,
                    transfer_interpretation = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(), accuracy, baseline_accuracy, delta,
                best_source_locale, best_source_accuracy, interpretation, exp_id
            ))

    def mark_failed(self, exp_id: int, error_message: str) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE layer_ablation
                SET status = 'failed', completed_at = ?, error_message = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), error_message, exp_id))

    def get_results_df(self, ablation_name: str) -> pd.DataFrame:
        """Get all completed results as a DataFrame."""
        records = self.find(ablation_name=ablation_name, status="completed")
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in records])

    def stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM layer_ablation").fetchone()[0]
            by_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM layer_ablation GROUP BY status"
            ).fetchall())
            return {"total": total, "by_status": by_status}


def interpret_transfer(delta: float, threshold: float = 0.005) -> str:
    """
    Interpret the transfer effect of a layer exclusion.

    Args:
        delta: accuracy - baseline_accuracy
        threshold: Minimum delta magnitude to consider significant

    Returns:
        "positive_transfer": Layer helps (excluding hurts)
        "interference": Layer hurts (excluding helps)
        "neutral": No significant effect
    """
    if delta < -threshold:
        return "positive_transfer"  # Excluding hurt performance
    elif delta > threshold:
        return "interference"  # Excluding helped performance
    else:
        return "neutral"


@dataclass
class LayerAblationConfig:
    """Configuration for a layer ablation study."""

    name: str
    description: str = ""

    # Target locale (for context, not used in CV)
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
    db_path: str = "layer_ablation.db"
    results_dir: str = "results/layer_ablation"

    # Execution
    dry_run: bool = False
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "LayerAblationConfig":
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
            db_path=ablation_data.get("db_path", "layer_ablation.db"),
            results_dir=ablation_data.get("results_dir", "results/layer_ablation"),
            dry_run=ablation_data.get("dry_run", False),
            resume=ablation_data.get("resume", True),
        )


class LeaveOneSourceOutCV:
    """
    Leave-one-source-out cross-validation for layer ablation.
    """

    def __init__(self, config: LayerAblationConfig):
        self.config = config
        self.db = LayerAblationDB(config.db_path)
        self.ablation_points = get_ablation_points()

        # Cache for baseline accuracies (holdout -> accuracy)
        self._baseline_cache: Dict[str, float] = {}

    def plan(self) -> List[LayerAblationResult]:
        """
        Plan all experiments without running them.
        """
        records = []

        for holdout in self.config.cv_source_locales:
            remaining = [s for s in self.config.cv_source_locales if s != holdout]

            for point_name, exclude_layers in self.ablation_points.items():
                record = LayerAblationResult(
                    ablation_name=self.config.name,
                    holdout_locale=holdout,
                    remaining_sources=json.dumps(remaining),
                    ablation_point=point_name,
                    exclude_layers=json.dumps(exclude_layers),
                    status="planned",
                    config_json=json.dumps({
                        "merge_method": self.config.merge_method,
                        "model_family": self.config.model_family,
                        "similarity_type": self.config.similarity_type,
                    }),
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
                holdout_locale=record.holdout_locale,
                ablation_point=record.ablation_point,
            )

            if existing and self.config.resume:
                # Skip if already exists
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
        """
        Run a single ablation experiment.
        """
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
            # Parse config
            exclude_layers = json.loads(record.exclude_layers)
            remaining_sources = json.loads(record.remaining_sources)

            # Find best source for excluded layers
            best_source, best_source_acc = find_best_source(
                holdout_locale=record.holdout_locale,
                source_locales=remaining_sources,
                nxn_matrix_path=self.config.nxn_matrix_path,
            )
            print(f"Best source for {record.holdout_locale}: {best_source} ({best_source_acc:.4f})")

            # Build model paths
            models_to_merge = [
                f"{project_root}/{self.config.models_root}/{self.config.model_family}_massive_k_{loc}"
                for loc in remaining_sources
            ]
            base_model_path = models_to_merge[0]
            best_source_path = f"{project_root}/{self.config.models_root}/{self.config.model_family}_massive_k_{best_source}"

            # Equal weights for simplicity
            weights = [1.0 / len(models_to_merge)] * len(models_to_merge)

            # Run selective merge
            merge_result = run_selective_merge_experiment(
                base_model_path=base_model_path,
                models_to_merge=models_to_merge,
                weights=weights,
                exclude_layers=exclude_layers,
                best_source_path=best_source_path if exclude_layers else None,
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

            accuracy = eval_results["performance"]["accuracy"] if eval_results else 0.0
            print(f"Accuracy on {record.holdout_locale}: {accuracy:.4f}")

            # Get baseline (no exclusion)
            baseline_accuracy = self._get_baseline(
                holdout_locale=record.holdout_locale,
                remaining_sources=remaining_sources,
                project_root=project_root,
            )

            # Mark completed
            self.db.mark_completed(
                exp_id=exp_id,
                accuracy=accuracy,
                baseline_accuracy=baseline_accuracy,
                best_source_locale=best_source,
                best_source_accuracy=best_source_acc,
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
        """Get baseline accuracy (no layer exclusion) for a holdout."""
        cache_key = holdout_locale

        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Check if baseline already computed in DB
        baseline_records = self.db.find(
            ablation_name=self.config.name,
            holdout_locale=holdout_locale,
            ablation_point="baseline_all_layers",
            status="completed",
        )

        if baseline_records:
            baseline = baseline_records[0].accuracy
            self._baseline_cache[cache_key] = baseline
            return baseline

        # Baseline not yet computed - return 0 (will be filled later)
        return 0.0

    def run_all(self, project_root: str) -> pd.DataFrame:
        """Run all planned experiments."""
        # Register plans
        exp_ids = self.register_plans()
        print(f"Registered {len(exp_ids)} experiments")

        # Run baseline first (for each holdout)
        print("\n=== Running baselines first ===")
        baseline_ids = [
            eid for eid in exp_ids
            if self.db.get(eid).ablation_point == "baseline_all_layers"
        ]
        for exp_id in baseline_ids:
            self.run_single(exp_id, project_root)

        # Run remaining experiments
        print("\n=== Running layer ablations ===")
        for exp_id in exp_ids:
            record = self.db.get(exp_id)
            if record.ablation_point != "baseline_all_layers":
                self.run_single(exp_id, project_root)

        # Return results
        return self.db.get_results_df(self.config.name)


def analyze_ablation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ablation results to identify positive/negative transfer layers.

    Args:
        results_df: DataFrame from run_full_ablation

    Returns:
        Summary DataFrame with layer transfer analysis
    """
    if results_df.empty:
        return pd.DataFrame()

    # Filter to completed experiments (exclude baseline)
    df = results_df[
        (results_df["status"] == "completed") &
        (results_df["ablation_point"] != "baseline_all_layers")
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Group by ablation point, average delta across holdouts
    summary = df.groupby("ablation_point").agg({
        "delta": ["mean", "std", "count"],
        "accuracy": "mean",
    }).round(4)

    summary.columns = ["delta_mean", "delta_std", "n_folds", "accuracy_mean"]
    summary = summary.reset_index()

    # Interpret transfer type based on mean delta
    summary["transfer_type"] = summary["delta_mean"].apply(
        lambda x: interpret_transfer(x, threshold=0.005)
    )

    # Sort by delta (most negative = strongest positive transfer)
    summary = summary.sort_values("delta_mean")

    return summary


def print_transfer_summary(summary_df: pd.DataFrame) -> None:
    """Print a human-readable transfer summary."""
    if summary_df.empty:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("LAYER TRANSFER ANALYSIS SUMMARY")
    print("=" * 70)
    print("\nInterpretation:")
    print("  - Negative delta = Excluding HURT performance = POSITIVE TRANSFER (keep merged)")
    print("  - Positive delta = Excluding HELPED performance = INTERFERENCE (don't merge)")
    print()

    # Positive transfer layers
    positive = summary_df[summary_df["transfer_type"] == "positive_transfer"]
    if not positive.empty:
        print("POSITIVE TRANSFER (should merge):")
        for _, row in positive.iterrows():
            print(f"  {row['ablation_point']:25s} delta={row['delta_mean']:+.4f} (+/- {row['delta_std']:.4f})")

    # Interference layers
    interference = summary_df[summary_df["transfer_type"] == "interference"]
    if not interference.empty:
        print("\nINTERFERENCE (should NOT merge):")
        for _, row in interference.iterrows():
            print(f"  {row['ablation_point']:25s} delta={row['delta_mean']:+.4f} (+/- {row['delta_std']:.4f})")

    # Neutral layers
    neutral = summary_df[summary_df["transfer_type"] == "neutral"]
    if not neutral.empty:
        print("\nNEUTRAL (no significant effect):")
        for _, row in neutral.iterrows():
            print(f"  {row['ablation_point']:25s} delta={row['delta_mean']:+.4f} (+/- {row['delta_std']:.4f})")

    print("=" * 70)
