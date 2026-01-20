"""
SQLite database for experiment tracking.

Provides persistent storage for experiment metadata and results,
enabling queries across all experiments.
"""

import sqlite3
import json
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ExperimentRecord:
    """A single experiment record."""

    # Identity
    id: Optional[int] = None
    ablation_name: Optional[str] = None  # e.g., "similarity_type_ablation"

    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Configuration
    locale: str = ""
    method: str = ""
    similarity_type: str = ""
    num_languages: int = 0
    include_target: bool = False
    model_family: str = ""

    # Results
    accuracy: Optional[float] = None
    status: str = "planned"  # planned | running | completed | failed
    results_dir: Optional[str] = None
    error_message: Optional[str] = None

    # Full config as JSON
    config_json: Optional[str] = None

    # Notes
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ExperimentRecord":
        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in dict(row).items() if k in known_fields}
        return cls(**data)


class ExperimentDB:
    """SQLite database for tracking experiments."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ablation_name TEXT,
        started_at TEXT,
        completed_at TEXT,
        locale TEXT NOT NULL,
        method TEXT NOT NULL,
        similarity_type TEXT,
        num_languages INTEGER,
        include_target INTEGER,
        model_family TEXT,
        accuracy REAL,
        status TEXT DEFAULT 'planned',
        results_dir TEXT,
        error_message TEXT,
        config_json TEXT,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_locale ON experiments(locale);
    CREATE INDEX IF NOT EXISTS idx_method ON experiments(method);
    CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
    CREATE INDEX IF NOT EXISTS idx_ablation ON experiments(ablation_name);
    """

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def insert(self, record: ExperimentRecord) -> int:
        """Insert a new experiment record. Returns the new ID."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO experiments (
                    ablation_name, started_at, completed_at, locale, method,
                    similarity_type, num_languages, include_target, model_family,
                    accuracy, status, results_dir, error_message, config_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.ablation_name, record.started_at, record.completed_at,
                record.locale, record.method, record.similarity_type,
                record.num_languages, int(record.include_target), record.model_family,
                record.accuracy, record.status, record.results_dir,
                record.error_message, record.config_json, record.notes
            ))
            return cursor.lastrowid

    def update(self, record: ExperimentRecord) -> None:
        """Update an existing experiment record."""
        if record.id is None:
            raise ValueError("Cannot update record without ID")

        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments SET
                    ablation_name = ?, started_at = ?, completed_at = ?,
                    locale = ?, method = ?, similarity_type = ?,
                    num_languages = ?, include_target = ?, model_family = ?,
                    accuracy = ?, status = ?, results_dir = ?,
                    error_message = ?, config_json = ?, notes = ?
                WHERE id = ?
            """, (
                record.ablation_name, record.started_at, record.completed_at,
                record.locale, record.method, record.similarity_type,
                record.num_languages, int(record.include_target), record.model_family,
                record.accuracy, record.status, record.results_dir,
                record.error_message, record.config_json, record.notes,
                record.id
            ))

    def get(self, exp_id: int) -> Optional[ExperimentRecord]:
        """Get a single experiment by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
            return ExperimentRecord.from_row(row) if row else None

    def delete(self, exp_id: int) -> None:
        """Delete an experiment by ID."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a raw SQL query and return results."""
        with self._get_conn() as conn:
            return conn.execute(sql, params).fetchall()

    def find(
        self,
        locale: Optional[str] = None,
        method: Optional[str] = None,
        similarity_type: Optional[str] = None,
        status: Optional[str] = None,
        ablation_name: Optional[str] = None,
        num_languages: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[ExperimentRecord]:
        """Find experiments matching the given criteria."""
        conditions = []
        params = []

        if locale:
            conditions.append("locale = ?")
            params.append(locale)
        if method:
            conditions.append("method = ?")
            params.append(method)
        if similarity_type:
            conditions.append("similarity_type = ?")
            params.append(similarity_type)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if ablation_name:
            conditions.append("ablation_name = ?")
            params.append(ablation_name)
        if num_languages is not None:
            conditions.append("num_languages = ?")
            params.append(num_languages)

        sql = "SELECT * FROM experiments"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {limit}"

        with self._get_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [ExperimentRecord.from_row(row) for row in rows]

    def get_all(self, limit: Optional[int] = None) -> List[ExperimentRecord]:
        """Get all experiments."""
        return self.find(limit=limit)

    # -------------------------------------------------------------------------
    # Aggregation Methods
    # -------------------------------------------------------------------------

    def summary_by_locale(self) -> List[Dict[str, Any]]:
        """Get summary statistics grouped by locale."""
        sql = """
            SELECT
                locale,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                AVG(CASE WHEN status = 'completed' THEN accuracy END) as avg_accuracy,
                MAX(CASE WHEN status = 'completed' THEN accuracy END) as best_accuracy
            FROM experiments
            GROUP BY locale
            ORDER BY locale
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(row) for row in rows]

    def summary_by_method(self) -> List[Dict[str, Any]]:
        """Get summary statistics grouped by method."""
        sql = """
            SELECT
                method,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                AVG(CASE WHEN status = 'completed' THEN accuracy END) as avg_accuracy
            FROM experiments
            GROUP BY method
            ORDER BY avg_accuracy DESC
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(row) for row in rows]

    def best_config_per_locale(self) -> List[Dict[str, Any]]:
        """Get the best configuration for each locale."""
        sql = """
            SELECT
                e.locale,
                e.method,
                e.similarity_type,
                e.num_languages,
                e.accuracy
            FROM experiments e
            INNER JOIN (
                SELECT locale, MAX(accuracy) as max_acc
                FROM experiments
                WHERE status = 'completed'
                GROUP BY locale
            ) best ON e.locale = best.locale AND e.accuracy = best.max_acc
            ORDER BY e.locale
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(row) for row in rows]

    def ablation_comparison(self, ablation_name: str) -> List[Dict[str, Any]]:
        """Get comparison of results for a specific ablation study."""
        sql = """
            SELECT
                locale,
                method,
                similarity_type,
                num_languages,
                include_target,
                accuracy,
                status
            FROM experiments
            WHERE ablation_name = ?
            ORDER BY locale, accuracy DESC
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql, (ablation_name,)).fetchall()
            return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def mark_running(self, exp_id: int) -> None:
        """Mark an experiment as running."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments
                SET status = 'running', started_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), exp_id))

    def mark_completed(self, exp_id: int, accuracy: float, results_dir: str) -> None:
        """Mark an experiment as completed with results."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments
                SET status = 'completed', completed_at = ?, accuracy = ?, results_dir = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), accuracy, results_dir, exp_id))

    def mark_failed(self, exp_id: int, error_message: str) -> None:
        """Mark an experiment as failed."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments
                SET status = 'failed', completed_at = ?, error_message = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), error_message, exp_id))

    def export_csv(self, output_path: str) -> None:
        """Export all experiments to CSV."""
        import csv

        records = self.get_all()
        if not records:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].to_dict().keys())
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

    def stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            by_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM experiments GROUP BY status"
            ).fetchall())

            return {
                "total": total,
                "by_status": by_status,
                "db_path": str(self.db_path),
            }
