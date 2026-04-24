"""
Utility functions for MergingUriel results analysis and plotting.

This module provides helper functions for data conversion, formatting,
and common operations used across the plotting package.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def safe_float(value: Any) -> float:
    """Convert value to float, returning 0.0 for invalid/missing values."""
    if pd.isna(value) or value == "" or value is None:
        return 0.0
    return float(value)


def maybe_float(value: Any) -> Optional[float]:
    """Return float(value) or None when not castable/empty."""
    try:
        if value in ("", None):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def extract_baselines(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """Return the best available baseline and all per-family baselines for a row."""
    baseline_map: Dict[str, float] = {}

    for col in row.index:
        if isinstance(col, str) and col.startswith("baseline_"):
            raw_val = row[col]
            if raw_val != "" and raw_val is not None and not pd.isna(raw_val):
                try:
                    baseline_map[col.replace("baseline_", "")] = float(raw_val)
                except Exception:
                    continue

    baseline_candidates = list(baseline_map.values())
    raw_baseline = row.get("baseline") if isinstance(row, (dict, pd.Series)) else None
    if raw_baseline not in (None, "") and not pd.isna(raw_baseline):
        try:
            baseline_candidates.append(float(raw_baseline))
        except Exception:
            pass

    best_baseline = max(baseline_candidates) if baseline_candidates else 0.0
    return best_baseline, baseline_map


def format_method_key_for_filename(
    method_key: str,
    model_family: Optional[str] = None,
    similarity_type: Optional[str] = None,
) -> str:
    """Convert method keys to clean filenames."""
    return method_key


def format_method_key_for_display(
    method_key: str,
    model_family: Optional[str] = None,
    similarity_type: Optional[str] = None,
) -> str:
    """Friendly display name for method keys with model family and merged count."""
    match = re.search(r"_(\d+)lang(?:_(?:IncTar|ExcTar))?$", method_key)
    if match:
        base = method_key[: match.start()]
        if base.startswith("ties"):
            base = base.replace("_", "")
        elif base.startswith("task_arithmetic"):
            base = base.replace("_", "")
        else:
            base = base.replace("_", " ").title()
        extra_info = []
        if model_family and model_family != "unknown":
            extra_info.append(model_family)
        if similarity_type and similarity_type != "unknown":
            extra_info.append(similarity_type)
        extra_info.append(f"Merged {match.group(1)}")
        return f"{base} ({', '.join(extra_info)})"

    display = method_key.replace("_", " ").title()
    if display.startswith("Ties"):
        display = display.replace(" ", "")
    elif display.startswith("Task Arithmetic"):
        display = display.replace(" ", "")
    extra_info = []
    if model_family and model_family != "unknown":
        extra_info.append(model_family)
    if similarity_type and similarity_type != "unknown":
        extra_info.append(similarity_type)
    if extra_info:
        return f"{display} ({', '.join(extra_info)})"
    return display


def get_method_num_language_set(summary_df: pd.DataFrame, method: str) -> set:
    """Return the set of num_languages associated with a given method column."""
    counts = set()

    match = re.search(r"_(\d+)lang(?:_(?:IncTar|ExcTar))?$", method)
    if match:
        try:
            counts.add(int(match.group(1)))
        except ValueError:
            pass

    for _, row in summary_df.iterrows():
        raw_map = row.get("num_languages_map")
        mapping = {}
        if isinstance(raw_map, str) and raw_map:
            try:
                mapping = json.loads(raw_map)
            except Exception:
                mapping = {}
        elif isinstance(raw_map, dict):
            mapping = raw_map

        value = mapping.get(method)
        if isinstance(value, (int, float)):
            counts.add(int(value))

    return counts


def infer_model_family_from_method_key(method_key: str) -> Optional[str]:
    """Attempt to infer the model family portion from a method key string."""
    parts = method_key.split("_")
    for part in parts:
        if "-" in part:
            return part
    return None


def get_method_model_family(
    method: str, experiment_results: Dict[str, Dict]
) -> str:
    """Get the model family name for a method."""
    inferred = infer_model_family_from_method_key(method)
    if inferred:
        return inferred

    method_base = re.sub(r"_\d+lang(?:_(?:IncTar|ExcTar))?$", "", method)
    method_base = re.sub(r"_.*?$", "", method_base)

    matched_family = None
    for exp_name, exp_data in experiment_results.items():
        if exp_data["type"] == "baseline":
            continue

        exp_type = exp_data["type"]
        if exp_type.startswith("merge_"):
            exp_method = exp_type.replace("merge_", "")

            if exp_method == method_base:
                if (
                    exp_data.get("model_family_name")
                    and exp_data["model_family_name"] != "unknown"
                ):
                    return exp_data["model_family_name"]
                if exp_data.get("model_family") and exp_data["model_family"] != "unknown":
                    matched_family = exp_data["model_family"]

    if matched_family:
        return matched_family

    if "_roberta-base" in method:
        return "roberta-base"
    elif "_roberta-large" in method:
        return "roberta-large"
    elif "_bert-base" in method:
        return "bert-base"
    elif "_bert-large" in method:
        return "bert-large"
    elif "_modernbert" in method:
        return "modernbert"

    return "unknown"


def get_method_similarity_type(method: str, experiment_results: Dict[str, Dict]) -> str:
    """Get the similarity type for a method."""
    for exp_name, exp_data in experiment_results.items():
        if exp_data["type"] == "baseline":
            continue

        exp_type = exp_data["type"]
        if exp_type.startswith("merge_"):
            exp_method = exp_type.replace("merge_", "")

            method_base = re.sub(r"_\d+lang$", "", method)
            method_base = re.sub(r"_.*?$", "", method_base)
            if exp_method == method_base:
                return exp_data.get("similarity_type", "unknown")

    if "_URIEL_" in method:
        return "URIEL"
    elif "_REAL_" in method:
        return "REAL"

    return "URIEL"


def calculate_improvements(
    scores: List[float], baseline_scores: List[float]
) -> List[float]:
    """Calculate improvements of scores over baseline."""
    return [score - baseline for score, baseline in zip(scores, baseline_scores)]


def compute_statistics(values: List[float], threshold: float = 0.001) -> Dict[str, Any]:
    """Compute summary statistics for a list of values."""
    valid_values = [v for v in values if abs(v) > threshold]
    if not valid_values:
        return {"mean": 0.0, "positive_count": 0, "total_count": 0, "win_rate": 0.0}

    positive_count = sum(1 for v in valid_values if v > 0)
    total_count = len(valid_values)
    win_rate = (positive_count / total_count) * 100 if total_count > 0 else 0.0

    return {
        "mean": np.mean(valid_values),
        "positive_count": positive_count,
        "total_count": total_count,
        "win_rate": win_rate,
    }


def get_method_columns(df: pd.DataFrame) -> List[str]:
    """Get all method columns from a dataframe (excluding metadata columns)."""
    excluded = {
        "locale",
        "baseline",
        "avg_zero_shot",
        "best_zero_shot",
        "best_source",
        "source_locales",
        "num_languages_map",
        "zero_shot_family",
    }
    return [
        col
        for col in df.columns
        if col not in excluded
        and not col.startswith("baseline_")
        and "_vs_" not in col
    ]


def get_vs_columns(df: pd.DataFrame, suffix: str) -> List[str]:
    """Get all columns ending with a specific suffix (e.g., '_vs_avg_zero')."""
    return [col for col in df.columns if col.endswith(suffix)]
