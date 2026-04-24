#!/usr/bin/env python3
"""
Cross-locale analysis of layer ablation results.

Reads all per-locale SQLite DBs and builds a unified view of which layers
cause interference vs positive transfer across different target languages.

Usage:
    python analyze_layer_ablation_cross_locale.py
    python analyze_layer_ablation_cross_locale.py --output-dir results/layer_ablation/cross_locale
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from merginguriel.selective_layer import (
    LayerAblationConfig,
    LayerAblationDB,
    analyze_ablation_results,
)


LANGUAGE_FAMILIES = {
    "sw-KE": "Bantu",
    "cy-GB": "Celtic",
    "sq-AL": "Albanian",
    "vi-VN": "Austroasiatic",
    "ja-JP": "Japonic",
    "hi-IN": "Indo-Aryan",
    "fi-FI": "Uralic",
    "ar-SA": "Semitic",
    "ko-KR": "Koreanic",
    "th-TH": "Tai",
}


def load_all_results() -> dict[str, pd.DataFrame]:
    """Load results from all locale DBs."""
    config_pattern = str(project_root / "configs" / "ablations" / "layer_ablation_*.yaml")
    config_files = sorted(glob.glob(config_pattern))

    all_results = {}
    for config_path in config_files:
        config = LayerAblationConfig.from_yaml(Path(config_path))
        db_path = project_root / config.db_path

        if not db_path.exists():
            print(f"  SKIP {config.target_locale}: DB not found ({db_path})")
            continue

        db = LayerAblationDB(str(db_path))
        results_df = db.get_results_df(config.name)

        if results_df.empty:
            print(f"  SKIP {config.target_locale}: No completed experiments")
            continue

        summary = analyze_ablation_results(results_df)
        if not summary.empty:
            all_results[config.target_locale] = {
                "summary": summary,
                "full": results_df,
            }
            n_completed = len(results_df)
            print(f"  OK   {config.target_locale}: {n_completed} completed experiments")

    return all_results


def build_delta_matrix(all_results: dict) -> pd.DataFrame:
    """Build ablation_point x locale delta matrix."""
    rows = []
    for locale, data in sorted(all_results.items()):
        summary = data["summary"]
        for _, row in summary.iterrows():
            rows.append({
                "locale": locale,
                "family": LANGUAGE_FAMILIES.get(locale, "Unknown"),
                "ablation_point": row["ablation_point"],
                "delta_mean": row["delta_mean"],
                "delta_std": row["delta_std"],
                "transfer_type": row["transfer_type"],
            })

    df = pd.DataFrame(rows)
    return df


def build_pivot_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Build pivot: ablation_point (rows) x locale (columns) with delta values."""
    pivot = delta_df.pivot_table(
        index="ablation_point",
        columns="locale",
        values="delta_mean",
        aggfunc="first",
    )

    # Sort: individual layers first (by number), then groups
    layer_order = ["exclude_layer_" + str(i) for i in range(12)]
    group_order = ["exclude_group_bottom", "exclude_group_middle", "exclude_group_top"]
    ordered = [p for p in layer_order + group_order if p in pivot.index]
    remaining = [p for p in pivot.index if p not in ordered]
    pivot = pivot.reindex(ordered + remaining)

    return pivot


def identify_consistent_layers(delta_df: pd.DataFrame) -> dict:
    """Identify layers that are consistently interference or positive transfer."""
    results = {"interference": [], "positive_transfer": [], "mixed": []}

    for point in delta_df["ablation_point"].unique():
        point_data = delta_df[delta_df["ablation_point"] == point]
        types = point_data["transfer_type"].value_counts()

        n_locales = len(point_data)
        n_interference = types.get("interference", 0)
        n_positive = types.get("positive_transfer", 0)
        mean_delta = point_data["delta_mean"].mean()

        entry = {
            "ablation_point": point,
            "mean_delta_across_locales": mean_delta,
            "n_interference": n_interference,
            "n_positive": n_positive,
            "n_neutral": types.get("neutral", 0),
            "n_locales": n_locales,
            "consistency": max(n_interference, n_positive) / n_locales,
        }

        # >60% agreement across locales
        if n_interference > 0.6 * n_locales:
            results["interference"].append(entry)
        elif n_positive > 0.6 * n_locales:
            results["positive_transfer"].append(entry)
        else:
            results["mixed"].append(entry)

    return results


def print_cross_locale_summary(
    pivot: pd.DataFrame,
    consistent: dict,
    delta_df: pd.DataFrame,
):
    """Print comprehensive cross-locale summary."""
    print("\n" + "=" * 80)
    print("CROSS-LOCALE LAYER ABLATION ANALYSIS")
    print("=" * 80)

    # Delta matrix
    print("\nDELTA MATRIX (positive = excluding helped = interference)")
    print("-" * 80)
    print(pivot.round(4).to_string())

    # Consistent interference layers
    print("\n\nCONSISTENT INTERFERENCE LAYERS (exclude from merging to improve):")
    print("-" * 60)
    if consistent["interference"]:
        for entry in sorted(consistent["interference"], key=lambda x: -x["mean_delta_across_locales"]):
            print(f"  {entry['ablation_point']:25s} | avg delta: {entry['mean_delta_across_locales']:+.4f} | "
                  f"interference in {entry['n_interference']}/{entry['n_locales']} locales ({entry['consistency']:.0%})")
    else:
        print("  None found")

    # Consistent positive transfer layers
    print("\nCONSISTENT POSITIVE TRANSFER LAYERS (keep merged):")
    print("-" * 60)
    if consistent["positive_transfer"]:
        for entry in sorted(consistent["positive_transfer"], key=lambda x: x["mean_delta_across_locales"]):
            print(f"  {entry['ablation_point']:25s} | avg delta: {entry['mean_delta_across_locales']:+.4f} | "
                  f"positive in {entry['n_positive']}/{entry['n_locales']} locales ({entry['consistency']:.0%})")
    else:
        print("  None found")

    # Mixed layers
    print("\nMIXED/LOCALE-DEPENDENT LAYERS:")
    print("-" * 60)
    if consistent["mixed"]:
        for entry in sorted(consistent["mixed"], key=lambda x: x["mean_delta_across_locales"]):
            print(f"  {entry['ablation_point']:25s} | avg delta: {entry['mean_delta_across_locales']:+.4f} | "
                  f"+transfer: {entry['n_positive']}, interference: {entry['n_interference']}, neutral: {entry['n_neutral']}")
    else:
        print("  None found")

    # Per-family analysis
    print("\n\nPER-FAMILY ANALYSIS:")
    print("-" * 60)
    families = delta_df.groupby("family")
    for family, group in sorted(families):
        locales = group["locale"].unique()
        avg_delta = group.groupby("ablation_point")["delta_mean"].mean()
        worst_layer = avg_delta.idxmax()
        best_layer = avg_delta.idxmin()
        print(f"  {family:15s} ({', '.join(locales)}):")
        print(f"    Most interference: {worst_layer} (delta={avg_delta[worst_layer]:+.4f})")
        print(f"    Most positive:     {best_layer} (delta={avg_delta[best_layer]:+.4f})")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Cross-locale layer ablation analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "layer_ablation" / "cross_locale",
    )
    args = parser.parse_args()

    print("Loading results from all locales...")
    all_results = load_all_results()

    if not all_results:
        print("\nNo completed results found. Run experiments first.")
        sys.exit(1)

    print(f"\nLoaded results for {len(all_results)} locales: {sorted(all_results.keys())}")

    # Build unified analysis
    delta_df = build_delta_matrix(all_results)
    pivot = build_pivot_table(delta_df)
    consistent = identify_consistent_layers(delta_df)

    # Print summary
    print_cross_locale_summary(pivot, consistent, delta_df)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pivot.to_csv(args.output_dir / "delta_matrix.csv")
    delta_df.to_csv(args.output_dir / "delta_all_locales.csv", index=False)

    # Save consistent layers as JSON
    import json
    with open(args.output_dir / "consistent_layers.json", "w") as f:
        json.dump(consistent, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
