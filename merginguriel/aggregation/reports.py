"""
Analysis and reporting functions for experiment results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from merginguriel import logger


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dynamic comparison table with all experiment types and baseline data."""
    column_field = "experiment_variant" if "experiment_variant" in df.columns else "experiment_type"

    experiment_types = sorted(df[column_field].dropna().unique())

    pivot_df = df.pivot_table(
        index="locale",
        columns=column_field,
        values="accuracy",
        aggfunc="first",
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename_axis(None, axis=1)

    baseline_columns = ["baseline"]
    discovered_columns = [col for col in experiment_types if col not in baseline_columns]

    baseline_variant_cols = [
        col for col in pivot_df.columns if isinstance(col, str) and col.startswith("baseline_")
    ]
    if "baseline" not in pivot_df.columns:
        pivot_df["baseline"] = None
    if baseline_variant_cols:
        pivot_df["baseline"] = pivot_df.apply(
            lambda row: max(
                [row[col] for col in baseline_variant_cols if pd.notna(row[col])],
                default=None,
            ),
            axis=1,
        )

    final_columns = ["locale"] + baseline_columns + discovered_columns

    baseline_data_columns = ["best_source_accuracy", "best_overall_accuracy"]
    for col in baseline_data_columns:
        if col in df.columns:
            final_columns.append(col)

    for col in final_columns:
        if col not in pivot_df.columns and col != "locale":
            pivot_df[col] = None

    pivot_df = pivot_df[final_columns]

    if "baseline" in pivot_df.columns:
        pivot_df = _add_improvement_columns(pivot_df, discovered_columns)

    return pivot_df


def _add_improvement_columns(pivot_df: pd.DataFrame, discovered_columns: List[str]) -> pd.DataFrame:
    """Add improvement columns comparing experiment results against baselines."""
    baseline_variant_cols = [
        col for col in pivot_df.columns if isinstance(col, str) and col.startswith("baseline_")
    ]

    for exp_type in discovered_columns:
        if exp_type.startswith("baseline_") or exp_type not in pivot_df.columns:
            continue

        matching_baseline = next(
            (col for col in baseline_variant_cols if col.replace("baseline_", "") in exp_type),
            None,
        )
        if matching_baseline is None:
            matching_baseline = "baseline"

        improvement_col = f"{exp_type}_improvement"
        pivot_df[improvement_col] = pivot_df.apply(
            lambda row, exp=exp_type, base=matching_baseline: _calculate_improvement(row, exp, base),
            axis=1,
        )

    return pivot_df


def _calculate_improvement(
    row: pd.Series, exp_type: str, baseline_col: str
) -> Optional[float]:
    """Calculate improvement between experiment result and baseline."""
    if baseline_col not in row:
        return None
    baseline_val = row[baseline_col]
    exp_val = row[exp_type]
    if baseline_val is None or exp_val is None:
        return None
    if pd.isna(baseline_val) or pd.isna(exp_val):
        return None
    return exp_val - baseline_val


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Generate dynamic summary statistics for all experiment types."""
    summary = {}

    column_field = "experiment_variant" if "experiment_variant" in df.columns else "experiment_type"
    experiment_types = sorted(df[column_field].dropna().unique())

    for exp_type in experiment_types:
        type_df = df[df[column_field] == exp_type]
        valid_acc = type_df[type_df["accuracy"].notna()]

        if len(valid_acc) > 0:
            summary[exp_type] = {
                "count": len(valid_acc),
                "mean_accuracy": valid_acc["accuracy"].mean(),
                "std_accuracy": valid_acc["accuracy"].std(),
                "min_accuracy": valid_acc["accuracy"].min(),
                "max_accuracy": valid_acc["accuracy"].max(),
            }

    baseline_types = ["best_source_accuracy", "best_overall_accuracy"]
    for baseline_type in baseline_types:
        if baseline_type in df.columns:
            valid_baseline = df[df[baseline_type].notna()]
            if len(valid_baseline) > 0:
                summary[f"baseline_{baseline_type}"] = {
                    "count": len(valid_baseline),
                    "mean_accuracy": valid_baseline[baseline_type].mean(),
                    "std_accuracy": valid_baseline[baseline_type].std(),
                    "min_accuracy": valid_baseline[baseline_type].min(),
                    "max_accuracy": valid_baseline[baseline_type].max(),
                }

    return summary


def generate_win_rate_analysis(
    df: pd.DataFrame, comparison_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Generate win rate analysis comparing merging methods against baselines."""
    win_analysis = {}

    experiment_types = [
        col
        for col in comparison_df.columns
        if col not in ["locale", "baseline", "best_source_accuracy", "best_overall_accuracy"]
    ]

    for exp_type in experiment_types:
        if exp_type not in comparison_df.columns:
            continue

        win_stats = _compute_win_stats(comparison_df, exp_type)
        win_analysis[exp_type] = win_stats

    return win_analysis


def _compute_win_stats(comparison_df: pd.DataFrame, exp_type: str) -> Dict[str, Any]:
    """Compute win statistics for a single experiment type."""
    win_stats = {
        "vs_baseline_wins": 0,
        "vs_baseline_total": 0,
        "vs_best_source_wins": 0,
        "vs_best_source_total": 0,
        "vs_best_overall_wins": 0,
        "vs_best_overall_total": 0,
        "avg_improvement_vs_baseline": 0.0,
        "avg_improvement_vs_best_source": 0.0,
        "avg_improvement_vs_best_overall": 0.0,
    }

    improvements_vs_baseline = []
    improvements_vs_best_source = []
    improvements_vs_best_overall = []

    for _, row in comparison_df.iterrows():
        if row[exp_type] is None:
            continue

        if row.get("baseline") is not None:
            win_stats["vs_baseline_total"] += 1
            improvement = row[exp_type] - row["baseline"]
            improvements_vs_baseline.append(improvement)
            if improvement > 0:
                win_stats["vs_baseline_wins"] += 1

        if row.get("best_source_accuracy") is not None:
            win_stats["vs_best_source_total"] += 1
            improvement = row[exp_type] - row["best_source_accuracy"]
            improvements_vs_best_source.append(improvement)
            if improvement > 0:
                win_stats["vs_best_source_wins"] += 1

        if row.get("best_overall_accuracy") is not None:
            win_stats["vs_best_overall_total"] += 1
            improvement = row[exp_type] - row["best_overall_accuracy"]
            improvements_vs_best_overall.append(improvement)
            if improvement > 0:
                win_stats["vs_best_overall_wins"] += 1

    if improvements_vs_baseline:
        win_stats["avg_improvement_vs_baseline"] = sum(improvements_vs_baseline) / len(
            improvements_vs_baseline
        )
    if improvements_vs_best_source:
        win_stats["avg_improvement_vs_best_source"] = sum(improvements_vs_best_source) / len(
            improvements_vs_best_source
        )
    if improvements_vs_best_overall:
        win_stats["avg_improvement_vs_best_overall"] = sum(improvements_vs_best_overall) / len(
            improvements_vs_best_overall
        )

    return win_stats


def save_results(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary: Dict[str, Any],
    win_analysis: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Save all results to files with comprehensive exports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"results_aggregated_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw results saved to {raw_path}")

    comparison_path = output_dir / f"results_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Comparison table saved to {comparison_path}")

    comprehensive_path = output_dir / f"results_comprehensive_{timestamp}.csv"
    comparison_df.to_csv(comprehensive_path, index=False)
    logger.info(f"Comprehensive results saved to {comprehensive_path}")

    summary_path = output_dir / f"results_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary statistics saved to {summary_path}")

    win_analysis_path = None
    if win_analysis:
        win_analysis_path = output_dir / f"results_win_analysis_{timestamp}.json"
        with open(win_analysis_path, "w") as f:
            json.dump(win_analysis, f, indent=2)
        logger.info(f"Win rate analysis saved to {win_analysis_path}")

    markdown_content = create_comprehensive_markdown_report(comparison_df, summary, win_analysis)
    markdown_path = output_dir / f"results_report_{timestamp}.md"
    with open(markdown_path, "w") as f:
        f.write(markdown_content)
    logger.info(f"Comprehensive report saved to {markdown_path}")

    return {
        "raw_filename": str(raw_path),
        "comparison_filename": str(comparison_path),
        "comprehensive_filename": str(comprehensive_path),
        "summary_filename": str(summary_path),
        "markdown_filename": str(markdown_path),
        "win_analysis_filename": str(win_analysis_path) if win_analysis_path else None,
    }


def create_comprehensive_markdown_report(
    comparison_df: pd.DataFrame,
    summary: Dict[str, Any],
    win_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a comprehensive markdown report with baseline comparisons and analysis."""
    report = "# Comprehensive Experiment Results Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += _build_executive_summary(comparison_df, summary)
    report += _build_summary_statistics_section(summary)
    report += _build_baseline_statistics_section(summary)

    if win_analysis:
        report += _build_win_rate_section(win_analysis)

    report += _build_detailed_comparison_section(comparison_df)
    report += _build_methodology_section()

    return report


def _build_executive_summary(comparison_df: pd.DataFrame, summary: Dict[str, Any]) -> str:
    """Build the executive summary section of the report."""
    section = "## Executive Summary\n\n"
    section += "This report presents the results of large-scale language model merging experiments.\n"
    section += f"The analysis includes **{len(comparison_df)}** locales and **{len(summary)}** experiment types.\n\n"
    return section


def _build_summary_statistics_section(summary: Dict[str, Any]) -> str:
    """Build the summary statistics section of the report."""
    section = "## Summary Statistics\n\n"
    for exp_type, stats in summary.items():
        if exp_type.startswith("baseline_"):
            continue

        section += f"### {exp_type.upper()}\n"
        section += f"- Number of experiments: {stats['count']}\n"
        section += f"- Mean accuracy: {stats['mean_accuracy']:.4f}\n"
        section += f"- Standard deviation: {stats['std_accuracy']:.4f}\n"
        section += f"- Min accuracy: {stats['min_accuracy']:.4f}\n"
        section += f"- Max accuracy: {stats['max_accuracy']:.4f}\n\n"

    return section


def _build_baseline_statistics_section(summary: Dict[str, Any]) -> str:
    """Build the baseline statistics section of the report."""
    baseline_summary_exists = any(key.startswith("baseline_") for key in summary.keys())
    if not baseline_summary_exists:
        return ""

    section = "### BASELINE PERFORMANCE\n\n"
    for key in ["baseline_best_source_accuracy", "baseline_best_overall_accuracy"]:
        if key not in summary:
            continue
        stats = summary[key]
        baseline_name = "Best Source Language" if "best_source" in key else "Best Overall Zero-Shot"
        section += f"**{baseline_name}**\n"
        section += f"- Number of comparisons: {stats['count']}\n"
        section += f"- Mean accuracy: {stats['mean_accuracy']:.4f}\n"
        section += f"- Standard deviation: {stats['std_accuracy']:.4f}\n"
        section += f"- Min accuracy: {stats['min_accuracy']:.4f}\n"
        section += f"- Max accuracy: {stats['max_accuracy']:.4f}\n\n"

    return section


def _build_win_rate_section(win_analysis: Dict[str, Any]) -> str:
    """Build the win rate analysis section of the report."""
    section = "## Win Rate Analysis\n\n"
    section += "This section shows how often each merging method outperforms the baselines.\n\n"

    for exp_type, stats in win_analysis.items():
        section += f"### {exp_type.upper()}\n"

        if stats["vs_baseline_total"] > 0:
            win_rate = (stats["vs_baseline_wins"] / stats["vs_baseline_total"]) * 100
            section += f"- **vs Native Baseline**: {stats['vs_baseline_wins']}/{stats['vs_baseline_total']} wins ({win_rate:.1f}%)\n"
            section += f"  - Average improvement: {stats['avg_improvement_vs_baseline']:+.4f}\n"

        if stats["vs_best_source_total"] > 0:
            win_rate = (stats["vs_best_source_wins"] / stats["vs_best_source_total"]) * 100
            section += f"- **vs Best Source Language**: {stats['vs_best_source_wins']}/{stats['vs_best_source_total']} wins ({win_rate:.1f}%)\n"
            section += f"  - Average improvement: {stats['avg_improvement_vs_best_source']:+.4f}\n"

        if stats["vs_best_overall_total"] > 0:
            win_rate = (stats["vs_best_overall_wins"] / stats["vs_best_overall_total"]) * 100
            section += f"- **vs Best Overall Zero-Shot**: {stats['vs_best_overall_wins']}/{stats['vs_best_overall_total']} wins ({win_rate:.1f}%)\n"
            section += f"  - Average improvement: {stats['avg_improvement_vs_best_overall']:+.4f}\n"

        section += "\n"

    return section


def _build_detailed_comparison_section(comparison_df: pd.DataFrame) -> str:
    """Build the detailed comparison table section of the report."""
    section = "## Detailed Comparison Results\n\n"

    headers = ["Locale"]
    if "baseline" in comparison_df.columns:
        headers.append("Baseline")

    experiment_columns = [
        col
        for col in comparison_df.columns
        if col not in ["locale", "baseline", "best_source_accuracy", "best_overall_accuracy"]
        and not col.endswith("_improvement")
        and not col.endswith("_vs_")
    ]
    headers.extend(experiment_columns)

    if "best_source_accuracy" in comparison_df.columns:
        headers.append("Best Source")
    if "best_overall_accuracy" in comparison_df.columns:
        headers.append("Best Overall")

    for exp_col in experiment_columns:
        if f"{exp_col}_improvement" in comparison_df.columns:
            headers.append(f"{exp_col} vs Base")
        if f"{exp_col}_vs_best_source" in comparison_df.columns:
            headers.append(f"{exp_col} vs Src")

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join(["-" * len(h) for h in headers]) + "|"
    section += header_row + "\n"
    section += separator_row + "\n"

    for _, row in comparison_df.iterrows():
        row_values = _format_comparison_row(row, comparison_df, experiment_columns)
        section += "| " + " | ".join(row_values) + " |\n"

    return section


def _format_comparison_row(
    row: pd.Series, comparison_df: pd.DataFrame, experiment_columns: List[str]
) -> List[str]:
    """Format a single row of the comparison table."""
    row_values = [row["locale"]]

    if "baseline" in comparison_df.columns:
        baseline_val = f"{row['baseline']:.4f}" if row["baseline"] is not None else "N/A"
        row_values.append(baseline_val)

    for exp_col in experiment_columns:
        exp_val = f"{row[exp_col]:.4f}" if row[exp_col] is not None else "N/A"
        row_values.append(exp_val)

    if "best_source_accuracy" in comparison_df.columns:
        source_val = (
            f"{row['best_source_accuracy']:.4f}"
            if row["best_source_accuracy"] is not None
            else "N/A"
        )
        row_values.append(source_val)

    if "best_overall_accuracy" in comparison_df.columns:
        overall_val = (
            f"{row['best_overall_accuracy']:.4f}"
            if row["best_overall_accuracy"] is not None
            else "N/A"
        )
        row_values.append(overall_val)

    for exp_col in experiment_columns:
        if f"{exp_col}_improvement" in comparison_df.columns:
            imp_val = row[f"{exp_col}_improvement"]
            imp_str = f"{imp_val:+.4f}" if imp_val is not None else "N/A"
            row_values.append(imp_str)

        if f"{exp_col}_vs_best_source" in comparison_df.columns:
            imp_val = row[f"{exp_col}_vs_best_source"]
            imp_str = f"{imp_val:+.4f}" if imp_val is not None else "N/A"
            row_values.append(imp_str)

    return row_values


def _build_methodology_section() -> str:
    """Build the methodology section of the report."""
    section = "\n## Methodology\n\n"
    section += "### Experimental Setup\n"
    section += "- Each target language was evaluated using multiple merging approaches\n"
    section += "- Results are compared against native performance and cross-lingual baselines\n"
    section += "- Best Source Language: Best performing single source model for the target\n"
    section += "- Best Overall Zero-Shot: Best performing any language model (excluding target)\n\n"

    section += "### Performance Metrics\n"
    section += "- **Accuracy**: Primary evaluation metric on MASSIVE intent classification\n"
    section += "- **Improvement**: Difference between merged model and baseline performance\n"
    section += "- **Win Rate**: Percentage of cases where merging method outperforms baseline\n\n"

    return section
