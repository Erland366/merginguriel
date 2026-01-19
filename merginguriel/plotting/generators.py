"""
Plot generation functions for MergingUriel results analysis.

This module contains all plot generation methods extracted from
the AdvancedResultsAnalyzer class for better modularity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from merginguriel.plotting.utils import (
    maybe_float,
    format_method_key_for_display,
    format_method_key_for_filename,
    compute_statistics,
    get_method_columns,
    get_vs_columns,
)


class PlotGenerator:
    """Handles generation of all plot types for results analysis."""

    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(exist_ok=True)

    def generate_pure_scores_plots(
        self,
        summary_df: pd.DataFrame,
        timestamp: str,
        get_method_model_family_fn,
        get_method_similarity_type_fn,
        get_method_num_language_set_fn,
    ) -> None:
        """Generate individual plots showing pure performance scores for each method."""
        print("Generating pure scores plots...")

        method_cols = get_method_columns(summary_df)

        if not method_cols:
            print("No method columns found for pure scores plots")
            return

        locales = summary_df["locale"].tolist()

        for method in method_cols:
            model_family = get_method_model_family_fn(method)
            similarity_type = get_method_similarity_type_fn(method)

            fig, ax = plt.subplots(figsize=(20, 8))
            display_name = format_method_key_for_display(method, model_family, similarity_type)
            file_method = format_method_key_for_filename(method, model_family, similarity_type)

            method_data = summary_df[method].fillna(0).tolist()
            baseline_data = (
                summary_df.get("baseline", pd.Series([0] * len(locales))).fillna(0).tolist()
            )
            avg_zero_data = (
                summary_df.get("avg_zero_shot", pd.Series([0] * len(locales))).fillna(0).tolist()
            )
            best_zero_data = (
                summary_df.get("best_zero_shot", pd.Series([0] * len(locales))).fillna(0).tolist()
            )
            best_source_data = (
                summary_df.get("best_source", pd.Series([0] * len(locales))).fillna(0).tolist()
            )

            x = np.arange(len(locales))
            width = 0.16

            ax.bar(x - 2 * width, baseline_data, width, label="Baseline", alpha=0.7, color="gray")
            ax.bar(
                x - width, avg_zero_data, width, label="Avg Zero-shot", alpha=0.7, color="lightblue"
            )
            ax.bar(x, best_zero_data, width, label="Best Zero-shot", alpha=0.7, color="lightgreen")
            ax.bar(
                x + width, best_source_data, width, label="Best Source", alpha=0.7, color="lightcoral"
            )
            bars5 = ax.bar(
                x + 2 * width, method_data, width, label=display_name, alpha=0.8, color="royalblue"
            )

            for bar in bars5:
                height = bar.get_height()
                if height > 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            ax.set_xlabel("Target Languages")
            ax.set_ylabel("Performance Score")
            ax.set_title(f"Pure Performance: {display_name} vs All Baselines")
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = self.plots_dir / f"pure_scores_{file_method}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Pure scores plot for {file_method} saved to: {output_file}")

    def generate_vs_avg_zero_plots(
        self,
        summary_df: pd.DataFrame,
        timestamp: str,
        get_method_model_family_fn,
        get_method_similarity_type_fn,
    ) -> None:
        """Generate individual plots showing improvement vs average zero-shot baseline."""
        print("Generating vs average zero-shot plots...")

        vs_avg_cols = get_vs_columns(summary_df, "_vs_avg_zero")

        if not vs_avg_cols:
            print("No vs_avg_zero columns found for comparison plots")
            return

        locales = summary_df["locale"].tolist()

        for col in vs_avg_cols:
            method_name = col.replace("_vs_avg_zero", "")
            model_family = get_method_model_family_fn(method_name)
            similarity_type = get_method_similarity_type_fn(method_name)

            improvement_data = summary_df[col].fillna(0).tolist()
            display_name = format_method_key_for_display(method_name, model_family, similarity_type)
            file_method = format_method_key_for_filename(method_name, model_family, similarity_type)

            self._create_improvement_bar_plot(
                locales=locales,
                improvements=improvement_data,
                display_name=display_name,
                baseline_name="Average Zero-shot",
                file_method=file_method,
                prefix="vs_avg_zero",
                timestamp=timestamp,
                pos_color="green",
                neg_color="lightcoral",
                stats_bg_color="wheat",
            )

    def generate_vs_best_zero_plots(
        self,
        summary_df: pd.DataFrame,
        timestamp: str,
        get_method_model_family_fn,
        get_method_similarity_type_fn,
    ) -> None:
        """Generate individual plots showing improvement vs best zero-shot baseline."""
        print("Generating vs best zero-shot plots...")

        vs_best_cols = get_vs_columns(summary_df, "_vs_best_zero")

        if not vs_best_cols:
            print("No vs_best_zero columns found for comparison plots")
            return

        locales = summary_df["locale"].tolist()

        for col in vs_best_cols:
            method_name = col.replace("_vs_best_zero", "")
            model_family = get_method_model_family_fn(method_name)
            similarity_type = get_method_similarity_type_fn(method_name)

            improvement_data = summary_df[col].fillna(0).tolist()
            display_name = format_method_key_for_display(method_name, model_family, similarity_type)
            file_method = format_method_key_for_filename(method_name, model_family, similarity_type)

            self._create_improvement_bar_plot(
                locales=locales,
                improvements=improvement_data,
                display_name=display_name,
                baseline_name="Best Zero-shot",
                file_method=file_method,
                prefix="vs_best_zero",
                timestamp=timestamp,
                pos_color="royalblue",
                neg_color="orange",
                stats_bg_color="lightblue",
            )

    def generate_vs_best_source_plots(
        self,
        summary_df: pd.DataFrame,
        timestamp: str,
        get_method_model_family_fn,
        get_method_similarity_type_fn,
    ) -> None:
        """Generate individual plots showing improvement vs best source baseline."""
        print("Generating vs best source plots...")

        vs_best_source_cols = get_vs_columns(summary_df, "_vs_best_source")

        if not vs_best_source_cols:
            print("No vs_best_source columns found for comparison plots")
            return

        locales = summary_df["locale"].tolist()

        for col in vs_best_source_cols:
            method_name = col.replace("_vs_best_source", "")
            model_family = get_method_model_family_fn(method_name)
            similarity_type = get_method_similarity_type_fn(method_name)

            improvement_data = summary_df[col].fillna(0).tolist()
            display_name = format_method_key_for_display(method_name, model_family, similarity_type)
            file_method = format_method_key_for_filename(method_name, model_family, similarity_type)

            self._create_improvement_bar_plot(
                locales=locales,
                improvements=improvement_data,
                display_name=display_name,
                baseline_name="Best Source",
                file_method=file_method,
                prefix="vs_best_source",
                timestamp=timestamp,
                pos_color="forestgreen",
                neg_color="indianred",
                stats_bg_color="lightgreen",
                title_suffix=" (Fair Comparison)",
            )

    def generate_baseline_comparison_plots(
        self,
        summary_df: pd.DataFrame,
        timestamp: str,
        get_method_model_family_fn,
        get_method_similarity_type_fn,
        infer_model_family_from_method_key_fn,
    ) -> None:
        """Generate plots comparing baseline performance directly against each method."""
        print("Generating baseline comparison plots...")

        if "locale" not in summary_df.columns or summary_df.empty:
            print("No locale data found for baseline comparison plots")
            return

        locales = summary_df["locale"].tolist()
        baseline_cols = [
            col
            for col in summary_df.columns
            if col.startswith("baseline_") and "_vs_" not in col
        ]

        if not baseline_cols:
            print("No baseline columns found for comparison plots")
            return

        method_cols = get_method_columns(summary_df)

        if not method_cols:
            print("No method columns found for baseline comparison plots")
            return

        for method in method_cols:
            model_family = get_method_model_family_fn(method)
            inferred_family = infer_model_family_from_method_key_fn(method)
            if inferred_family:
                model_family = inferred_family

            baseline_col = (
                f"baseline_{model_family}"
                if model_family and model_family != "unknown"
                else None
            )

            if baseline_col not in summary_df.columns:
                baseline_col = "baseline" if "baseline" in summary_df.columns else None

            if not baseline_col:
                continue

            baseline_values = (
                pd.to_numeric(summary_df[baseline_col], errors="coerce").fillna(0).tolist()
            )
            method_values = (
                pd.to_numeric(summary_df[method], errors="coerce").fillna(0).tolist()
            )
            display_name = format_method_key_for_display(
                method, model_family, get_method_similarity_type_fn(method)
            )
            file_method = format_method_key_for_filename(
                method, model_family, get_method_similarity_type_fn(method)
            )

            differences = [m - b for m, b in zip(method_values, baseline_values)]

            fig, ax = plt.subplots(figsize=(20, 8))
            x = np.arange(len(locales))
            width = 0.35

            if baseline_col == "baseline" or not model_family or model_family == "unknown":
                baseline_label = "Baseline"
            else:
                baseline_label = f"Baseline ({model_family})"

            ax.bar(
                x - width / 2, baseline_values, width, label=baseline_label, alpha=0.7, color="gray"
            )
            method_bars = ax.bar(
                x + width / 2, method_values, width, label=display_name, alpha=0.85, color="royalblue"
            )

            for bar, diff in zip(method_bars, differences):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{diff:+.3f}",
                    ha="center",
                    va="bottom" if diff >= 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                    color="darkgreen" if diff >= 0 else "darkred",
                )

            diff_values = [d for d in differences if abs(d) > 1e-6]
            if diff_values:
                mean_diff = np.mean(diff_values)
                positive = sum(1 for d in diff_values if d > 0)
                stats_text = f"Mean Delta: {mean_diff:+.4f}\nImproved: {positive}/{len(diff_values)}"
                ax.text(
                    0.02,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                )

            ax.set_xlabel("Target Languages")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Baseline vs {display_name}")
            ax.set_xticks(x)
            ax.set_xticklabels(locales, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = self.plots_dir / f"baseline_vs_{file_method}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Baseline comparison plot for {file_method} saved to: {output_file}")

    def _create_improvement_bar_plot(
        self,
        locales: List[str],
        improvements: List[float],
        display_name: str,
        baseline_name: str,
        file_method: str,
        prefix: str,
        timestamp: str,
        pos_color: str,
        neg_color: str,
        stats_bg_color: str,
        title_suffix: str = "",
    ) -> None:
        """Create a bar plot showing improvements over a baseline."""
        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(locales))
        width = 0.6

        bars = ax.bar(x, improvements, width, alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.001:
                if height >= 0:
                    bar.set_color(pos_color)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"+{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    bar.set_color(neg_color)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=2)

        stats = compute_statistics(improvements)
        if stats["total_count"] > 0:
            stats_text = (
                f"Mean: {stats['mean']:+.4f}\n"
                f"Win Rate: {stats['win_rate']:.1f}%\n"
                f"Count: {stats['total_count']}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=stats_bg_color, alpha=0.8),
            )

        ax.set_xlabel("Target Languages")
        ax.set_ylabel(f"Improvement over {baseline_name}")
        ax.set_title(f"{display_name} vs {baseline_name} Baseline{title_suffix}")
        ax.set_xticks(x)
        ax.set_xticklabels(locales, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.plots_dir / f"{prefix}_{file_method}_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {prefix} plot for {file_method} saved to: {output_file}")

    def generate_advanced_performance_plot(
        self, merging_results: List[Dict], timestamp: str
    ) -> None:
        """Create comprehensive performance comparison for advanced methods."""
        print("Creating advanced performance comparison...")

        advanced_methods = ["ties", "task_arithmetic", "slerp", "regmean", "dare", "fisher"]
        method_data = {method: [] for method in advanced_methods}
        method_data["similarity"] = []
        method_data["average"] = []
        method_data["baseline"] = []
        method_data["avg_zero_shot"] = []
        method_data["best_zero_shot"] = []
        locales = []

        for result in merging_results:
            locales.append(result["target_locale"])
            method_data["baseline"].append(maybe_float(result.get("baseline")) or 0)
            method_data["similarity"].append(maybe_float(result.get("similarity")) or 0)
            method_data["average"].append(maybe_float(result.get("average")) or 0)
            method_data["avg_zero_shot"].append(maybe_float(result.get("avg_zero_shot")) or 0)
            method_data["best_zero_shot"].append(maybe_float(result.get("best_zero_shot")) or 0)

            for method in advanced_methods:
                method_data[method].append(maybe_float(result.get(method)) or 0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

        x = np.arange(len(locales))
        width = 0.08
        bar_positions = np.arange(-3, 4) * width
        methods = ["baseline", "avg_zero_shot", "similarity", "average"] + advanced_methods[:3]
        colors = ["gray", "lightgray", "blue", "red", "green", "orange", "purple"]

        for i, (method, color) in enumerate(zip(methods, colors)):
            if method in method_data and method_data[method]:
                ax1.bar(
                    x + bar_positions[i],
                    method_data[method],
                    width,
                    label=method.replace("_", " ").title(),
                    alpha=0.8,
                    color=color,
                )

        ax1.set_xlabel("Target Languages")
        ax1.set_ylabel("Performance Score")
        ax1.set_title("Advanced Merging Methods Performance Comparison (Part 1)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(locales, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        methods_2 = advanced_methods[3:]
        colors_2 = ["brown", "cyan", "magenta"]
        bar_positions_2 = np.arange(-1, len(methods_2)) * width

        for i, (method, color) in enumerate(zip(methods_2, colors_2)):
            if method in method_data and method_data[method]:
                ax2.bar(
                    x + bar_positions_2[i],
                    method_data[method],
                    width,
                    label=method.replace("_", " ").title(),
                    alpha=0.8,
                    color=color,
                )

        ax2.bar(
            x + bar_positions_2[-1],
            method_data["baseline"],
            width,
            label="Baseline",
            alpha=0.8,
            color="gray",
        )

        ax2.set_xlabel("Target Languages")
        ax2.set_ylabel("Performance Score")
        ax2.set_title("Advanced Merging Methods Performance Comparison (Part 2)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(locales, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / f"advanced_merging_comparison_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_ensemble_comparison_plot(
        self, ensemble_results: List[Dict], timestamp: str
    ) -> None:
        """Create ensemble methods comparison."""
        print("Creating ensemble comparison...")

        if not ensemble_results:
            print("No ensemble results found")
            return

        locale_data = {}
        for result in ensemble_results:
            locale = result["target_locale"]
            if locale not in locale_data:
                locale_data[locale] = {}
            locale_data[locale][result["ensemble_method"]] = result["ensemble_accuracy"]
            locale_data[locale]["baseline"] = result.get("baseline_accuracy", 0)
            locale_data[locale]["avg_zero_shot"] = result.get("avg_zero_shot", 0)
            locale_data[locale]["best_zero_shot"] = result.get("best_zero_shot", 0)

        locales = list(locale_data.keys())
        methods = ["majority", "weighted_majority", "soft", "uriel_logits"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        x = np.arange(len(locales))
        width = 0.15

        baseline_scores = []
        for locale in locales:
            base_val = locale_data[locale].get("baseline", 0)
            if base_val is None or pd.isna(base_val):
                base_val = 0
            baseline_scores.append(base_val)
        ax1.bar(
            x - 1.5 * width, baseline_scores, width, label="Baseline", alpha=0.7, color="gray"
        )

        for i, method in enumerate(methods):
            scores = []
            for locale in locales:
                val = locale_data[locale].get(method, 0)
                if val is None or pd.isna(val):
                    val = 0
                scores.append(val)
            ax1.bar(
                x + (i - 0.5) * width,
                scores,
                width,
                label=method.replace("_", " ").title(),
                alpha=0.8,
            )

        ax1.set_xlabel("Target Languages")
        ax1.set_ylabel("Performance Score")
        ax1.set_title("Ensemble Methods Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(locales, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for i, method in enumerate(methods):
            advantages = []
            for locale in locales:
                ensemble_score = locale_data[locale].get(method, 0)
                if ensemble_score is None or pd.isna(ensemble_score):
                    ensemble_score = 0
                avg_zero_shot = locale_data[locale].get("avg_zero_shot", 0)
                if avg_zero_shot is None or pd.isna(avg_zero_shot):
                    avg_zero_shot = 0
                advantages.append(ensemble_score - avg_zero_shot)

            bars = ax2.bar(
                x + (i - 1.5) * width,
                advantages,
                width,
                label=f"{method} vs Zero-shot",
                alpha=0.8,
            )

            for bar in bars:
                if bar.get_height() >= 0:
                    bar.set_color("green")
                else:
                    bar.set_color("lightcoral")

        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.set_xlabel("Target Languages")
        ax2.set_ylabel("Advantage over Average Zero-shot")
        ax2.set_title("Ensemble Methods Advantage Over Zero-shot Performance")
        ax2.set_xticks(x)
        ax2.set_xticklabels(locales, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / f"ensemble_comparison_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_num_languages_separated_plots(
        self, merging_results: List[Dict], timestamp: str
    ) -> None:
        """Generate separate plots grouped by num_languages."""
        print("Generating num_languages separated plots...")

        grouped_results: Dict[int, Dict[str, Dict[str, float]]] = {}
        for result in merging_results:
            locale = result["target_locale"]
            baseline_score = result.get("baseline", 0)
            num_lang_map = result.get("num_languages_map", {})

            if not num_lang_map:
                legacy_count = result.get("num_languages")
                if legacy_count:
                    locale_entry = grouped_results.setdefault(legacy_count, {}).setdefault(
                        locale, {}
                    )
                    locale_entry["baseline"] = baseline_score
                continue

            for method_key, num_lang in num_lang_map.items():
                locale_entry = grouped_results.setdefault(num_lang, {}).setdefault(locale, {})
                locale_entry[method_key] = result.get(method_key, 0)
                locale_entry["baseline"] = baseline_score

        if len(grouped_results) <= 1:
            print("Only one num_languages group found, skipping separate plots")
            return

        for num_lang, locale_map in grouped_results.items():
            print(f"Generating plots for {num_lang} languages ({len(locale_map)} locales)...")

            locales = sorted(locale_map.keys())
            method_keys = sorted(
                {
                    key
                    for locale_dict in locale_map.values()
                    for key in locale_dict.keys()
                    if key != "baseline"
                }
            )

            if not method_keys:
                print(f"  No method variants found for {num_lang} languages, skipping")
                continue

            group_data = {"locale": locales}
            for method_key in method_keys:
                group_data[method_key] = [
                    locale_map[loc].get(method_key, 0) for loc in locales
                ]
            group_data["baseline"] = [locale_map[loc].get("baseline", 0) for loc in locales]

            group_df = pd.DataFrame(group_data).set_index("locale")

            self._generate_group_pure_scores_plot(group_df, num_lang, timestamp)

            if "baseline" in group_df.columns:
                self._generate_group_improvement_plot(group_df, num_lang, timestamp)

        print(f"Generated plots for {len(grouped_results)} num_languages groups")

    def _generate_group_pure_scores_plot(
        self, df: pd.DataFrame, num_lang: int, timestamp: str
    ) -> None:
        """Generate pure scores plot for a specific num_languages group."""
        available_methods = [
            col for col in df.columns if col not in ["locale", "baseline"] and df[col].notna().any()
        ]

        if not available_methods:
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        locales = df.index.tolist()
        x = np.arange(len(locales))
        width = 0.1

        colors = plt.cm.Set3(np.linspace(0, 1, len(available_methods)))

        for i, method in enumerate(available_methods):
            scores = [
                df.loc[locale, method]
                if method in df.columns and pd.notna(df.loc[locale, method])
                else 0
                for locale in locales
            ]

            display_name = format_method_key_for_display(method)
            ax.bar(
                x + i * width, scores, width, label=display_name, alpha=0.8, color=colors[i]
            )

        ax.set_xlabel("Target Languages")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Method Performance Comparison ({num_lang} Languages Used)")
        ax.set_xticks(x + width * len(available_methods) / 2)
        ax.set_xticklabels(locales, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        filename = f"num_languages_{num_lang}_pure_scores_{timestamp}.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    def _generate_group_improvement_plot(
        self, df: pd.DataFrame, num_lang: int, timestamp: str
    ) -> None:
        """Generate improvement over baseline plot for a specific num_languages group."""
        available_methods = [
            col for col in df.columns if col not in ["locale", "baseline"] and df[col].notna().any()
        ]

        if not available_methods or "baseline" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        locales = df.index.tolist()
        x = np.arange(len(locales))
        width = 0.1

        colors = plt.cm.Set3(np.linspace(0, 1, len(available_methods)))

        for i, method in enumerate(available_methods):
            improvements = []
            for locale in locales:
                method_score = (
                    df.loc[locale, method]
                    if method in df.columns and pd.notna(df.loc[locale, method])
                    else 0
                )
                baseline_score = (
                    df.loc[locale, "baseline"]
                    if "baseline" in df.columns and pd.notna(df.loc[locale, "baseline"])
                    else 0
                )
                improvements.append(method_score - baseline_score)

            display_name = format_method_key_for_display(method)
            bars = ax.bar(
                x + i * width, improvements, width, label=display_name, alpha=0.8, color=colors[i]
            )

            for bar in bars:
                if bar.get_height() >= 0:
                    bar.set_alpha(0.8)
                else:
                    bar.set_alpha(0.6)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=2)
        ax.set_xlabel("Target Languages")
        ax.set_ylabel("Improvement over Baseline")
        ax.set_title(f"Improvement over Baseline ({num_lang} Languages Used)")
        ax.set_xticks(x + width * len(available_methods) / 2)
        ax.set_xticklabels(locales, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"num_languages_{num_lang}_improvement_{timestamp}.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    def create_pure_scores_plot_for_group(
        self,
        locales: List[str],
        scores: List[float],
        baseline_scores: List[float],
        avg_zero_scores: List[float],
        best_zero_scores: List[float],
        best_source_scores: List[float],
        method: str,
        num_lang: int,
        timestamp: str,
    ) -> None:
        """Create pure scores plot for a specific method and num_languages group."""
        fig, ax = plt.subplots(figsize=(20, 8))
        display_name = format_method_key_for_display(method)
        file_method = format_method_key_for_filename(method)

        x = np.arange(len(locales))
        width = 0.16

        ax.bar(x - 2 * width, baseline_scores, width, label="Baseline", alpha=0.7, color="gray")
        ax.bar(
            x - width, avg_zero_scores, width, label="Avg Zero-shot", alpha=0.7, color="lightblue"
        )
        ax.bar(x, best_zero_scores, width, label="Best Zero-shot", alpha=0.7, color="lightgreen")
        ax.bar(
            x + width, best_source_scores, width, label="Best Source", alpha=0.7, color="lightcoral"
        )
        bars5 = ax.bar(
            x + 2 * width, scores, width, label=display_name, alpha=0.8, color="royalblue"
        )

        for bar in bars5:
            height = bar.get_height()
            if height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlabel("Target Languages")
        ax.set_ylabel("Performance Score")
        ax.set_title(f"Pure Performance: {display_name} vs All Baselines ({num_lang} Languages)")
        ax.set_xticks(x)
        ax.set_xticklabels(locales, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.plots_dir / f"pure_scores_{file_method}_{num_lang}lang_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Pure scores plot saved: {output_file}")

    def create_improvement_plot(
        self,
        locales: List[str],
        improvements: List[float],
        method: str,
        num_lang: int,
        baseline_name: str,
        prefix: str,
        timestamp: str,
        pos_color: str,
        neg_color: str,
    ) -> None:
        """Create improvement plot for a specific baseline."""
        fig, ax = plt.subplots(figsize=(16, 8))
        display_name = format_method_key_for_display(method)
        file_method = format_method_key_for_filename(method)

        x = np.arange(len(locales))
        width = 0.6

        bars = ax.bar(x, improvements, width, alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.001:
                if height >= 0:
                    bar.set_color(pos_color)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"+{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    bar.set_color(neg_color)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=2)

        stats = compute_statistics(improvements)
        if stats["total_count"] > 0:
            stats_text = (
                f"Mean: {stats['mean']:+.4f}\n"
                f"Win Rate: {stats['win_rate']:.1f}%\n"
                f"Count: {stats['total_count']}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        ax.set_xlabel("Target Languages")
        ax.set_ylabel(f"Improvement over {baseline_name}")
        ax.set_title(f"{display_name} vs {baseline_name} ({num_lang} Languages)")
        ax.set_xticks(x)
        ax.set_xticklabels(locales, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.plots_dir / f"{prefix}_{file_method}_{num_lang}lang_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    {prefix} plot saved: {output_file}")
