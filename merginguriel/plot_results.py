"""
Advanced Results Analyzer for MergingUriel experiments.

This module provides the main AdvancedResultsAnalyzer class that orchestrates
results analysis and plot generation using modular components from the
plotting package.
"""

import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from merginguriel.plotting.data_loader import ResultsDataLoader
from merginguriel.plotting.generators import PlotGenerator
from merginguriel.plotting.utils import (
    safe_float,
    maybe_float,
    extract_baselines,
    format_method_key_for_filename,
    format_method_key_for_display,
    get_method_num_language_set,
    get_method_model_family,
    get_method_similarity_type,
    infer_model_family_from_method_key,
    get_method_columns,
)

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 300


class AdvancedResultsAnalyzer:
    """Main analyzer class that orchestrates results analysis using composition."""

    def __init__(
        self,
        results_dir: str = ".",
        plots_dir: str = "plots",
        num_languages_filter: Optional[List[int]] = None,
        similarity_types: Optional[List[str]] = None,
    ):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.num_languages_filter = num_languages_filter
        self.similarity_types = similarity_types

        # Initialize data loader
        self.data_loader = ResultsDataLoader(
            results_dir=results_dir,
            num_languages_filter=num_languages_filter,
            similarity_types=similarity_types,
        )

        # Initialize plot generator
        self.plot_generator = PlotGenerator(self.plots_dir)

        # Load all data
        self.data_loader.load_all_data()

        # Create convenience aliases for data loader properties
        self.main_results_df = self.data_loader.main_results_df
        self.nxn_matrices = self.data_loader.nxn_matrices
        self.experiment_results = self.data_loader.experiment_results

    # Delegate utility methods for backward compatibility
    def _safe_float(self, value: Any) -> float:
        return safe_float(value)

    def _maybe_float(self, value: Any) -> Optional[float]:
        return maybe_float(value)

    def _extract_baselines(self, row: pd.Series):
        return extract_baselines(row)

    def format_method_key_for_filename(
        self, method_key: str, model_family: str = None, similarity_type: str = None
    ) -> str:
        return format_method_key_for_filename(method_key, model_family, similarity_type)

    def format_method_key_for_display(
        self, method_key: str, model_family: str = None, similarity_type: str = None
    ) -> str:
        return format_method_key_for_display(method_key, model_family, similarity_type)

    def get_method_model_family(self, method: str) -> str:
        return get_method_model_family(method, self.experiment_results)

    def get_method_similarity_type(self, method: str) -> str:
        return get_method_similarity_type(method, self.experiment_results)

    def infer_model_family_from_method_key(self, method_key: str) -> Optional[str]:
        return infer_model_family_from_method_key(method_key)

    def get_method_num_language_set(self, summary_df: pd.DataFrame, method: str) -> set:
        return get_method_num_language_set(summary_df, method)

    # Delegate data loading methods
    def _get_nxn_for_family(self, model_family: Optional[str]) -> Optional[pd.DataFrame]:
        return self.data_loader.get_nxn_for_family(model_family)

    def get_zero_shot_scores(
        self, target_locale: str, source_locales: List[str], model_family: Optional[str]
    ) -> Dict[str, float]:
        return self.data_loader.get_zero_shot_scores(target_locale, source_locales, model_family)

    def get_best_source_performance(
        self, target_locale: str, source_locales: List[str], model_family: Optional[str] = None
    ) -> Optional[float]:
        return self.data_loader.get_best_source_performance(
            target_locale, source_locales, model_family
        )

    def get_best_overall_zero_shot(
        self, target_locale: str, model_family: Optional[str]
    ) -> Optional[float]:
        return self.data_loader.get_best_overall_zero_shot(target_locale, model_family)

    def find_merge_locales(self, target_locale: str, merge_type: str = "similarity") -> List[str]:
        return self.data_loader.find_merge_locales(target_locale, merge_type)

    def extract_num_languages_from_details(
        self, target_locale: str, merge_type: str
    ) -> Optional[int]:
        return self.data_loader.extract_num_languages_from_details(target_locale, merge_type)

    def extract_source_locales_from_details(
        self, target_locale: str, model_family: Optional[str] = None, num_languages: Optional[int] = None
    ) -> List[str]:
        return self.data_loader.extract_source_locales_from_details(
            target_locale, model_family, num_languages
        )

    def get_num_languages_from_merged_models(self, locale: str, method: str) -> Optional[int]:
        return self.data_loader.get_num_languages_from_merged_models(locale, method)

    def analyze_advanced_merging_methods(self) -> List[Dict]:
        """Analyze results for advanced merging methods."""
        print("Analyzing advanced merging methods...")

        method_columns = [
            col
            for col in self.main_results_df.columns
            if col not in ["locale", "baseline", "best_source_accuracy", "best_overall_accuracy"]
            and not col.endswith("_improvement")
            and "_vs_" not in col
            and not col.startswith("baseline_")
        ]
        results = []

        for locale in self.main_results_df["locale"].unique():
            locale_data = {"target_locale": locale}

            locale_row = self.main_results_df[self.main_results_df["locale"] == locale]
            if len(locale_row) == 0:
                continue

            main_data = locale_row.iloc[0]

            baseline_value, baseline_map = self._extract_baselines(main_data)
            locale_data["baseline"] = baseline_value
            for family, score in baseline_map.items():
                locale_data[f"baseline_{family}"] = score

            num_lang_map = {}
            for method_key in method_columns:
                if method_key in main_data and pd.notna(main_data[method_key]):
                    method_val = main_data[method_key]
                    if method_val != "" and method_val is not None:
                        locale_data[method_key] = float(method_val)
                    match = re.search(r"_(\d+)lang$", method_key)
                    if match:
                        num_lang_map[method_key] = int(match.group(1))

            if num_lang_map:
                locale_data["num_languages_map"] = num_lang_map

            locale_family = None
            if "baseline_xlm-roberta-large" in baseline_map:
                locale_family = "xlm-roberta-large"
            elif "baseline_xlm-roberta-base" in baseline_map:
                locale_family = "xlm-roberta-base"
            else:
                for method_key in method_columns:
                    if "xlm-roberta-large" in method_key:
                        locale_family = "xlm-roberta-large"
                        break
                    if "xlm-roberta-base" in method_key:
                        locale_family = "xlm-roberta-base"
                        break
            locale_data["zero_shot_family"] = locale_family

            num_languages = self.extract_num_languages_from_details(locale, "similarity")
            if num_languages is None:
                for method in ["average", "fisher", "ties"]:
                    num_languages = self.extract_num_languages_from_details(locale, method)
                    if num_languages is not None:
                        break

            source_locales = self.extract_source_locales_from_details(
                locale, locale_family, num_languages
            )
            zero_shot_scores = self.get_zero_shot_scores(locale, source_locales, locale_family)

            if self.num_languages_filter is not None:
                candidate_counts = set(num_lang_map.values()) if num_lang_map else set()
                if num_languages is not None:
                    candidate_counts.add(num_languages)
                if not candidate_counts:
                    print(
                        f"Skipping {locale} - has {num_languages} languages, not in filter {self.num_languages_filter}"
                    )
                    continue
                if not any(count in self.num_languages_filter for count in candidate_counts):
                    print(
                        f"Skipping {locale} - has {candidate_counts} languages, not in filter {self.num_languages_filter}"
                    )
                    continue

            if zero_shot_scores:
                locale_data["avg_zero_shot"] = np.mean(list(zero_shot_scores.values()))
            else:
                locale_data["avg_zero_shot"] = None

            locale_data["best_zero_shot"] = self.get_best_overall_zero_shot(locale, locale_family)
            locale_data["best_source"] = self.get_best_source_performance(
                locale, source_locales, locale_family
            )
            locale_data["source_locales"] = source_locales or []
            locale_data["num_languages"] = num_languages
            results.append(locale_data)

        return results

    def analyze_ensemble_methods(self) -> List[Dict]:
        """Analyze ensemble inference methods from the main aggregated results."""
        print("Analyzing ensemble methods...")

        ensemble_methods = [
            "ensemble_majority",
            "ensemble_weighted_majority",
            "ensemble_soft",
            "ensemble_uriel_logits",
        ]
        results = []

        for _, row in self.main_results_df.iterrows():
            locale = row["locale"]

            baseline_accuracy, _ = self._extract_baselines(row)

            source_locales = self.extract_source_locales_from_details(locale)
            locale_family = None
            if "baseline_xlm-roberta-large" in row.index and not pd.isna(
                row.get("baseline_xlm-roberta-large")
            ):
                locale_family = "xlm-roberta-large"
            elif "baseline_xlm-roberta-base" in row.index and not pd.isna(
                row.get("baseline_xlm-roberta-base")
            ):
                locale_family = "xlm-roberta-base"
            zero_shot_scores = self.get_zero_shot_scores(locale, source_locales, locale_family)

            for method in ensemble_methods:
                if method in row and pd.notna(row[method]):
                    method_name = method.replace("ensemble_", "")

                    locale_data = {
                        "target_locale": locale,
                        "ensemble_method": method_name,
                        "ensemble_accuracy": row[method],
                        "baseline_accuracy": baseline_accuracy,
                        "source_locales": source_locales,
                        "zero_shot_family": locale_family,
                    }

                    if zero_shot_scores:
                        locale_data["avg_zero_shot"] = np.mean(list(zero_shot_scores.values()))
                        locale_data["best_zero_shot"] = max(zero_shot_scores.values())
                        locale_data["best_source"] = self.get_best_source_performance(
                            locale, source_locales
                        )
                    else:
                        locale_data["avg_zero_shot"] = None
                        locale_data["best_zero_shot"] = None
                        locale_data["best_source"] = None

                    results.append(locale_data)

        return results

    def create_comprehensive_summary(
        self,
        merging_results: List[Dict],
        ensemble_results: List[Dict],
        nvn_df=None,
        available_locales=None,
    ) -> pd.DataFrame:
        """Create comprehensive summary CSV with all methods vs baselines."""
        print("Creating comprehensive summary...")

        locale_data = {}

        metadata_keys = {
            "target_locale",
            "baseline",
            "baseline_map",
            "avg_zero_shot",
            "best_zero_shot",
            "best_source",
            "source_locales",
            "zero_shot_family",
            "num_languages",
            "num_languages_map",
        }

        for result in merging_results:
            locale = result["target_locale"]
            entry = locale_data.setdefault(
                locale,
                {
                    "locale": locale,
                    "baseline": self._maybe_float(result.get("baseline")),
                    "avg_zero_shot": self._maybe_float(result.get("avg_zero_shot")),
                    "best_zero_shot": self._maybe_float(result.get("best_zero_shot")),
                    "best_source": self._maybe_float(result.get("best_source")),
                    "source_locales": result.get("source_locales", []),
                    "zero_shot_family": result.get("zero_shot_family"),
                },
            )

            for key in ["baseline", "avg_zero_shot", "best_zero_shot", "best_source"]:
                if key in result and (entry.get(key) in (None, 0)):
                    entry[key] = self._maybe_float(result.get(key))
            if "source_locales" in result and not entry.get("source_locales"):
                entry["source_locales"] = result["source_locales"]
            if "zero_shot_family" in result and not entry.get("zero_shot_family"):
                entry["zero_shot_family"] = result.get("zero_shot_family")
            if "num_languages_map" in result:
                entry.setdefault("num_languages_map", {}).update(result["num_languages_map"])

            for method_key, value in result.items():
                if method_key in metadata_keys:
                    continue
                is_baseline_col = method_key.startswith("baseline_")
                method_val = self._maybe_float(value)
                if method_val is None:
                    continue
                entry[method_key] = method_val
                if is_baseline_col:
                    continue
                method_family = self.get_method_model_family(method_key)
                zero_family = entry.get("zero_shot_family")
                family_mismatch = zero_family and method_family and zero_family != method_family

                avg_zero = self._maybe_float(entry.get("avg_zero_shot"))
                best_zero = self._maybe_float(entry.get("best_zero_shot"))
                best_source = self._maybe_float(entry.get("best_source"))

                if family_mismatch:
                    avg_zero = best_zero = best_source = None

                if avg_zero is not None:
                    entry[f"{method_key}_vs_avg_zero"] = method_val - avg_zero
                if best_zero is not None:
                    entry[f"{method_key}_vs_best_zero"] = method_val - best_zero
                if best_source is not None:
                    entry[f"{method_key}_vs_best_source"] = method_val - best_source

        for result in ensemble_results:
            locale = result["target_locale"]
            method = result["ensemble_method"]
            accuracy = self._maybe_float(result.get("ensemble_accuracy", 0))
            if accuracy is None:
                continue

            if locale not in locale_data:
                locale_data[locale] = {
                    "locale": locale,
                    "baseline": self._maybe_float(result.get("baseline_accuracy")),
                    "avg_zero_shot": self._maybe_float(result.get("avg_zero_shot")),
                    "best_zero_shot": self._maybe_float(result.get("best_zero_shot")),
                    "best_source": self._maybe_float(result.get("best_source")),
                    "source_locales": result.get("source_locales", []),
                    "zero_shot_family": result.get("zero_shot_family"),
                }

            locale_data[locale][method] = accuracy
            zero_family = locale_data[locale].get("zero_shot_family")
            avg_zero = self._maybe_float(locale_data[locale].get("avg_zero_shot"))
            best_zero = self._maybe_float(locale_data[locale].get("best_zero_shot"))
            best_source = self._maybe_float(locale_data[locale].get("best_source"))

            if zero_family is None:
                avg_zero = best_zero = best_source = None

            if avg_zero is not None:
                locale_data[locale][f"{method}_vs_avg_zero"] = accuracy - avg_zero
            if best_zero is not None:
                locale_data[locale][f"{method}_vs_best_zero"] = accuracy - best_zero
            if best_source is not None:
                locale_data[locale][f"{method}_vs_best_source"] = accuracy - best_source

        summary_records = []
        for locale, data in locale_data.items():
            record = dict(data)
            if "num_languages_map" in record and isinstance(record["num_languages_map"], dict):
                record["num_languages_map"] = json.dumps(record["num_languages_map"])
            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records)

        static_cols = {
            "locale",
            "baseline",
            "avg_zero_shot",
            "best_zero_shot",
            "best_source",
            "source_locales",
            "num_languages_map",
            "zero_shot_family",
        }
        method_cols = [
            c
            for c in summary_df.columns
            if c not in static_cols and not c.startswith("baseline") and "_vs_" not in c
        ]
        for method in method_cols:
            method_family = self.get_method_model_family(method)
            vs_avg_col = f"{method}_vs_avg_zero"
            vs_best_col = f"{method}_vs_best_zero"
            vs_source_col = f"{method}_vs_best_source"

            values = pd.to_numeric(summary_df[method], errors="coerce")
            vs_avg_list = []
            vs_best_list = []
            vs_source_list = []

            for idx, row in summary_df.iterrows():
                val = values.iloc[idx]
                if pd.isna(val):
                    vs_avg_list.append(np.nan)
                    vs_best_list.append(np.nan)
                    vs_source_list.append(np.nan)
                    continue

                target_locale = row["locale"]
                source_locales = row.get("source_locales") or []
                if isinstance(source_locales, str):
                    try:
                        source_locales = json.loads(source_locales)
                    except Exception:
                        source_locales = []

                zs_scores = self.get_zero_shot_scores(target_locale, source_locales, method_family)
                avg_zero = np.mean(list(zs_scores.values())) if zs_scores else np.nan
                best_zero = self.get_best_overall_zero_shot(target_locale, method_family)
                best_source = self.get_best_source_performance(
                    target_locale, source_locales, method_family
                )

                vs_avg_list.append(val - avg_zero if pd.notna(avg_zero) else np.nan)
                vs_best_list.append(val - best_zero if best_zero is not None else np.nan)
                vs_source_list.append(val - best_source if best_source is not None else np.nan)

            summary_df[vs_avg_col] = vs_avg_list
            summary_df[vs_best_col] = vs_best_list
            summary_df[vs_source_col] = vs_source_list

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_df.to_csv(f"advanced_analysis_summary_{timestamp}.csv", index=False)

        print(f"Comprehensive summary saved to: advanced_analysis_summary_{timestamp}.csv")

        self.print_summary_statistics(summary_df)

        return summary_df

    def print_summary_statistics(self, summary_df: pd.DataFrame) -> None:
        """Print comprehensive summary statistics."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\nTotal locales analyzed: {len(summary_df)}")

        if "baseline" in summary_df.columns:
            print(f"\nAverage baseline: {summary_df['baseline'].mean():.4f}")

        static_fields = {
            "locale",
            "baseline",
            "avg_zero_shot",
            "best_zero_shot",
            "best_source",
            "source_locales",
            "num_languages_map",
            "zero_shot_family",
        }

        method_cols = [
            col for col in summary_df.columns if col not in static_fields and "_vs_" not in col
        ]
        if method_cols:
            print("\n--- METHOD PERFORMANCE ---")
            for method in sorted(method_cols):
                avg_score = summary_df[method].mean()
                print(f"Average {method}: {avg_score:.4f}")

        improvement_cols = [col for col in summary_df.columns if col.endswith("_vs_avg_zero")]
        if improvement_cols:
            print("\n--- IMPROVEMENT VS AVG ZERO-SHOT ---")
            for col in sorted(improvement_cols):
                avg_improvement = summary_df[col].mean()
                positive_count = (summary_df[col] > 0).sum()
                print(
                    f"{col}: avg {avg_improvement:+.4f}, positive in {positive_count}/{len(summary_df)} locales"
                )

        best_zero_cols = [col for col in summary_df.columns if col.endswith("_vs_best_zero")]
        if best_zero_cols:
            print("\n--- IMPROVEMENT VS BEST ZERO-SHOT ---")
            for col in sorted(best_zero_cols):
                avg_improvement = summary_df[col].mean()
                positive_count = (summary_df[col] > 0).sum()
                print(
                    f"{col}: avg {avg_improvement:+.4f}, positive in {positive_count}/{len(summary_df)} locales"
                )

        best_source_cols = [col for col in summary_df.columns if col.endswith("_vs_best_source")]
        if best_source_cols:
            print("\n--- IMPROVEMENT VS BEST SOURCE ---")
            for col in sorted(best_source_cols):
                avg_improvement = summary_df[col].mean()
                positive_count = (summary_df[col] > 0).sum()
                print(
                    f"{col}: avg {avg_improvement:+.4f}, positive in {positive_count}/{len(summary_df)} locales"
                )

        if "avg_zero_shot" in summary_df.columns:
            print("\n--- ZERO-SHOT COMPARISON ---")
            print(f"Average zero-shot performance: {summary_df['avg_zero_shot'].mean():.4f}")
            print(f"Best zero-shot performance: {summary_df['best_zero_shot'].max():.4f}")
            print(f"Best source performance: {summary_df['best_source'].max():.4f}")

    def generate_advanced_analysis(self) -> pd.DataFrame:
        """Main method to generate complete advanced analysis."""
        print("Starting Advanced Results Analysis...")
        print("=" * 80)

        self.plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Plots will be saved to: {self.plots_dir}")

        merging_results = self.analyze_advanced_merging_methods()
        ensemble_results = self.analyze_ensemble_methods()
        summary_df = self.create_comprehensive_summary(merging_results, ensemble_results, None)

        if summary_df is not None:
            # Generate plots using the plot generator
            self.plot_generator.generate_pure_scores_plots(
                summary_df,
                timestamp,
                self.get_method_model_family,
                self.get_method_similarity_type,
                lambda method: self.get_method_num_language_set(summary_df, method),
            )
            self.plot_generator.generate_vs_avg_zero_plots(
                summary_df,
                timestamp,
                self.get_method_model_family,
                self.get_method_similarity_type,
            )
            self.plot_generator.generate_vs_best_zero_plots(
                summary_df,
                timestamp,
                self.get_method_model_family,
                self.get_method_similarity_type,
            )
            self.plot_generator.generate_vs_best_source_plots(
                summary_df,
                timestamp,
                self.get_method_model_family,
                self.get_method_similarity_type,
            )
            self.plot_generator.generate_baseline_comparison_plots(
                summary_df,
                timestamp,
                self.get_method_model_family,
                self.get_method_similarity_type,
                self.infer_model_family_from_method_key,
            )

            if self.num_languages_filter is not None or len(merging_results) > 0:
                self.plot_generator.generate_num_languages_separated_plots(
                    merging_results, timestamp
                )
                self._generate_num_languages_method_plots(summary_df, timestamp)

        print("\nAdvanced analysis complete!")
        print("Generated files:")
        print(f"- advanced_analysis_summary_{timestamp}.csv")
        print("- Individual pure score plots for each method")
        print("- Individual vs avg zero-shot comparison plots for each method")
        print("- Individual vs best zero-shot comparison plots for each method")
        print("- Individual vs best source comparison plots for each method (FAIR COMPARISON)")
        print("- Baseline comparison plots for each method")
        print(f"All plots saved in {self.plots_dir} directory with {timestamp} suffix")

        return summary_df

    def _generate_num_languages_method_plots(
        self, summary_df: pd.DataFrame, timestamp: str
    ) -> None:
        """Generate separate plots for each method separated by num_languages."""
        print("Generating num_languages separated plots for ALL methods...")

        methods = get_method_columns(summary_df)

        locale_method_num_lang = {}

        for _, row in summary_df.iterrows():
            locale = row["locale"]
            raw_map = row.get("num_languages_map")
            num_lang_map = {}
            if isinstance(raw_map, str) and raw_map:
                num_lang_map = json.loads(raw_map)

            for method in methods:
                val = pd.to_numeric(row.get(method), errors="coerce")
                if pd.notna(val) and val > 0:
                    num_lang = None
                    if method in num_lang_map:
                        num_lang = num_lang_map[method]
                    else:
                        match = re.match(r"(.+?)_(\d+)lang$", method)
                        if match:
                            num_lang = int(match.group(2))
                        if num_lang is None:
                            num_lang = self.get_num_languages_from_merged_models(locale, method)

                    if num_lang:
                        locale_method_num_lang[(locale, method)] = num_lang

        if not locale_method_num_lang:
            print("No num_languages information found in merged_models folder")
            return

        grouped_dfs = {}
        for (locale, method), num_lang in locale_method_num_lang.items():
            if num_lang not in grouped_dfs:
                grouped_dfs[num_lang] = []
            locale_row = summary_df[summary_df["locale"] == locale].iloc[0]
            grouped_dfs[num_lang].append((locale, method, locale_row))

        print(f"Found num_languages groups: {sorted(grouped_dfs.keys())}")

        for num_lang, entries in grouped_dfs.items():
            print(f"Generating plots for {num_lang} languages ({len(entries)} entries)...")

            method_data = {method: [] for method in methods}

            for locale, method, row in entries:
                method_data[method].append((locale, row[method]))

            for method in methods:
                if not method_data[method]:
                    continue

                print(f"  Creating {method} plots for {num_lang} languages...")

                locales = []
                scores = []
                baseline_scores = []
                avg_zero_scores = []
                best_zero_scores = []
                best_source_scores = []

                for locale, score in method_data[method]:
                    locale_row = summary_df[summary_df["locale"] == locale].iloc[0]
                    locales.append(locale)
                    scores.append(score)
                    baseline_scores.append(locale_row.get("baseline", 0))
                    avg_zero_scores.append(locale_row.get("avg_zero_shot", 0))
                    best_zero_scores.append(locale_row.get("best_zero_shot", 0))
                    best_source_scores.append(locale_row.get("best_source", 0))

                self.plot_generator.create_pure_scores_plot_for_group(
                    locales,
                    scores,
                    baseline_scores,
                    avg_zero_scores,
                    best_zero_scores,
                    best_source_scores,
                    method,
                    num_lang,
                    timestamp,
                )

                avg_zero_improvements = [s - a for s, a in zip(scores, avg_zero_scores)]
                self.plot_generator.create_improvement_plot(
                    locales,
                    avg_zero_improvements,
                    method,
                    num_lang,
                    "Average Zero-shot",
                    "vs_avg_zero",
                    timestamp,
                    "green",
                    "lightcoral",
                )

                best_zero_improvements = [s - b for s, b in zip(scores, best_zero_scores)]
                self.plot_generator.create_improvement_plot(
                    locales,
                    best_zero_improvements,
                    method,
                    num_lang,
                    "Best Zero-shot",
                    "vs_best_zero",
                    timestamp,
                    "royalblue",
                    "orange",
                )

                best_source_improvements = [s - b for s, b in zip(scores, best_source_scores)]
                self.plot_generator.create_improvement_plot(
                    locales,
                    best_source_improvements,
                    method,
                    num_lang,
                    "Best Source",
                    "vs_best_source",
                    timestamp,
                    "forestgreen",
                    "indianred",
                )

        print(f"Generated method-specific plots for {len(grouped_dfs)} num_languages groups")


def main():
    """Main function to run the advanced analysis system."""
    parser = argparse.ArgumentParser(
        description="Enhanced Results Analysis with Similarity Type and num_languages Support"
    )
    parser.add_argument(
        "--num-languages",
        type=str,
        help='Filter by number of languages (comma-separated, e.g., "3,5")',
    )
    parser.add_argument(
        "--similarity-types",
        type=str,
        help='Filter by similarity types (comma-separated, e.g., "URIEL,REAL")',
    )
    parser.add_argument(
        "--list-num-languages",
        action="store_true",
        help="List available num_languages values in data",
    )
    parser.add_argument(
        "--list-similarity-types",
        action="store_true",
        help="List available similarity types in data",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results (default: results)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory for saving plots (default: plots)",
    )

    args = parser.parse_args()

    num_languages_filter = None
    if args.num_languages:
        num_languages_filter = [int(x.strip()) for x in args.num_languages.split(",")]
        print(f"Filtering experiments with num_languages: {num_languages_filter}")

    similarity_types_filter = None
    if args.similarity_types:
        similarity_types_filter = [x.strip().upper() for x in args.similarity_types.split(",")]
        valid_types = {"URIEL", "REAL"}
        invalid_types = [t for t in similarity_types_filter if t not in valid_types]
        if invalid_types:
            print(f"Error: Invalid similarity types: {invalid_types}. Valid types: {valid_types}")
            return None
        print(f"Filtering experiments with similarity types: {similarity_types_filter}")

    if args.list_num_languages:
        analyzer = AdvancedResultsAnalyzer(
            results_dir=args.results_dir, plots_dir=args.plots_dir
        )
        available_num_langs = set()
        for locale in analyzer.main_results_df["locale"].unique():
            for method in ["similarity", "average", "fisher"]:
                num_lang = analyzer.extract_num_languages_from_details(locale, method)
                if num_lang is not None:
                    available_num_langs.add(num_lang)
                    break

        if available_num_langs:
            print(f"Available num_languages in data: {sorted(list(available_num_langs))}")
        else:
            print("No num_languages information found in merge details")
        return None

    if args.list_similarity_types:
        analyzer = AdvancedResultsAnalyzer(
            results_dir=args.results_dir, plots_dir=args.plots_dir
        )
        available_similarity_types = set()
        for exp_name, exp_data in analyzer.experiment_results.items():
            if exp_data.get("similarity_type") and exp_data["similarity_type"] != "unknown":
                available_similarity_types.add(exp_data["similarity_type"])

        if available_similarity_types:
            print(f"Available similarity types in data: {sorted(list(available_similarity_types))}")
        else:
            print("No similarity type information found in experiment results")
        return None

    analyzer = AdvancedResultsAnalyzer(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        num_languages_filter=num_languages_filter,
        similarity_types=similarity_types_filter,
    )
    summary_df = analyzer.generate_advanced_analysis()
    return summary_df


if __name__ == "__main__":
    exit(main())
