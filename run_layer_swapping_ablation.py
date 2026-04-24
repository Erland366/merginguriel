#!/usr/bin/env python3
"""
Layer-swapping ablation experiment runner.

Runs systematic ablations of layer-swapping merges:
  - Which layers to swap (top/bottom sweep, single-layer, groups)
  - Non-layer source (embeddings/classifier from model_a vs model_b)
  - Source model pair selection (NxN-based strategies)

Results are saved per-experiment as JSON and aggregated into summary CSV.
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from merginguriel.evaluate_specific_model import evaluate_specific_model
from merginguriel.layer_swap_ablation import (
    AblationResult,
    MethodComparisonConfig,
    aggregate_results,
    generate_ablation_configs,
    generate_method_comparison_configs,
    get_model_path,
    load_nxn_matrix,
    run_conventional_merge,
    select_source_pairs,
)
from merginguriel.layer_swapping import LayerSwappingConfig, run_layer_swapping_merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _result_exists(results_dir: str, experiment_name: str) -> bool:
    result_file = os.path.join(results_dir, experiment_name, "results.json")
    return os.path.exists(result_file)


def _get_baseline_accuracy(
    nxn_df: pd.DataFrame,
    source_locale: str,
    target_locale: str,
) -> Optional[float]:
    try:
        return float(nxn_df.at[source_locale, target_locale])
    except (KeyError, ValueError):
        return None


def run_single_ablation(
    experiment_name: str,
    model_a_path: str,
    model_b_path: str,
    swap_config: LayerSwappingConfig,
    target_locale: str,
    results_base_dir: str,
    merged_models_dir: str,
    cleanup_after_eval: bool = True,
) -> Dict:
    """Run a single layer-swapping ablation: merge, evaluate, save results."""
    exp_results_dir = os.path.join(results_base_dir, experiment_name)
    os.makedirs(exp_results_dir, exist_ok=True)

    merged_output_path = os.path.join(merged_models_dir, experiment_name)
    os.makedirs(merged_output_path, exist_ok=True)

    # Step 1: Merge
    logger.info(f"Merging: {experiment_name}")
    merge_metadata = run_layer_swapping_merge(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        output_path=merged_output_path,
        config=swap_config,
    )

    # Step 2: Evaluate on target locale
    logger.info(f"Evaluating: {experiment_name} on {target_locale}")
    eval_results = evaluate_specific_model(
        model_name=merged_output_path,
        locale=target_locale,
        eval_folder=exp_results_dir,
    )

    # Step 3: Save merge metadata alongside eval results
    meta_path = os.path.join(exp_results_dir, "merge_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(merge_metadata, f, indent=2)

    # Step 4: Cleanup merged model to save disk space
    if cleanup_after_eval:
        shutil.rmtree(merged_output_path, ignore_errors=True)

    return {
        "merge_metadata": merge_metadata,
        "eval_results": eval_results,
    }


def run_method_comparison_suite(
    target_locale: str,
    model_a_locale: str,
    model_b_locale: str,
    models_root: str,
    base_model: str,
    results_dir: str,
    nxn_matrix_path: str,
    cleanup_after_eval: bool = True,
    resume: bool = True,
) -> List[AblationResult]:
    """Run conventional merge baselines for comparison with layer-swapping."""
    model_a_path = get_model_path(model_a_locale, models_root, base_model)
    model_b_path = get_model_path(model_b_locale, models_root, base_model)

    pair_dir = f"{model_a_locale}__{model_b_locale}"
    results_base = os.path.join(results_dir, target_locale, pair_dir)
    merged_base = os.path.join(results_dir, "_merged_tmp", target_locale, pair_dir)

    nxn_df = load_nxn_matrix(nxn_matrix_path)
    baseline_a = _get_baseline_accuracy(nxn_df, model_a_locale, target_locale)
    baseline_b = _get_baseline_accuracy(nxn_df, model_b_locale, target_locale)

    configs = generate_method_comparison_configs()
    logger.info(f"Generated {len(configs)} method comparison configs")

    results: List[AblationResult] = []

    for i, (exp_name, method_config) in enumerate(configs):
        if resume and _result_exists(results_base, exp_name):
            logger.info(f"[method {i+1}/{len(configs)}] Skipping (exists): {exp_name}")
            result_file = os.path.join(results_base, exp_name, "results.json")
            with open(result_file) as f:
                existing = json.load(f)
            accuracy = existing.get("performance", {}).get("accuracy")
            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=method_config.method,
                swap_indices=[],
                non_layer_source="N/A",
                accuracy=accuracy,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
            ))
            continue

        logger.info(f"[method {i+1}/{len(configs)}] Running: {exp_name}")
        exp_results_dir = os.path.join(results_base, exp_name)
        merged_output_path = os.path.join(merged_base, exp_name)
        os.makedirs(exp_results_dir, exist_ok=True)

        try:
            merge_meta = run_conventional_merge(
                model_a_path=model_a_path,
                model_b_path=model_b_path,
                output_path=merged_output_path,
                config=method_config,
                base_model=base_model,
            )

            logger.info(f"Evaluating: {exp_name} on {target_locale}")
            eval_results = evaluate_specific_model(
                model_name=merged_output_path,
                locale=target_locale,
                eval_folder=exp_results_dir,
            )

            meta_path = os.path.join(exp_results_dir, "merge_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(merge_meta, f, indent=2)

            if cleanup_after_eval:
                shutil.rmtree(merged_output_path, ignore_errors=True)

            accuracy = None
            if eval_results and isinstance(eval_results, dict):
                accuracy = eval_results.get("performance", {}).get("accuracy")

            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=method_config.method,
                swap_indices=[],
                non_layer_source="N/A",
                accuracy=accuracy,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
                metadata=merge_meta,
            ))

        except Exception as e:
            logger.error(f"Failed: {exp_name}: {e}")
            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=method_config.method,
                swap_indices=[],
                non_layer_source="N/A",
                accuracy=None,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
                metadata={"error": str(e)},
            ))

    return results


def run_ablation_suite(
    target_locale: str,
    model_a_locale: str,
    model_b_locale: str,
    ablation_type: str,
    models_root: str,
    base_model: str,
    results_dir: str,
    nxn_matrix_path: str,
    device: str = "cuda",
    cleanup_after_eval: bool = True,
    resume: bool = True,
    include_method_comparison: bool = False,
) -> List[AblationResult]:
    """Run a full ablation suite for one (model_a, model_b, target) triple."""
    model_a_path = get_model_path(model_a_locale, models_root, base_model)
    model_b_path = get_model_path(model_b_locale, models_root, base_model)

    pair_dir = f"{model_a_locale}__{model_b_locale}"
    results_base = os.path.join(results_dir, target_locale, pair_dir)
    merged_base = os.path.join(results_dir, "_merged_tmp", target_locale, pair_dir)

    # Load NxN for baselines
    nxn_df = load_nxn_matrix(nxn_matrix_path)
    baseline_a = _get_baseline_accuracy(nxn_df, model_a_locale, target_locale)
    baseline_b = _get_baseline_accuracy(nxn_df, model_b_locale, target_locale)

    logger.info(f"Target: {target_locale}")
    logger.info(f"Model A: {model_a_locale} (baseline on target: {baseline_a})")
    logger.info(f"Model B: {model_b_locale} (baseline on target: {baseline_b})")

    configs = generate_ablation_configs(ablation_type)
    logger.info(f"Generated {len(configs)} ablation configs (type={ablation_type})")

    results: List[AblationResult] = []

    for i, (exp_name, swap_config) in enumerate(configs):
        swap_config.device = device

        if resume and _result_exists(results_base, exp_name):
            logger.info(f"[{i+1}/{len(configs)}] Skipping (exists): {exp_name}")
            # Load existing result
            result_file = os.path.join(results_base, exp_name, "results.json")
            with open(result_file) as f:
                existing = json.load(f)
            accuracy = existing.get("performance", {}).get("accuracy")
            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=swap_config.swap_strategy,
                swap_indices=[],
                non_layer_source=swap_config.non_layer_source,
                accuracy=accuracy,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
            ))
            continue

        logger.info(f"[{i+1}/{len(configs)}] Running: {exp_name}")
        try:
            run_result = run_single_ablation(
                experiment_name=exp_name,
                model_a_path=model_a_path,
                model_b_path=model_b_path,
                swap_config=swap_config,
                target_locale=target_locale,
                results_base_dir=results_base,
                merged_models_dir=merged_base,
                cleanup_after_eval=cleanup_after_eval,
            )

            merge_meta = run_result["merge_metadata"]
            eval_res = run_result["eval_results"]
            accuracy = None
            if eval_res and isinstance(eval_res, dict):
                accuracy = eval_res.get("performance", {}).get("accuracy")

            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=merge_meta.get("swap_strategy", ""),
                swap_indices=merge_meta.get("swap_indices", []),
                non_layer_source=merge_meta.get("non_layer_source", ""),
                accuracy=accuracy,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
                metadata=merge_meta,
            ))

        except Exception as e:
            logger.error(f"Failed: {exp_name}: {e}")
            results.append(AblationResult(
                experiment_name=exp_name,
                model_a_locale=model_a_locale,
                model_b_locale=model_b_locale,
                target_locale=target_locale,
                swap_strategy=swap_config.swap_strategy,
                swap_indices=[],
                non_layer_source=swap_config.non_layer_source,
                accuracy=None,
                baseline_a_accuracy=baseline_a,
                baseline_b_accuracy=baseline_b,
                metadata={"error": str(e)},
            ))

    # Run method comparison baselines if requested
    if include_method_comparison:
        method_results = run_method_comparison_suite(
            target_locale=target_locale,
            model_a_locale=model_a_locale,
            model_b_locale=model_b_locale,
            models_root=models_root,
            base_model=base_model,
            results_dir=results_dir,
            nxn_matrix_path=nxn_matrix_path,
            cleanup_after_eval=cleanup_after_eval,
            resume=resume,
        )
        results.extend(method_results)

    # Save summary CSV
    df = aggregate_results(results)
    summary_path = os.path.join(results_base, "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layer-swapping ablation experiments."
    )
    parser.add_argument(
        "--target-locale", required=True,
        help="Target MASSIVE locale to evaluate on (e.g., sw-KE)",
    )
    parser.add_argument(
        "--source-strategy",
        choices=["nxn_top2", "nxn_best_plus_diverse", "manual"],
        default="nxn_top2",
        help="Strategy for selecting source model pairs",
    )
    parser.add_argument("--model-a-locale", help="Source locale A (for manual strategy)")
    parser.add_argument("--model-b-locale", help="Source locale B (for manual strategy)")
    parser.add_argument(
        "--ablation-type",
        choices=["top_bottom_sweep", "single_layer", "layer_groups",
                 "non_layer_source", "full"],
        default="full",
        help="Which ablation configs to generate",
    )
    parser.add_argument(
        "--models-root", default="haryos_model",
        help="Root directory containing fine-tuned models",
    )
    parser.add_argument(
        "--base-model", default="xlm-roberta-base",
        help="Base model name prefix in model directory names",
    )
    parser.add_argument(
        "--results-dir", default="results_layer_swap",
        help="Output directory for results",
    )
    parser.add_argument(
        "--nxn-matrix-path",
        default="nxn_results/nxn_eval_20251027_103544/evaluation_matrix_xlm-roberta-base.csv",
        help="Path to NxN evaluation matrix CSV",
    )
    parser.add_argument("--device", default="cuda", help="Device for model loading")
    parser.add_argument(
        "--cleanup-after-eval", action="store_true", default=True,
        help="Delete merged models after evaluation to save disk space",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep merged models after evaluation",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip experiments that already have results",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-run all experiments even if results exist",
    )
    parser.add_argument(
        "--include-method-comparison", action="store_true",
        help="Also run conventional merge baselines (average, TIES, task_arithmetic, DARE)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cleanup = not args.no_cleanup
    resume = not args.no_resume

    # Select source pairs
    pairs = select_source_pairs(
        target_locale=args.target_locale,
        strategy=args.source_strategy,
        nxn_matrix_path=args.nxn_matrix_path,
        model_a_locale=args.model_a_locale,
        model_b_locale=args.model_b_locale,
    )

    logger.info(f"Selected {len(pairs)} source pair(s) for target {args.target_locale}")
    for a, b in pairs:
        logger.info(f"  {a} (base) + {b} (donor)")

    all_results: List[AblationResult] = []

    for model_a_locale, model_b_locale in pairs:
        results = run_ablation_suite(
            target_locale=args.target_locale,
            model_a_locale=model_a_locale,
            model_b_locale=model_b_locale,
            ablation_type=args.ablation_type,
            models_root=args.models_root,
            base_model=args.base_model,
            results_dir=args.results_dir,
            nxn_matrix_path=args.nxn_matrix_path,
            device=args.device,
            cleanup_after_eval=cleanup,
            resume=resume,
            include_method_comparison=args.include_method_comparison,
        )
        all_results.extend(results)

    # Final aggregate summary
    if all_results:
        df = aggregate_results(all_results)
        final_path = os.path.join(args.results_dir, args.target_locale, "all_pairs_summary.csv")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        df.to_csv(final_path, index=False)
        logger.info(f"Final summary: {final_path}")

        # Print top results
        if "accuracy" in df.columns:
            valid = df[df["accuracy"].notna()].sort_values("accuracy", ascending=False)
            if not valid.empty:
                logger.info("\nTop 5 results:")
                for _, row in valid.head(5).iterrows():
                    delta = row.get("delta_vs_best_baseline", "N/A")
                    logger.info(
                        f"  {row['experiment_name']:30s} "
                        f"acc={row['accuracy']:.4f}  "
                        f"delta={delta}"
                    )


if __name__ == "__main__":
    main()
