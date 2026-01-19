import os
import sys
import pandas as pd
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re

# Ensure project root on sys.path for method discovery
# When this script is inside merginguriel/, repo root is one level up
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBMODULES_DIR = os.path.join(REPO_ROOT, "submodules", "auto_merge_llm")
if SUBMODULES_DIR not in sys.path:
    sys.path.insert(0, SUBMODULES_DIR)

from auto_merge_llm.methods import merging_methods_dict
from merginguriel.naming_config import naming_manager
from merginguriel.config import (
    MergingExperimentConfig,
    ConfigLoader,
    TrackProvidedArgsAction,
    MERGING_EXPERIMENT_ARG_MAP,
)

def get_all_locales_from_similarity_matrix(similarity_type="URIEL"):
    """Extract all unique locales from the similarity matrix."""
    if similarity_type == "URIEL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "language_similarity_matrix_unified.csv")
    elif similarity_type == "REAL":
        similarity_matrix_path = os.path.join(REPO_ROOT, "nxn_results", "nxn_eval_20251027_103544", "evaluation_matrix.csv")
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    df = pd.read_csv(similarity_matrix_path, index_col=0)
    # Return unique locale codes from the index (and columns, they should be the same)
    locales = sorted(set(df.index.tolist()))
    return locales

def get_model_for_locale(locale, models_root="haryos_model"):
    """Get the model path for a specific locale using specified model directory."""
    # Try to detect model size from directory name (base/large)
    model_size = "base"  # default
    if "large" in models_root.lower():
        model_size = "large"
    elif "base" in models_root.lower():
        model_size = "base"

    # Use consistent naming pattern: {models_root}/xlm-roberta-{size}_massive_k_{locale}
    model_path = os.path.join(REPO_ROOT, f"{models_root}/xlm-roberta-{model_size}_massive_k_{locale}")

    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"Warning: Model directory not found for locale {locale}: {model_path}")
        return None

    return model_path


def run_merge(mode: str, target_lang: str, extra_args: List[str], base_model: Optional[str] = None, merged_models_dir: str = "merged_models") -> bool:
    """Run the merging pipeline for a specific mode and target language."""
    print(f"Running {mode} merge for {target_lang}...")

    cmd = [sys.executable, os.path.join(REPO_ROOT, "merginguriel", "run_merging_pipeline_refactored.py"),
           "--mode", mode,
           "--target-lang", target_lang,
           "--merged-models-dir", merged_models_dir]
    if base_model:
        cmd.extend(["--base-model", base_model])
    if extra_args:
        cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {mode} merge completed for {target_lang}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode} merge failed for {target_lang}: {e}")
        return False


def detect_num_languages_from_model_path(model_path: Optional[str]) -> Optional[int]:
    """Infer how many source languages were merged for a model directory."""
    if not model_path:
        return None

    normalized_path = model_path.rstrip("/\\")
    base_name = os.path.basename(normalized_path)

    match = re.search(r'_(\d+)merged$', base_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            print(f"Warning: could not parse merged count from {base_name}")

    merge_details_path = os.path.join(normalized_path, "merge_details.txt")
    if os.path.exists(merge_details_path):
        try:
            with open(merge_details_path, "r", encoding="utf-8") as f:
                content = f.read()
            matches = re.findall(r'^\s*\d+\.\s*Model:', content, re.MULTILINE)
            if matches:
                return len(matches)
        except Exception as exc:
            print(f"Warning: unable to count languages in {merge_details_path}: {exc}")

    if "merged_models" in normalized_path:
        # Legacy merges default to 4 merged sources when unspecified.
        return 4

    return None


def results_already_exist(
    experiment_type: str,
    method: str,
    similarity_type: Optional[str],
    locale: str,
    model_family: str,
    results_dir: str,
    num_languages: Optional[int] = None,
    include_target: Optional[bool] = None,
) -> Optional[str]:
    """Return the path to an existing results directory if it contains results.json."""
    base_dir = os.path.join(REPO_ROOT, results_dir)
    if not os.path.exists(base_dir):
        return None

    existing = naming_manager.find_results_directory(
        base_dir,
        experiment_type=experiment_type,
        method=method,
        similarity_type=similarity_type,
        locale=locale,
        model_family=model_family,
        num_languages=num_languages,
        include_target=include_target,
    )
    if existing:
        results_file = os.path.join(existing, "results.json")
        if os.path.exists(results_file):
            return existing
    return None


def merged_model_exists(
    method: str,
    similarity_type: str,
    locale: str,
    model_family: str,
    merged_models_dir: str,
    include_target: Optional[bool] = None,
) -> Optional[str]:
    """Return the path to an existing merged model directory if available."""
    base_dir = os.path.join(REPO_ROOT, merged_models_dir)
    return naming_manager.find_merged_model_directory(
        base_dir,
        method=method,
        similarity_type=similarity_type,
        locale=locale,
        model_family=model_family,
        include_target=include_target,
    )


def run_evaluation(model_path, locale, method_or_prefix, results_dir_name=None):
    """Run evaluation for a specific model using the consolidated model path."""
    print(f"Running evaluation for {model_path} with locale {locale}...")

    # Use custom results directory name if provided, otherwise generate one
    if results_dir_name is None:
        # Legacy mode - generate directory name from prefix
        num_languages = detect_num_languages_from_model_path(model_path)
        if num_languages is not None:
            enhanced_prefix = f"{method_or_prefix}_{num_languages}lang"
        else:
            enhanced_prefix = method_or_prefix
        results_dir_name = enhanced_prefix

    cmd = [sys.executable, os.path.join(REPO_ROOT, "merginguriel", "evaluate_specific_model.py"),
           "--base-model", model_path,
           "--locale", locale,
           "--prefix", method_or_prefix,
           "--results-dir", results_dir_name]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Evaluation completed for {locale} ({method_or_prefix})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for {locale} ({method_or_prefix}): {e}")
        return False

def run_experiment_for_locale(
    locale: str,
    modes: List[str],
    merge_extra_args: List[str],
    include_target_modes: List[bool],
    cleanup_after_eval: bool = False,
    models_root: str = "haryos_model",
    similarity_type: str = "URIEL",
    num_languages: int = 5,
    merged_models_dir: str = "merged_models",
    results_dir: str = "results",
    resume: bool = True,
):
    """Run the requested experiment modes for a single locale."""
    print(f"\n{'='*60}")
    print(f"Running experiment for locale: {locale} (models: {models_root}, similarity: {similarity_type})")
    print(f"{'='*60}")

    # Get model path for this locale
    base_model_path = get_model_for_locale(locale, models_root)

    if not base_model_path:
        print(f"Skipping {locale} - no model found")
        return {}

    # Get model family name using centralized naming system
    try:
        model_family = naming_manager.detect_model_family_from_path(base_model_path)
        print(f"✓ Detected model family: {model_family}")
    except Exception as e:
        print(f"❌ Failed to detect model family: {e}")
        return {}

    # Validate required components
    try:
        naming_manager.validate_required_components(
            experiment_type='merging',
            method=modes[0] if modes else 'baseline',
            similarity_type=similarity_type,
            locale=locale,
            model_family=model_family,
            model_path=base_model_path
        )
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        return {}

    results: Dict[str, bool] = {}

    # Baseline evaluation (run once regardless of target inclusion mode)
    if "baseline" in modes:
        existing_baseline = None
        if resume:
            existing_baseline = results_already_exist(
                experiment_type="baseline",
                method="baseline",
                similarity_type=None,
                locale=locale,
                model_family=model_family,
                results_dir=results_dir,
            )
        if existing_baseline:
            print(f"↩️  Skipping baseline for {locale}; found existing results at {existing_baseline}")
            results["baseline"] = True
        else:
            print(f"\n--- Baseline Evaluation for {locale} ---")
            results_dir_name = naming_manager.get_results_dir_name(
                experiment_type='baseline',
                method='baseline',
                similarity_type=None,
                locale=locale,
                model_family=model_family
            )
            results_full_path = os.path.join(REPO_ROOT, results_dir, results_dir_name)
            os.makedirs(results_full_path, exist_ok=True)
            success = run_evaluation(base_model_path, locale, "baseline", results_full_path)
            results['baseline'] = success

    merge_modes = [mode for mode in modes if mode != "baseline"]

    for include_target in include_target_modes:
        variant_label = "IncTar" if include_target else "ExcTar"
        print(f"\n### Running target inclusion variant: {variant_label} ###")

        for mode in merge_modes:
            print(f"\n--- {mode} Merge for {locale} ({variant_label}) ---")
            # Extract base model name using model-agnostic detection
            base_model_name = None
            if base_model_path:
                try:
                    base_model_name = naming_manager.extract_model_family(base_model_path)
                except ValueError:
                    pass

            try:
                lookup_model_family = naming_manager.extract_model_family(model_family)
            except ValueError:
                lookup_model_family = model_family

            result_key = f"{mode}_{variant_label}"
            variant_merge_args = list(merge_extra_args)
            if include_target:
                variant_merge_args.append("--include-target")

            existing_results = None
            if resume:
                existing_results = results_already_exist(
                    experiment_type="merging",
                    method=mode,
                    similarity_type=similarity_type,
                    locale=locale,
                    model_family=model_family,
                    results_dir=results_dir,
                    num_languages=num_languages,
                    include_target=include_target,
                )
            if existing_results:
                print(f"↩️  Skipping {mode} ({variant_label}) for {locale}; results already at {existing_results}")
                results[result_key] = True
                continue

            existing_merge = None
            if resume:
                existing_merge = merged_model_exists(
                    method=mode,
                    similarity_type=similarity_type,
                    locale=locale,
                    model_family=lookup_model_family,
                    merged_models_dir=merged_models_dir,
                    include_target=include_target,
                )
                if existing_merge:
                    print(f"↩️  Reusing existing merged model at {existing_merge}")

            merge_success = True if existing_merge else run_merge(mode, locale, variant_merge_args, base_model_name, merged_models_dir)

            if merge_success:
                merged_models_dir_full = os.path.join(REPO_ROOT, merged_models_dir)
                merged_model_path = existing_merge or naming_manager.find_merged_model_directory(
                    merged_models_dir_full,
                    method=mode,
                    similarity_type=similarity_type,
                    locale=locale,
                    model_family=lookup_model_family,
                    num_languages=num_languages,
                    include_target=include_target
                )

                if merged_model_path and os.path.exists(merged_model_path):
                    results_dir_name = naming_manager.get_results_dir_name(
                        experiment_type='merging',
                        method=mode,
                        similarity_type=similarity_type,
                        locale=locale,
                        model_family=model_family,
                        num_languages=num_languages,
                        include_target=include_target
                    )
                    results_full_path = os.path.join(REPO_ROOT, results_dir, results_dir_name)
                    os.makedirs(results_full_path, exist_ok=True)

                    eval_success = run_evaluation(merged_model_path, locale, mode, results_full_path)
                    results[result_key] = eval_success

                    if cleanup_after_eval and eval_success and not existing_merge:
                        # Preserve merge_details before cleanup so aggregation/plots keep source info
                        merge_details_file = os.path.join(merged_model_path, "merge_details.txt")
                        if os.path.exists(merge_details_file):
                            dest_details = os.path.join(results_full_path, "merge_details.txt")
                            shutil.copyfile(merge_details_file, dest_details)
                            print(f"📝 Saved merge details to {dest_details}")
                        print(f"🗑️  Cleaning up merged model: {merged_model_path}")
                        shutil.rmtree(merged_model_path)
                        print(f"✅ Successfully deleted {merged_model_path}")
                else:
                    print(f"❌ Could not find merged model directory for {mode} merge of {locale} ({variant_label})")
                    results[result_key] = False
            else:
                results[result_key] = False

    return results

def create_config_from_args(args) -> MergingExperimentConfig:
    """Create a MergingExperimentConfig from parsed CLI arguments."""
    from merginguriel.config import (
        SimilarityConfig,
        TargetConfig,
        ModelConfig,
        DatasetConfig,
        FisherConfig,
        OutputConfig,
    )

    # Determine target inclusion
    inclusion = args.target_inclusion or "ExcTar"
    if inclusion in ("include", "IncTar"):
        inclusion = "IncTar"
    elif inclusion in ("exclude", "ExcTar"):
        inclusion = "ExcTar"

    return MergingExperimentConfig(
        locales=args.locales,
        modes=args.modes,
        similarity=SimilarityConfig(
            type=args.similarity_type,
            source=args.similarity_source,
            top_k=args.top_k,
            sinkhorn_iters=args.sinkhorn_iters,
        ),
        target=TargetConfig(
            locale="",  # Will be set per-locale in run_experiment_for_locale
            inclusion=inclusion,
        ),
        model=ModelConfig(
            base_model="xlm-roberta-base",  # Detected from models_root
            models_root=args.models_root,
            num_languages=args.num_languages,
        ),
        dataset=DatasetConfig(
            name=args.dataset_name,
            split=args.dataset_split,
            text_column=args.text_column,
        ),
        fisher=FisherConfig(
            data_mode=args.fisher_data_mode,
            preweight=args.preweight,
            num_examples=args.num_fisher_examples,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        ),
        output=OutputConfig(
            results_dir=args.results_dir,
            merged_models_dir=args.merged_models_dir,
            cleanup_after_eval=args.cleanup_after_eval,
        ),
        preset=args.preset,
        resume=args.resume,
        start_from=args.start_from,
        max_locales=args.max_locales,
    )


def main():
    parser = argparse.ArgumentParser(description="Run large-scale merging experiments")

    # Config file argument (new, preferred)
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to YAML config file. When provided, CLI args override config values with deprecation warnings.")

    # Experiment control arguments
    parser.add_argument("--locales", nargs="+", default=None,
                       action=TrackProvidedArgsAction,
                       help="Specific locales to run (default: all locales)")
    parser.add_argument("--modes", nargs="+",
                       default=["baseline", "similarity", "average", "fisher", "ties", "task_arithmetic", "slerp", "regmean"],
                       action=TrackProvidedArgsAction,
                       help="Which modes to run per locale (include 'baseline' to evaluate base model)")
    parser.add_argument("--start-from", type=int, default=0,
                       action=TrackProvidedArgsAction,
                       help="Start from specific locale index (for resuming)")
    parser.add_argument("--max-locales", type=int, default=None,
                       action=TrackProvidedArgsAction,
                       help="Maximum number of locales to process")
    parser.add_argument("--list-locales", action="store_true",
                       help="List all available locales and exit")
    parser.add_argument("--list-modes", action="store_true",
                       help="List available merge method keys and exit")

    # Pass-through options for run_merging_pipeline_refactored.py
    parser.add_argument("--num-languages", type=int, default=5,
                       action=TrackProvidedArgsAction,
                       help="Top-K languages to include in merges that auto-select sources")
    parser.add_argument("--similarity-source", type=str, choices=["sparse","dense"], default="dense",
                       action=TrackProvidedArgsAction,
                       help="Use precomputed sparse CSV or compute dense similarities on-the-fly")
    parser.add_argument("--similarity-type", type=str, choices=["URIEL","REAL"], default="URIEL",
                       action=TrackProvidedArgsAction,
                       help="Type of similarity matrix to use: URIEL (linguistic features) or REAL (empirical evaluation results)")
    parser.add_argument("--target-inclusion", type=str,
                       choices=["IncTar", "ExcTar", "include", "exclude", "both"],
                       default=None,
                       action=TrackProvidedArgsAction,
                       help="Target inclusion mode: IncTar (include target), ExcTar (exclude target), or both (default).")
    parser.add_argument("--include-target", action="store_true", dest="legacy_include_target",
                       help="Deprecated alias for --target-inclusion IncTar.")
    parser.add_argument("--exclude-target", action="store_true", dest="legacy_exclude_target",
                       help="Deprecated alias for --target-inclusion ExcTar.")
    parser.add_argument("--top-k", type=int, default=20,
                       action=TrackProvidedArgsAction,
                       help="Top-K neighbors per language for on-the-fly similarity")
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       action=TrackProvidedArgsAction,
                       help="Sinkhorn iterations for on-the-fly similarity")
    parser.add_argument("--dataset-name", type=str, default="AmazonScience/massive",
                       action=TrackProvidedArgsAction,
                       help="Dataset name for dataset-enabled Fisher")
    parser.add_argument("--dataset-split", type=str, default="train",
                       action=TrackProvidedArgsAction,
                       help="Dataset split for Fisher ('train' or 'validation')")
    parser.add_argument("--text-column", type=str, default="utt",
                       action=TrackProvidedArgsAction,
                       help="Text column to use (MASSIVE uses 'utt')")
    parser.add_argument("--num-fisher-examples", type=int, default=1000,
                       action=TrackProvidedArgsAction,
                       help="Total examples for Fisher computation")
    parser.add_argument("--fisher-data-mode", type=str, choices=["target","sources","both"], default="target",
                       action=TrackProvidedArgsAction,
                       help="Distribution to compute Fisher on")
    parser.add_argument("--preweight", type=str, choices=["equal","uriel"], default="equal",
                       action=TrackProvidedArgsAction,
                       help="Pre-weighting for models before Fisher")
    parser.add_argument("--batch-size", type=int, default=16,
                       action=TrackProvidedArgsAction,
                       help="Batch size for Fisher computation")
    parser.add_argument("--max-seq-length", type=int, default=128,
                       action=TrackProvidedArgsAction,
                       help="Max sequence length for Fisher tokenization")
    parser.add_argument("--preset", type=str, choices=["none", "fairness", "target"], default="none",
                       action=TrackProvidedArgsAction,
                       help="Convenience presets for Fisher config: fairness = sources-only + equal preweights; target = target-only + URIEL preweights")
    parser.add_argument("--cleanup-after-eval", action="store_true",
                       help="Delete merged model files after evaluation to save storage space")
    parser.add_argument("--models-root", type=str, default="haryos_model",
                       action=TrackProvidedArgsAction,
                       help="Root directory containing models (default: haryos_model)")
    parser.add_argument("--results-dir", type=str, default="results",
                       action=TrackProvidedArgsAction,
                       help="Directory for experiment results (default: results)")
    parser.add_argument("--merged-models-dir", type=str, default="merged_models",
                       action=TrackProvidedArgsAction,
                       help="Directory for merged models (default: merged_models)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing merged models and results when present (default: enabled). Use --no-resume to force reruns.")

    args = parser.parse_args()

    # Load config from file or create from CLI args
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config = MergingExperimentConfig.from_yaml(args.config)

        # Extended arg map for large-scale experiment specific args
        extended_arg_map = {
            **MERGING_EXPERIMENT_ARG_MAP,
            "locales": "locales",
            "modes": "modes",
            "start_from": "start_from",
            "max_locales": "max_locales",
            "preset": "preset",
            "resume": "resume",
        }

        # Merge CLI args with deprecation warnings
        config = ConfigLoader.merge_cli_args(config, args, emit_warnings=True, arg_to_config_map=extended_arg_map)
    else:
        # Legacy path: create config from CLI args (no warnings)
        config = create_config_from_args(args)
    
    # Handle legacy target inclusion flags (only relevant when not using config file)
    if args.config is None:
        if args.target_inclusion is None:
            if args.legacy_include_target and args.legacy_exclude_target:
                parser.error("Cannot specify both --include-target and --exclude-target.")
            if args.legacy_include_target:
                config.target.inclusion = "IncTar"
            elif args.legacy_exclude_target:
                config.target.inclusion = "ExcTar"
            else:
                config.target.inclusion = "both"
        else:
            if args.legacy_include_target or args.legacy_exclude_target:
                parser.error("Do not mix --target-inclusion with legacy include/exclude flags.")
            config.target.inclusion = args.target_inclusion

    # Apply preset defaults if requested
    argv = sys.argv[1:]
    if config.preset != "none":
        if config.preset == "fairness":
            if "--fisher-data-mode" not in argv:
                config.fisher.data_mode = "sources"
            if "--preweight" not in argv:
                config.fisher.preweight = "equal"
        elif config.preset == "target":
            if "--fisher-data-mode" not in argv:
                config.fisher.data_mode = "target"
            if "--preweight" not in argv:
                config.fisher.preweight = "uriel"

    all_locales = get_all_locales_from_similarity_matrix(config.similarity.type)

    if args.list_modes:
        if merging_methods_dict:
            print("Available merge method keys:")
            for k in sorted(merging_methods_dict.keys()):
                print(f"  - {k}")
        else:
            print("Could not import merging_methods_dict; ensure submodule path is correct.")
        return

    if args.list_locales:
        print("Available locales:")
        for i, locale in enumerate(all_locales):
            print(f"  {i:2d}. {locale}")
        print(f"\nTotal: {len(all_locales)} locales")
        return

    # Filter locales
    if config.locales:
        locales = [loc for loc in all_locales if loc in config.locales]
    else:
        locales = all_locales

    # Apply start index and max limit
    if config.start_from > 0:
        locales = locales[config.start_from:]
    if config.max_locales:
        locales = locales[:config.max_locales]

    print(f"Starting experiment for {len(locales)} locales")
    print(f"Similarity Type: {config.similarity.type}")
    print(f"Model Root: {config.model.models_root}")
    print(f"Cleanup After Eval: {config.output.cleanup_after_eval}")
    print(f"Resume Using Existing Outputs: {config.resume}")
    print(f"Modes: {config.modes}")
    print(f"Start from index: {config.start_from}")

    # Track overall results
    overall_results = {}

    # Resolve target inclusion modes
    inclusion_map = {
        "IncTar": [True],
        "include": [True],
        "ExcTar": [False],
        "exclude": [False],
        "both": [False, True]
    }
    include_target_modes = inclusion_map.get(config.target.inclusion, [False])

    # Build pass-through args once (for subprocess calls)
    merge_extra_args = [
        "--num-languages", str(config.model.num_languages),
        "--similarity-source", config.similarity.source,
        "--similarity-type", config.similarity.type,
        "--top-k", str(config.similarity.top_k),
        "--sinkhorn-iters", str(config.similarity.sinkhorn_iters),
        "--dataset-name", config.dataset.name,
        "--dataset-split", config.dataset.split,
        "--text-column", config.dataset.text_column,
        "--num-fisher-examples", str(config.fisher.num_examples),
        "--fisher-data-mode", config.fisher.data_mode,
        "--preweight", config.fisher.preweight,
        "--batch-size", str(config.fisher.batch_size),
        "--max-seq-length", str(config.fisher.max_seq_length),
        "--base-model-dir", str(Path(REPO_ROOT) / config.model.models_root),
    ]

    # Remove None values (for conditional args like --include-target)
    merge_extra_args = [arg for arg in merge_extra_args if arg is not None]

    for i, locale in enumerate(locales):
        print(f"\nProcessing locale {i+1}/{len(locales)}: {locale}")
        results = run_experiment_for_locale(
            locale,
            config.modes,
            merge_extra_args,
            include_target_modes,
            config.output.cleanup_after_eval,
            config.model.models_root,
            config.similarity.type,
            config.model.num_languages,
            config.output.merged_models_dir,
            config.output.results_dir,
            config.resume,
        )

        overall_results[locale] = results

        # Save progress after each locale
        results_dir_full = os.path.join(REPO_ROOT, config.output.results_dir)
        os.makedirs(results_dir_full, exist_ok=True)
        progress_file = os.path.join(REPO_ROOT, f"{config.output.results_dir}_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_locales': len(all_locales),
                'processed_locales': i + 1,
                'current_locale': locale,
                'similarity_type': config.similarity.type,
                'models_root': config.model.models_root,
                'cleanup_after_eval': config.output.cleanup_after_eval,
                'results': overall_results
            }, f, indent=2)

        print(f"Progress saved to {progress_file}")

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")

    # Summary
    total_locales = len(overall_results)
    print(f"Total locales processed: {len(overall_results)}")
    # Dynamic summary per mode
    mode_success_counts: Dict[str, int] = {}
    for locale, res in overall_results.items():
        if isinstance(res, dict):  # Ensure res is a dictionary
            for mode, ok in res.items():
                if ok is True:
                    mode_success_counts[mode] = mode_success_counts.get(mode, 0) + 1
    if mode_success_counts:
        print("Successful runs per mode:")
        for mode, cnt in sorted(mode_success_counts.items()):
            print(f"  - {mode}: {cnt}")

    # Save final results with config as dict
    final_results_file = os.path.join(REPO_ROOT, f"{config.output.results_dir}_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'summary': {
                'total_locales': len(overall_results),
                'mode_success_counts': mode_success_counts
            },
            'detailed_results': overall_results
        }, f, indent=2)

    print(f"Final results saved to {final_results_file}")

if __name__ == "__main__":
    main()
