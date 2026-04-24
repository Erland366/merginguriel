"""
LoRA Adapter Merging Experiment Runner

Converts LoRA adapters to full models (merge_and_unload), then merges them
using TIES or task_arithmetic via the existing auto_merge_llm pipeline.
Evaluates merged models on MASSIVE intent classification.

Usage:
    # Single experiment
    python -m merginguriel.run_lora_merging_experiment \
        --target sw-KE --method ties --similarity_type REAL --num_sources 2

    # Batch: run all planned experiments
    python -m merginguriel.run_lora_merging_experiment --run_batch E1
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "submodules" / "auto_merge_llm"))

from merginguriel.similarity_utils import load_and_process_similarity
from auto_merge_llm.methods import merging_methods_dict

# Paths
LORA_MODEL_DIR = PROJECT_ROOT / "haryos_model_loras"
NXN_MATRIX_PATH = Path("/home/coder/Python_project/MergingUriel_Root/MergingUriel/nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv")
NXN_VALIDATION_PATH = PROJECT_ROOT / "nxn_results" / "nxn_eval_validation" / "evaluation_matrix.csv"
URIEL_MATRIX_PATH = PROJECT_ROOT / "language_similarity_matrix_unified.csv"
BASE_MODEL_NAME = "FacebookAI/xlm-roberta-base"
NUM_LABELS = 60

# Experiment targets
TARGETS = ["sw-KE", "ar-SA", "jv-ID", "sq-AL", "vi-VN"]

# Experiment definitions
EXPERIMENTS = {
    "E1": {"method": "ties", "similarity_type": "REAL", "num_sources": 2, "description": "LoRA TIES 2-model REAL"},
    "E2": {"method": "ties", "similarity_type": "REAL", "num_sources": 5, "description": "LoRA TIES 5-model REAL"},
    "E3": {"method": "ties", "similarity_type": "URIEL", "num_sources": 2, "description": "LoRA TIES 2-model URIEL"},
    "E4": {"method": "ties", "similarity_type": "URIEL", "num_sources": 5, "description": "LoRA TIES 5-model URIEL"},
    "E5": {"method": "task_arithmetic", "similarity_type": "REAL", "num_sources": 2, "description": "LoRA task_arithmetic 2-model REAL"},
}


@dataclass
class ExperimentResult:
    experiment_id: str
    target: str
    method: str
    similarity_type: str
    num_sources: int
    source_locales: List[str]
    source_weights: List[float]
    merged_accuracy: float
    best_zs_accuracy: float
    best_zs_source: str
    delta_vs_best_zs: float
    eval_time_s: float


def lora_adapter_path(locale: str) -> Path:
    return LORA_MODEL_DIR / f"xlm-roberta-base_massive_lora_{locale}"


def convert_lora_to_full_model(locale: str, output_dir: str) -> str:
    """Load a LoRA adapter, merge into base model, save as full model."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    adapter_path = lora_adapter_path(locale)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    print(f"  Converting {locale} LoRA → full model...")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    save_path = os.path.join(output_dir, f"lora_full_{locale}")
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(save_path)

    # Copy id2label/label2id from adapter config if available
    adapter_config_path = adapter_path / "config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        # The merged model should already have correct config from base + PEFT
        # but let's ensure id2label is set
        model_config_path = os.path.join(save_path, "config.json")
        with open(model_config_path) as f:
            model_cfg = json.load(f)
        if "id2label" not in model_cfg or len(model_cfg.get("id2label", {})) != NUM_LABELS:
            # Load from LoRA training's original model config
            orig_config_path = adapter_path / "adapter_config.json"
            if orig_config_path.exists():
                print(f"    Note: id2label may need mapping from training data")

    print(f"    Saved to {save_path}")
    return save_path


def create_pretrained_base(output_dir: str) -> str:
    """Create pretrained base model with correct classifier head (60 classes)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("  Creating pretrained base model (60-class head)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    save_path = os.path.join(output_dir, "pretrained_base")
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(save_path)
    print(f"    Saved to {save_path}")
    return save_path


def get_source_locales(target: str, similarity_type: str, num_sources: int) -> List[Tuple[str, float]]:
    """Get source locales and weights using REAL or URIEL similarity."""
    if similarity_type == "REAL":
        matrix_path = str(NXN_MATRIX_PATH)
    elif similarity_type == "URIEL":
        matrix_path = str(URIEL_MATRIX_PATH)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    similar_languages = load_and_process_similarity(
        matrix_path, target, num_sources,
        top_k=20, sinkhorn_iterations=20,
        include_target=False, verbose=True
    )
    return similar_languages


def get_best_zs_for_target(target: str, nxn_path: Path | None = None) -> Tuple[float, str]:
    """Get best zero-shot accuracy and source for a target from NxN matrix.

    Args:
        target: Target locale code.
        nxn_path: Path to NxN evaluation matrix CSV. Defaults to NXN_MATRIX_PATH (test).
    """
    path = nxn_path or NXN_MATRIX_PATH
    nxn = pd.read_csv(str(path), index_col=0)
    col = nxn[target].drop(target)
    best_acc = col.max()
    best_source = col.idxmax()
    return best_acc, best_source


def evaluate_merged_model_massive(model_path: str, target_locale: str, split: str = "test") -> float:
    """Evaluate a merged model on MASSIVE. Returns accuracy.

    Args:
        model_path: Path to the merged model directory.
        target_locale: MASSIVE locale code (e.g. "sw-KE").
        split: Dataset split to evaluate on ("test" or "validation").
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset

    print(f"\n  Evaluating on MASSIVE {split} ({target_locale})...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_dataset("AmazonScience/massive", target_locale, split=split, trust_remote_code=True)

    correct = 0
    total = 0
    batch_size = 64

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        inputs = tokenizer(
            batch["utt"], padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        labels = batch["intent"]

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

        correct += sum(p == l for p, l in zip(preds, labels))
        total += len(labels)

    accuracy = correct / total
    print(f"    Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def run_single_experiment(
    target: str,
    method: str,
    similarity_type: str,
    num_sources: int,
    experiment_id: str = "",
) -> ExperimentResult:
    """Run a single LoRA merging experiment."""
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_id}")
    print(f"Target: {target} | Method: {method} | Sim: {similarity_type} | Sources: {num_sources}")
    print(f"{'='*70}")

    start_time = time.time()

    # Step 1: Get source locales
    print("\n[1/5] Selecting source locales...")
    sources_and_weights = get_source_locales(target, similarity_type, num_sources)
    source_locales = [loc for loc, _ in sources_and_weights]
    source_weights = [w for _, w in sources_and_weights]
    print(f"  Sources: {list(zip(source_locales, [f'{w:.4f}' for w in source_weights]))}")

    # Step 2: Get baseline (best zero-shot)
    print("\n[2/5] Getting baseline...")
    best_zs_acc, best_zs_source = get_best_zs_for_target(target)
    print(f"  Best ZS: {best_zs_acc:.4f} ({best_zs_source})")

    # Step 3: Convert LoRA adapters to full models
    print("\n[3/5] Converting LoRA adapters to full models...")
    temp_dir = tempfile.mkdtemp(prefix=f"lora_merge_{target}_{method}_{similarity_type}_")
    print(f"  Temp dir: {temp_dir}")

    try:
        # Convert all source adapters
        source_model_paths = []
        for locale in source_locales:
            path = convert_lora_to_full_model(locale, temp_dir)
            source_model_paths.append(path)

        # Create pretrained base for task vector computation
        pretrained_base_path = create_pretrained_base(temp_dir)

        # Step 4: Merge models
        print(f"\n[4/5] Merging with {method}...")
        merger = merging_methods_dict[method]()

        if method == "ties":
            method_params = {
                "scaling_coefficient": 1.0,
                "param_value_mask_rate": 0.2,
                "dare_enabled": False,
                "dare_drop_rate": 0.0,
                "dare_rescale": True,
                "dare_seed": None,
            }
        elif method == "task_arithmetic":
            method_params = {
                "scaling_coefficient": 1.0,
            }
        elif method == "breadcrumbs":
            method_params = {
                "scaling_coefficient": 1.0,
                "param_value_mask_rate": 0.2,
                "param_density": 0.9,
            }
        elif method == "stock":
            method_params = {}
        elif method == "linear":
            method_params = {
                "scaling_coefficient": 1.0,
            }
        else:
            raise ValueError(f"Unsupported method: {method}")

        result = merger.merge(
            base_model=pretrained_base_path,
            models_to_merge=source_model_paths,
            method_params=method_params,
            exclude_param_names_regex=[],
        )

        merged_model = result["merged_model"]
        base_tokenizer = result["base_tokenizer"]

        # Save merged model for evaluation
        merged_model_path = os.path.join(temp_dir, "merged_model")
        merged_model.save_pretrained(merged_model_path)
        base_tokenizer.save_pretrained(merged_model_path)
        print(f"  Merged model saved to {merged_model_path}")

        # Step 5: Evaluate
        print("\n[5/5] Evaluating merged model...")
        merged_accuracy = evaluate_merged_model_massive(merged_model_path, target)

        eval_time = time.time() - start_time
        delta = merged_accuracy - best_zs_acc

        print(f"\n{'─'*50}")
        print(f"  Result: {merged_accuracy:.4f} (delta vs best ZS: {delta:+.4f})")
        print(f"  Best ZS baseline: {best_zs_acc:.4f} ({best_zs_source})")
        print(f"  Time: {eval_time:.1f}s")
        print(f"{'─'*50}")

        return ExperimentResult(
            experiment_id=experiment_id,
            target=target,
            method=method,
            similarity_type=similarity_type,
            num_sources=num_sources,
            source_locales=source_locales,
            source_weights=source_weights,
            merged_accuracy=merged_accuracy,
            best_zs_accuracy=best_zs_acc,
            best_zs_source=best_zs_source,
            delta_vs_best_zs=delta,
            eval_time_s=eval_time,
        )

    finally:
        print(f"\n  Cleaning up temp dir: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batch(experiment_id: str, targets: Optional[List[str]] = None) -> List[ExperimentResult]:
    """Run a batch of experiments for a given experiment ID (E1-E5)."""
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_id}. Available: {list(EXPERIMENTS.keys())}")

    exp = EXPERIMENTS[experiment_id]
    targets = targets or TARGETS
    results = []

    print(f"\n{'#'*70}")
    print(f"# Batch: {experiment_id} - {exp['description']}")
    print(f"# Targets: {targets}")
    print(f"{'#'*70}")

    for target in targets:
        result = run_single_experiment(
            target=target,
            method=exp["method"],
            similarity_type=exp["similarity_type"],
            num_sources=exp["num_sources"],
            experiment_id=f"{experiment_id}_{target}",
        )
        results.append(result)

    return results


def print_results_table(results: List[ExperimentResult]):
    """Print a summary table of all experiment results."""
    print(f"\n{'='*90}")
    print(f"{'Experiment':<15} {'Target':<8} {'Method':<16} {'Sim':<6} {'#Src':<5} "
          f"{'Merged':<8} {'BestZS':<8} {'Delta':<8} {'Sources'}")
    print(f"{'─'*90}")

    for r in results:
        delta_str = f"{r.delta_vs_best_zs:+.4f}"
        marker = "+" if r.delta_vs_best_zs > 0 else ""
        sources_str = ", ".join(r.source_locales)
        print(f"{r.experiment_id:<15} {r.target:<8} {r.method:<16} {r.similarity_type:<6} "
              f"{r.num_sources:<5} {r.merged_accuracy:.4f}  {r.best_zs_accuracy:.4f}  "
              f"{delta_str:<8} {sources_str}")

    print(f"{'='*90}")

    # Summary stats
    positive = sum(1 for r in results if r.delta_vs_best_zs > 0)
    mean_delta = sum(r.delta_vs_best_zs for r in results) / len(results) if results else 0
    print(f"\nPositive: {positive}/{len(results)} | Mean delta: {mean_delta:+.4f}")


def save_results(results: List[ExperimentResult], output_path: str):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            "experiment_id": r.experiment_id,
            "target": r.target,
            "method": r.method,
            "similarity_type": r.similarity_type,
            "num_sources": r.num_sources,
            "source_locales": r.source_locales,
            "source_weights": r.source_weights,
            "merged_accuracy": r.merged_accuracy,
            "best_zs_accuracy": r.best_zs_accuracy,
            "best_zs_source": r.best_zs_source,
            "delta_vs_best_zs": r.delta_vs_best_zs,
            "eval_time_s": r.eval_time_s,
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Merging Experiments")

    subparsers = parser.add_subparsers(dest="command")

    # Single experiment
    single = subparsers.add_parser("single", help="Run a single experiment")
    single.add_argument("--target", required=True, help="Target locale (e.g., sw-KE)")
    single.add_argument("--method", default="ties", choices=["ties", "task_arithmetic"])
    single.add_argument("--similarity_type", default="REAL", choices=["REAL", "URIEL"])
    single.add_argument("--num_sources", type=int, default=2)

    # Batch experiment
    batch = subparsers.add_parser("batch", help="Run a batch experiment (E1-E5)")
    batch.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    batch.add_argument("--targets", nargs="+", default=None, help="Override target list")

    # Run all planned experiments in order
    run_all = subparsers.add_parser("all", help="Run all experiments E1-E5 in order")

    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "lora_merging_results"
    results_dir.mkdir(exist_ok=True)

    if args.command == "single":
        result = run_single_experiment(
            target=args.target,
            method=args.method,
            similarity_type=args.similarity_type,
            num_sources=args.num_sources,
            experiment_id=f"single_{args.target}_{args.method}_{args.similarity_type}_{args.num_sources}",
        )
        print_results_table([result])
        save_results([result], str(results_dir / f"single_{args.target}_{args.method}_{args.similarity_type}.json"))

    elif args.command == "batch":
        results = run_batch(args.experiment, args.targets)
        print_results_table(results)
        save_results(results, str(results_dir / f"{args.experiment}_results.json"))

    elif args.command == "all":
        all_results = []
        for exp_id in ["E1", "E3", "E5", "E2", "E4"]:
            results = run_batch(exp_id)
            all_results.extend(results)
            save_results(results, str(results_dir / f"{exp_id}_results.json"))

        print("\n\n" + "=" * 90)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 90)
        print_results_table(all_results)
        save_results(all_results, str(results_dir / "all_results.json"))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
