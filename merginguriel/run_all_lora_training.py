#!/usr/bin/env python
"""Run LoRA training for all 49 MASSIVE locales across 2 GPUs.

Splits locales into 2 queues (one per GPU), runs sequentially within each
queue but 2 locales in parallel (one per GPU). Skips already-completed locales.

Usage:
    source .venv/bin/activate
    python -m merginguriel.run_all_lora_training

    # Override defaults:
    python -m merginguriel.run_all_lora_training --lora_rank 8 --num_train_epochs 3
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ALL_MASSIVE_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "hi-IN",
    "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE",
    "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM",
    "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL",
    "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN",
    "zh-TW",
]


def is_locale_complete(output_dir: str) -> bool:
    """Check if a locale has already been trained (adapter weights exist)."""
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    adapter_path_bin = os.path.join(output_dir, "adapter_model.bin")
    return os.path.exists(adapter_path) or os.path.exists(adapter_path_bin)


def _model_prefix(model_name: str) -> str:
    """Extract a short prefix from model name for directory naming."""
    # "FacebookAI/xlm-roberta-base" -> "xlm-roberta-base"
    return model_name.split("/")[-1]


def train_locale(locale: str, gpu_id: int, args: argparse.Namespace) -> dict:
    """Train a single locale on a specific GPU. Returns result dict."""
    prefix = _model_prefix(args.model_name_or_path)
    output_dir = os.path.join(
        args.output_base_dir,
        f"{prefix}_massive_lora_{locale}",
    )

    if is_locale_complete(output_dir):
        return {"locale": locale, "gpu": gpu_id, "status": "skipped", "message": "Already complete"}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-m", "merginguriel.training_lora",
        "--locale", locale,
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", output_dir,
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_target_modules", args.lora_target_modules,
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", str(args.eval_batch_size),
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
        "--load_best_model_at_end", "True",
        "--metric_for_best_model", "eval_accuracy",
        "--logging_steps", "50",
        "--report_to", "wandb",
        "--bf16", str(args.bf16),
        "--do_train",
        "--do_eval",
    ]

    if args.push_to_hub:
        cmd.extend(["--push_adapter_to_hub"])

    if args.wandb_tags:
        cmd.extend(["--wandb_tags", args.wandb_tags])

    start_time = time.time()
    print(f"[GPU {gpu_id}] Starting {locale} -> {output_dir}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start_time
    status = "success" if result.returncode == 0 else "failed"

    if result.returncode != 0:
        print(f"[GPU {gpu_id}] FAILED {locale} after {elapsed:.1f}s")
        print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
    else:
        print(f"[GPU {gpu_id}] Done {locale} in {elapsed:.1f}s")

    return {
        "locale": locale,
        "gpu": gpu_id,
        "status": status,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-500:] if result.returncode != 0 else "",
    }


def run_gpu_queue(locales: list, gpu_id: int, args: argparse.Namespace) -> list:
    """Run all locales in a queue sequentially on one GPU."""
    results = []
    for locale in locales:
        result = train_locale(locale, gpu_id, args)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LoRA training for all MASSIVE locales.")
    parser.add_argument("--model_name_or_path", type=str, default="FacebookAI/xlm-roberta-base",
                        help="Pretrained model (e.g., FacebookAI/xlm-roberta-large).")
    parser.add_argument("--output_base_dir", type=str, default="haryos_model_loras",
                        help="Base directory for output models.")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs to use.")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--wandb_tags", type=str, default="lora,massive,xlm-roberta-base")
    parser.add_argument("--locales", type=str, default=None,
                        help="Comma-separated subset of locales. Default: all 49.")
    args = parser.parse_args()

    locales = ALL_MASSIVE_LOCALES
    if args.locales:
        locales = [l.strip() for l in args.locales.split(",")]
        for l in locales:
            assert l in ALL_MASSIVE_LOCALES, f"Unknown locale: {l}"

    os.makedirs(args.output_base_dir, exist_ok=True)

    # Check which are already done
    prefix = _model_prefix(args.model_name_or_path)
    pending = []
    skipped = []
    for locale in locales:
        output_dir = os.path.join(args.output_base_dir, f"{prefix}_massive_lora_{locale}")
        if is_locale_complete(output_dir):
            skipped.append(locale)
        else:
            pending.append(locale)

    print(f"Total locales: {len(locales)}")
    print(f"Already complete: {len(skipped)} ({', '.join(skipped) if skipped else 'none'})")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All locales already trained. Nothing to do.")
        return

    # Split into GPU queues
    num_gpus = min(args.num_gpus, len(pending))
    gpu_queues = [[] for _ in range(num_gpus)]
    for i, locale in enumerate(pending):
        gpu_queues[i % num_gpus].append(locale)

    for gpu_id, queue in enumerate(gpu_queues):
        print(f"GPU {gpu_id} queue ({len(queue)}): {', '.join(queue)}")

    # Run queues in parallel (one process per GPU)
    all_results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(run_gpu_queue, queue, gpu_id, args): gpu_id
            for gpu_id, queue in enumerate(gpu_queues)
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            results = future.result()
            all_results.extend(results)

    total_elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    succeeded = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]
    skipped_results = [r for r in all_results if r["status"] == "skipped"]

    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped_results) + len(skipped)}")
    print(f"Total time: {total_elapsed / 60:.1f} minutes")

    if failed:
        print("\nFailed locales:")
        for r in failed:
            print(f"  {r['locale']} (GPU {r['gpu']}): {r['stderr_tail'][:200]}")

    # Write results to file
    results_path = os.path.join(args.output_base_dir, "training_summary.txt")
    with open(results_path, "w") as f:
        f.write(f"LoRA Training Summary\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_elapsed / 60:.1f} minutes\n")
        f.write(f"Succeeded: {len(succeeded)}, Failed: {len(failed)}, Skipped: {len(skipped_results) + len(skipped)}\n\n")
        for r in all_results:
            elapsed_str = f"{r.get('elapsed', 0):.1f}s" if "elapsed" in r else "N/A"
            f.write(f"{r['locale']}: {r['status']} ({elapsed_str})\n")
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
