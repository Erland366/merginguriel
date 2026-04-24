#!/usr/bin/env python3
"""
Train missing locale models for MASSIVE using training_bert.py.

Discovers which MASSIVE locale models are missing from ./haryos_model and
trains them with standard hyperparameters. Supports dry-run.
"""

import argparse
import os
import subprocess
from pathlib import Path
import pandas as pd


def expected_local_dir(locale: str, base_model: str = "xlm-roberta-base") -> Path:
    # Convert model name to directory-friendly format
    model_name = base_model.replace("FacebookAI/", "")
    return Path(f"haryos_model/{model_name}_massive_k_{locale}")


def find_missing_locales(mapping_csv: str, restrict_locales=None, base_model: str = "xlm-roberta-base") -> list[str]:
    # Handle both CSV format and simple text format (like available_locales.txt)
    try:
        df = pd.read_csv(mapping_csv)
        if 'locale' not in df.columns:
            # Try to read as simple text file with locales
            with open(mapping_csv, 'r') as f:
                lines = f.readlines()
            locales = []
            for line in lines:
                # Handle numbered format like "1→af-ZA" or simple "af-ZA"
                locale = line.strip()
                if '→' in locale:
                    locale = locale.split('→')[1] if '→' in locale else locale
                # Remove any leading numbers/characters
                locale = ''.join(c for c in locale if c.isalnum() or c == '-')
                if locale and len(locale) > 2:  # Basic validation
                    locales.append(locale)
        else:
            locales = df['locale'].tolist()
    except Exception as e:
        # Fallback: try to read as simple text file
        with open(mapping_csv, 'r') as f:
            lines = f.readlines()
        locales = []
        for line in lines:
            locale = line.strip()
            if '→' in locale:
                locale = locale.split('→')[1] if '→' in locale else locale
            locale = ''.join(c for c in locale if c.isalnum() or c == '-')
            if locale and len(locale) > 2:
                locales.append(locale)

    if restrict_locales:
        locales = [l for l in locales if l in restrict_locales]
    missing = []
    for loc in locales:
        out_dir = expected_local_dir(loc, base_model)
        if not out_dir.exists():
            missing.append(loc)
    return missing

# python training_bert.py \
#     --do_train \
#     --do_eval \
#     --num-train-epochs 15 \
#     --early-stopping-patience 3 \
#     --per-device-train-batch-size 128 \
#     --per-device-eval-batch-size 128 \
#     --save-strategy epoch \
#     --save-total-limit 15 \
#     --torch-compile \
#     --load-best-model-at-end=True \
#     --eval-strategy epoch \
#     --model-name-or-path xlm-roberta-base \
#     --overwrite-output-dir \
#     --do_predict

def build_train_cmd(locale: str, args: argparse.Namespace) -> list[str]:
    out_dir = expected_local_dir(locale, args.base_model)
    script_path = Path(__file__).resolve().parent / "training_bert.py"
    cmd = [
        os.environ.get("PYTHON", "python"), str(script_path),
        "--model_name_or_path", args.base_model,
        "--dataset_name", "AmazonScience/massive",
        "--dataset_config_name", locale,
        "--do_train", "--do_eval", "--do_predict",
        "--per-device-train-batch-size", str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
        "--learning-rate", str(args.lr),
        "--num-train-epochs", str(args.num_train_epochs),
        "--eval-strategy", "epoch",
        "--save-strategy", "epoch",
        "--save-total-limit", str(args.save_total_limit),
        "--load-best-model-at-end", "true",
        "--greater-is-better", "true",
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--overwrite-output-dir",
        "--logging-steps", "50",
        "--output-dir", str(out_dir),
    ]
    if args.warmup_ratio > 0:
        cmd += ["--warmup_ratio", str(args.warmup_ratio)]
    if args.bf16:
        cmd += ["--bf16"]
    if args.torch_compile:
        cmd += ["--torch_compile"]
    if args.wandb:
        cmd += ["--report_to", "wandb"]
    else:
        cmd += ["--report_to", "none"]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Train missing MASSIVE locale models")
    parser.add_argument("--mapping-csv", default="model_mapping_unified.csv", help="CSV with a locale column")
    parser.add_argument("--base-model", default="xlm-roberta-base", help="Pretrained base model (e.g., xlm-roberta-base, FacebookAI/xlm-roberta-large)")
    parser.add_argument("--locales", nargs="+", default=None, help="Subset of locales to consider")
    parser.add_argument("--max", type=int, default=None, help="Max number of missing locales to train")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--per-device-train-batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=128, help="Evaluation batch size")
    # Shorthand aliases for user convenience
    parser.add_argument("--train-bs", dest="per_device_train_batch_size", type=int, default=128, help="Training batch size (shorthand for --per-device-train-batch-size)")
    parser.add_argument("--eval-bs", dest="per_device_eval_batch_size", type=int, default=128, help="Evaluation batch size (shorthand for --per-device-eval-batch-size)")
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save-total-limit", type=int, default=15, help="Number of checkpoints to keep")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 precision (default: enabled)")
    parser.add_argument("--torch-compile", dest="torch_compile", action="store_true", help="Enable torch compilation")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--warmup-ratio", type=float, default=0, help="Warmup ratio for learning rate scheduler")
    args = parser.parse_args()

    missing = find_missing_locales(args.mapping_csv, args.locales, args.base_model)
    if args.max:
        missing = missing[:args.max]

    if not missing:
        print(f'No missing locales found for base model {args.base_model}. All expected models exist in haryos_model.')
        return

    print(f"Found {len(missing)} missing locales for {args.base_model}:")
    for loc in missing:
        print(f"  - {loc} -> {expected_local_dir(loc, args.base_model)}")

    for loc in missing:
        cmd = build_train_cmd(loc, args)
        print('\n=== Training', loc, '===')
        print('Command:', ' '.join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
