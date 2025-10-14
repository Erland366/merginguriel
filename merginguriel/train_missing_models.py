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


def expected_local_dir(locale: str) -> Path:
    return Path(f"haryos_model/xlm-roberta-base_massive_k_{locale}")


def find_missing_locales(mapping_csv: str, restrict_locales=None) -> list[str]:
    df = pd.read_csv(mapping_csv)
    if 'locale' not in df.columns:
        raise ValueError(f"'{mapping_csv}' must contain a 'locale' column")
    locales = df['locale'].tolist()
    if restrict_locales:
        locales = [l for l in locales if l in restrict_locales]
    missing = []
    for loc in locales:
        out_dir = expected_local_dir(loc)
        if not out_dir.exists():
            missing.append(loc)
    return missing

# python training_bert.py \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 15 \
#     --early_stopping_patience 3 \
#     --per_device_train_batch_size 128 \
#     --save_strategy epoch \
#     --save_total_limit 15 \
#     --torch_compile \
#     --load_best_model_at_end=True \
#     --eval_strategy epoch \
#     --model_name_or_path xlm-roberta-base \
#     --overwrite_output_dir \
#     --do_predict

def build_train_cmd(locale: str, args: argparse.Namespace) -> list[str]:
    out_dir = expected_local_dir(locale)
    script_path = Path(__file__).resolve().parent / 'training_bert.py'
    cmd = [
        os.environ.get('PYTHON', 'python'), str(script_path),
        '--model_name_or_path', args.base_model,
        '--dataset_name', 'AmazonScience/massive',
        '--dataset_config_name', locale,
        '--do_train', '--do_eval', '--do_predict',
        '--per_device_train_batch_size', str(args.train_bs),
        '--per_device_eval_batch_size', str(args.eval_bs),
        '--learning_rate', str(args.lr),
        '--num_train_epochs', str(args.epochs),
        '--eval_strategy', 'epoch',
        '--save_strategy', 'epoch',
        '--save_total_limit', str(args.save_total_limit),
        '--load_best_model_at_end', 'true',
        '--greater_is_better', 'true',
        '--early_stopping_patience', str(args.early_stopping_patience),
        '--overwrite_output_dir',
        '--logging_steps', '50',
        '--output_dir', str(out_dir),
    ]
    if args.fp16:
        cmd += ['--fp16']
    if args.torch_compile:
        cmd += ['--torch_compile']
    if args.wandb:
        cmd += ['--report_to', 'wandb']
    else:
        cmd += ['--report_to', 'none']
    return cmd


def main():
    p = argparse.ArgumentParser(description='Train missing MASSIVE locale models')
    p.add_argument('--mapping-csv', default='model_mapping_unified.csv', help='CSV with a locale column')
    p.add_argument('--base-model', default='xlm-roberta-base', help='Pretrained base model')
    p.add_argument('--locales', nargs='+', default=None, help='Subset of locales to consider')
    p.add_argument('--max', type=int, default=None, help='Max number of missing locales to train')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running')
    p.add_argument('--train-bs', type=int, default=128)
    p.add_argument('--eval-bs', type=int, default=128)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--save-total-limit', type=int, default=15)
    p.add_argument('--early-stopping-patience', type=int, default=3)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--torch-compile', dest='torch_compile', action='store_true')
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()

    missing = find_missing_locales(args.mapping_csv, args.locales)
    if args.max:
        missing = missing[:args.max]

    if not missing:
        print('No missing locales found. All expected models exist in haryos_model.')
        return

    print(f"Found {len(missing)} missing locales:")
    for loc in missing:
        print(f"  - {loc} -> {expected_local_dir(loc)}")

    for loc in missing:
        cmd = build_train_cmd(loc, args)
        print('\n=== Training', loc, '===')
        print('Command:', ' '.join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
