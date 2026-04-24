#!/usr/bin/env python3
"""
Compute and save pairwise source compatibility matrix.

This script pre-computes compatibility scores between all pairs of locale-specific
models, measuring how well they work together when merged.

Metrics:
- task_vector_cosine: Parameter-space alignment (fast, no data needed)
- cka: Representation-space alignment (requires shared input data)

Usage:
    # Compute Task Vector Cosine matrix (fast, ~30 min)
    python -m merginguriel.compute_compatibility_matrix \
        --metric task_vector_cosine \
        --output-dir nxn_results/compatibility_matrix

    # Compute CKA matrix (slower, ~1-2 hrs)
    python -m merginguriel.compute_compatibility_matrix \
        --metric cka \
        --output-dir nxn_results/compatibility_matrix \
        --cka-samples 500

    # Compute both metrics
    python -m merginguriel.compute_compatibility_matrix \
        --metric both \
        --output-dir nxn_results/compatibility_matrix
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.compatibility import (
    ALL_LOCALES,
    CompatibilityConfig,
    compute_pairwise_tv_cosine_matrix,
    compute_pairwise_cka_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise source compatibility matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metric",
        choices=["task_vector_cosine", "cka", "both"],
        default="task_vector_cosine",
        help="Compatibility metric to compute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(project_root) / "nxn_results" / "compatibility_matrix",
        help="Output directory for matrices",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(project_root) / "haryos_model",
        help="Directory containing locale-specific models",
    )
    parser.add_argument(
        "--pretrained-model",
        default="xlm-roberta-base",
        help="Pretrained model name/path",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=None,
        help="Specific locales to include (default: all 49)",
    )
    parser.add_argument(
        "--cka-samples",
        type=int,
        default=500,
        help="Number of samples for CKA computation",
    )
    parser.add_argument(
        "--cka-layer",
        type=int,
        default=-1,
        help="Layer to extract for CKA (-1 = last layer)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for CKA computation",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to output directory",
    )
    args = parser.parse_args()

    # Resolve locales
    locales = args.locales if args.locales else ALL_LOCALES
    print(f"Computing compatibility for {len(locales)} locales")

    # Create output directory
    if args.no_timestamp:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"compat_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build config
    config = CompatibilityConfig(
        metric=args.metric if args.metric != "both" else "task_vector_cosine",
        pretrained_model_name=args.pretrained_model,
        models_dir=args.models_dir,
        cka_num_samples=args.cka_samples,
        cka_layer=args.cka_layer,
        device=args.device,
    )

    # Determine which metrics to compute
    metrics_to_compute = []
    if args.metric == "both":
        metrics_to_compute = ["task_vector_cosine", "cka"]
    else:
        metrics_to_compute = [args.metric]

    # Compute matrices
    for metric in metrics_to_compute:
        print(f"\n{'=' * 60}")
        print(f"Computing {metric} compatibility matrix")
        print(f"{'=' * 60}\n")

        if metric == "task_vector_cosine":
            matrix = compute_pairwise_tv_cosine_matrix(config, locales, verbose=True)
            matrix_path = output_dir / "task_vector_cosine_matrix.csv"
        else:  # cka
            config.metric = "cka"
            matrix = compute_pairwise_cka_matrix(config, locales, verbose=True)
            matrix_path = output_dir / "cka_matrix.csv"

        # Save matrix
        matrix.to_csv(matrix_path)
        print(f"\nSaved {metric} matrix to {matrix_path}")

        # Print summary statistics
        print(f"\nMatrix statistics:")
        # Get off-diagonal values
        mask = ~(matrix.index.values[:, None] == matrix.columns.values)
        off_diag = matrix.values[mask]
        print(f"  Off-diagonal mean: {off_diag.mean():.4f}")
        print(f"  Off-diagonal std:  {off_diag.std():.4f}")
        print(f"  Off-diagonal min:  {off_diag.min():.4f}")
        print(f"  Off-diagonal max:  {off_diag.max():.4f}")

    # Save computation log
    log = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_to_compute,
        "num_locales": len(locales),
        "locales": locales,
        "models_dir": str(args.models_dir),
        "pretrained_model": args.pretrained_model,
        "cka_samples": args.cka_samples,
        "cka_layer": args.cka_layer,
    }
    log_path = output_dir / "computation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved computation log to {log_path}")

    print(f"\nDone! Matrices saved in {output_dir}")


if __name__ == "__main__":
    main()
