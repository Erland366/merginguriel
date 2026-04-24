#!/usr/bin/env python3
"""
Layer-Swapping merge CLI.

Implements the core Layer-Swapping merge from:
The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs
"""

import argparse
import json
from pathlib import Path

from merginguriel.layer_swapping import LayerSwappingConfig, run_layer_swapping_merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer-Swapping merge.")
    parser.add_argument("--model-a-path", required=True, help="Path to base model (retains non-swapped layers)")
    parser.add_argument("--model-b-path", required=True, help="Path to donor model (provides swapped layers)")
    parser.add_argument("--output-path", required=True, help="Output directory for merged model")
    parser.add_argument(
        "--swap-strategy",
        choices=["top_bottom", "custom"],
        default="top_bottom",
        help="Swap strategy: top_bottom or custom",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of top layers to swap (top_bottom)")
    parser.add_argument("--bottom-k", type=int, default=2, help="Number of bottom layers to swap (top_bottom)")
    parser.add_argument(
        "--custom-layers",
        default="",
        help="Comma-separated list of layer indices to swap (custom)",
    )
    parser.add_argument(
        "--non-layer-source",
        choices=["model_a", "model_b"],
        default="model_a",
        help="Source for non-layer params (embeddings, classifier)",
    )
    parser.add_argument(
        "--tokenizer-source",
        choices=["model_a", "model_b"],
        default="model_a",
        help="Which tokenizer to save",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device or device_map to load models (e.g., cpu, cuda, auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    custom_layers = None
    if args.custom_layers:
        custom_layers = [int(x.strip()) for x in args.custom_layers.split(",") if x.strip() != ""]

    config = LayerSwappingConfig(
        swap_strategy=args.swap_strategy,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        custom_layers=custom_layers,
        non_layer_source=args.non_layer_source,
        tokenizer_source=args.tokenizer_source,
        device=args.device,
    )

    metadata = run_layer_swapping_merge(
        model_a_path=args.model_a_path,
        model_b_path=args.model_b_path,
        output_path=str(output_path),
        config=config,
    )

    metadata_path = output_path / "merge_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Layer-Swapping merge complete. Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

