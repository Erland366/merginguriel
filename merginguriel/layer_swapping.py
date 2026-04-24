"""
Layer-swapping merge utilities (Bandarkar & Peng, 2025).

This module implements a simple layer-swapping merge:
 - Start from a base model (model_a)
 - Replace selected encoder layers with weights from a donor model (model_b)

Terminology:
 - model_a (base): retains non-swapped layers and optionally embeddings/classifier
 - model_b (donor): provides the swapped layers
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from merginguriel.selective_layer.layer_masking import identify_layer_from_param_name


@dataclass
class LayerSwappingConfig:
    swap_strategy: str = "top_bottom"  # "top_bottom" or "custom"
    top_k: int = 2
    bottom_k: int = 2
    custom_layers: list[int] | None = None
    non_layer_source: str = "model_a"  # "model_a" or "model_b"
    tokenizer_source: str = "model_a"  # "model_a" or "model_b"
    device: str = "cpu"

    def validate(self, num_layers: int) -> None:
        if self.swap_strategy not in {"top_bottom", "custom"}:
            raise ValueError(f"swap_strategy must be 'top_bottom' or 'custom', got {self.swap_strategy}")
        _VALID_SOURCES = {"model_a", "model_b", "math", "language"}
        if self.non_layer_source not in _VALID_SOURCES:
            raise ValueError(f"non_layer_source must be one of {_VALID_SOURCES}, got {self.non_layer_source}")
        if self.tokenizer_source not in _VALID_SOURCES:
            raise ValueError(f"tokenizer_source must be one of {_VALID_SOURCES}, got {self.tokenizer_source}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if self.swap_strategy == "top_bottom":
            if self.top_k < 0 or self.bottom_k < 0:
                raise ValueError("top_k and bottom_k must be >= 0")
            if self.top_k + self.bottom_k > num_layers:
                raise ValueError(
                    f"top_k + bottom_k exceeds num_layers ({self.top_k}+{self.bottom_k}>{num_layers})"
                )
        if self.swap_strategy == "custom":
            if not self.custom_layers:
                raise ValueError("custom_layers must be provided for swap_strategy='custom'")
            for layer_id in self.custom_layers:
                if not 0 <= layer_id < num_layers:
                    raise ValueError(f"Layer index {layer_id} out of range [0, {num_layers})")


def get_swap_indices(
    num_layers: int,
    top_k: int,
    bottom_k: int,
    custom_layers: Iterable[int] | None = None,
    swap_strategy: str = "top_bottom",
) -> list[int]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if swap_strategy == "custom":
        if not custom_layers:
            raise ValueError("custom_layers must be provided for swap_strategy='custom'")
        indices = sorted(set(int(i) for i in custom_layers))
        if any(i < 0 or i >= num_layers for i in indices):
            raise ValueError(f"custom_layers contain out-of-range indices for num_layers={num_layers}")
        return indices
    if top_k < 0 or bottom_k < 0:
        raise ValueError("top_k and bottom_k must be >= 0")
    if top_k + bottom_k > num_layers:
        raise ValueError("top_k + bottom_k exceeds num_layers")
    bottom = list(range(0, bottom_k))
    top = list(range(num_layers - top_k, num_layers)) if top_k > 0 else []
    return sorted(set(bottom + top))


def _normalize_source(source: str) -> str:
    """Normalize legacy 'math'/'language' to 'model_a'/'model_b'."""
    return {"math": "model_a", "language": "model_b"}.get(source, source)


def _validate_state_dict_compatibility(
    model_a_state: dict[str, torch.Tensor],
    model_b_state: dict[str, torch.Tensor],
) -> None:
    if model_a_state.keys() != model_b_state.keys():
        missing_in_b = model_a_state.keys() - model_b_state.keys()
        missing_in_a = model_b_state.keys() - model_a_state.keys()
        raise ValueError(
            "State dict keys do not match between model_a and model_b. "
            f"Missing in model_b: {sorted(list(missing_in_b))[:5]} "
            f"Missing in model_a: {sorted(list(missing_in_a))[:5]}"
        )


def apply_layer_swapping_state_dicts(
    model_a_state: dict[str, torch.Tensor],
    model_b_state: dict[str, torch.Tensor],
    swap_layer_indices: Iterable[int],
    non_layer_source: str = "model_a",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Swap selected layers from model_b into model_a.

    Args:
        model_a_state: Base model state dict (retains non-swapped layers).
        model_b_state: Donor model state dict (provides swapped layers).
        swap_layer_indices: Layer indices to take from model_b.
        non_layer_source: Which model provides non-layer params (embeddings,
            classifier). Accepts "model_a", "model_b" or legacy "math", "language".
    """
    non_layer_source = _normalize_source(non_layer_source)
    if non_layer_source not in {"model_a", "model_b"}:
        raise ValueError(f"non_layer_source must be 'model_a' or 'model_b', got {non_layer_source}")

    _validate_state_dict_compatibility(model_a_state, model_b_state)

    swap_set = set(swap_layer_indices)
    merged_state: dict[str, torch.Tensor] = {}
    swapped_params = 0
    non_layer_params = 0

    for name, a_value in model_a_state.items():
        layer_id = identify_layer_from_param_name(name)
        if layer_id >= 0 and layer_id in swap_set:
            merged_state[name] = model_b_state[name].clone()
            swapped_params += 1
        elif layer_id < 0 and non_layer_source == "model_b":
            merged_state[name] = model_b_state[name].clone()
            non_layer_params += 1
        else:
            merged_state[name] = a_value.clone()

    metadata = {
        "swapped_layers": sorted(list(swap_set)),
        "swapped_param_count": swapped_params,
        "non_layer_params_from_model_b": non_layer_params,
    }
    return merged_state, metadata


def _validate_model_compatibility(model_a, model_b) -> None:
    a_cfg = model_a.config
    b_cfg = model_b.config

    if a_cfg.model_type != b_cfg.model_type:
        raise ValueError(f"Model types differ: {a_cfg.model_type} vs {b_cfg.model_type}")
    if a_cfg.num_hidden_layers != b_cfg.num_hidden_layers:
        raise ValueError(
            f"num_hidden_layers differ: {a_cfg.num_hidden_layers} vs {b_cfg.num_hidden_layers}"
        )
    if getattr(a_cfg, "hidden_size", None) != getattr(b_cfg, "hidden_size", None):
        raise ValueError(
            f"hidden_size differ: {getattr(a_cfg, 'hidden_size', None)} "
            f"vs {getattr(b_cfg, 'hidden_size', None)}"
        )


def run_layer_swapping_merge(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    config: LayerSwappingConfig,
) -> dict[str, Any]:
    """Run a layer-swapping merge between two models.

    Args:
        model_a_path: Path to base model (retains non-swapped layers).
        model_b_path: Path to donor model (provides swapped layers).
        output_path: Directory to save the merged model.
        config: Layer-swapping configuration.
    """
    model_a = AutoModelForSequenceClassification.from_pretrained(
        model_a_path,
        device_map=config.device,
    )
    model_b = AutoModelForSequenceClassification.from_pretrained(
        model_b_path,
        device_map=config.device,
    )

    _validate_model_compatibility(model_a, model_b)

    num_layers = model_a.config.num_hidden_layers
    config.validate(num_layers)

    swap_indices = get_swap_indices(
        num_layers=num_layers,
        top_k=config.top_k,
        bottom_k=config.bottom_k,
        custom_layers=config.custom_layers,
        swap_strategy=config.swap_strategy,
    )

    model_a_state = model_a.state_dict()
    model_b_state = model_b.state_dict()

    merged_state, metadata = apply_layer_swapping_state_dicts(
        model_a_state=model_a_state,
        model_b_state=model_b_state,
        swap_layer_indices=swap_indices,
        non_layer_source=config.non_layer_source,
    )

    model_a.load_state_dict(merged_state)

    tok_source = _normalize_source(config.tokenizer_source)
    tokenizer_path = model_b_path if tok_source == "model_b" else model_a_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_a.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    merge_metadata = {
        "model_a_path": model_a_path,
        "model_b_path": model_b_path,
        "output_path": output_path,
        "swap_strategy": config.swap_strategy,
        "swap_indices": swap_indices,
        "non_layer_source": config.non_layer_source,
        "tokenizer_source": config.tokenizer_source,
        "model_type": model_a.config.model_type,
        "num_hidden_layers": num_layers,
        **metadata,
    }

    return merge_metadata

