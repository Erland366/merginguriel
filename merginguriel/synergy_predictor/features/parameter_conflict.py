"""
Approach A: Parameter-space conflict features.

For a set of source task vectors, computes:
1. Pairwise sign agreement rate (global and per-layer)
2. Magnitude ratio between task vectors
3. Norm statistics across sources

Only requires model weights — no data, no inference.
"""

from __future__ import annotations

import re
from itertools import combinations
import numpy as np
import torch

# XLM-RoBERTa layer pattern
LAYER_PATTERN = re.compile(r"roberta\.encoder\.layer\.(\d+)\.")
NUM_LAYERS = 12


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of elements where sign(a) == sign(b)."""
    return (torch.sign(a) == torch.sign(b)).float().mean().item()


def compute_pairwise_sign_agreement(
    tv_a: dict[str, torch.Tensor],
    tv_b: dict[str, torch.Tensor],
) -> float:
    """
    Global sign agreement: fraction of all parameters where
    sign(τ_A) == sign(τ_B).

    Range: [0, 1]. Higher = less conflict.
    0.5 = random. Below 0.5 = anti-correlated.
    """
    common = sorted(set(tv_a.keys()) & set(tv_b.keys()))
    parts_a = [tv_a[k].flatten().float() for k in common]
    parts_b = [tv_b[k].flatten().float() for k in common]
    return _sign_agreement(torch.cat(parts_a), torch.cat(parts_b))


def compute_per_layer_sign_agreement(
    tv_a: dict[str, torch.Tensor],
    tv_b: dict[str, torch.Tensor],
) -> dict[int, float]:
    """
    Sign agreement computed per transformer layer.

    Returns dict: layer_id -> agreement_rate.
    """
    layer_params_a: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
    layer_params_b: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}

    common = sorted(set(tv_a.keys()) & set(tv_b.keys()))
    for name in common:
        match = LAYER_PATTERN.match(name)
        if match:
            layer_id = int(match.group(1))
            layer_params_a[layer_id].append(tv_a[name].flatten().float())
            layer_params_b[layer_id].append(tv_b[name].flatten().float())

    result = {}
    for layer_id in range(NUM_LAYERS):
        if layer_params_a[layer_id]:
            a_cat = torch.cat(layer_params_a[layer_id])
            b_cat = torch.cat(layer_params_b[layer_id])
            result[layer_id] = _sign_agreement(a_cat, b_cat)

    return result


def compute_magnitude_ratio(
    tv_a: dict[str, torch.Tensor],
    tv_b: dict[str, torch.Tensor],
) -> float:
    """
    max(||τ_A||, ||τ_B||) / min(||τ_A||, ||τ_B||).

    Always >= 1. Large values mean one source dominates.
    """
    common = sorted(set(tv_a.keys()) & set(tv_b.keys()))
    vec_a = torch.cat([tv_a[k].flatten().float() for k in common])
    vec_b = torch.cat([tv_b[k].flatten().float() for k in common])
    norm_a = vec_a.norm().item()
    norm_b = vec_b.norm().item()
    if min(norm_a, norm_b) < 1e-10:
        return float("inf")
    return max(norm_a, norm_b) / min(norm_a, norm_b)


def compute_task_vector_norm(tv: dict[str, torch.Tensor]) -> float:
    """L2 norm of a task vector (all parameters flattened)."""
    vec = torch.cat([v.flatten().float() for v in tv.values()])
    return vec.norm().item()


def compute_parameter_conflict_features(
    task_vectors: dict[str, dict[str, torch.Tensor]],
    source_locales: list[str],
) -> dict[str, float]:
    """
    Compute all parameter conflict features for a source set.

    Args:
        task_vectors: locale -> {param_name: delta_tensor}
        source_locales: which locales to analyze

    Returns:
        Flat dict of feature_name -> value.
    """
    tvs = {loc: task_vectors[loc] for loc in source_locales}
    pairs = list(combinations(source_locales, 2))

    # Global sign agreement
    global_agreements = [
        compute_pairwise_sign_agreement(tvs[a], tvs[b]) for a, b in pairs
    ]

    # Per-layer sign agreement (averaged across pairs)
    layer_agreements = {i: [] for i in range(NUM_LAYERS)}
    for a, b in pairs:
        per_layer = compute_per_layer_sign_agreement(tvs[a], tvs[b])
        for layer_id, val in per_layer.items():
            layer_agreements[layer_id].append(val)

    # Magnitude ratios
    mag_ratios = [compute_magnitude_ratio(tvs[a], tvs[b]) for a, b in pairs]

    # Norms
    norms = [compute_task_vector_norm(tvs[loc]) for loc in source_locales]

    features = {
        # Global sign agreement stats
        "sign_agreement_mean": float(np.mean(global_agreements)),
        "sign_agreement_min": float(np.min(global_agreements)),
        "sign_agreement_std": float(np.std(global_agreements)),
        # Per-layer sign agreement (mean across pairs)
        **{
            f"sign_agreement_layer_{i}": float(np.mean(layer_agreements[i]))
            for i in range(NUM_LAYERS)
            if layer_agreements[i]
        },
        # Layer groups
        "sign_agreement_bottom": float(
            np.mean([np.mean(layer_agreements[i]) for i in range(4) if layer_agreements[i]])
        ),
        "sign_agreement_middle": float(
            np.mean([np.mean(layer_agreements[i]) for i in range(4, 8) if layer_agreements[i]])
        ),
        "sign_agreement_top": float(
            np.mean([np.mean(layer_agreements[i]) for i in range(8, 12) if layer_agreements[i]])
        ),
        # Magnitude ratio stats
        "magnitude_ratio_max": float(np.max(mag_ratios)),
        "magnitude_ratio_mean": float(np.mean(mag_ratios)),
        # Norm stats
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_cv": float(np.std(norms) / np.mean(norms)) if np.mean(norms) > 0 else 0.0,
    }

    return features
