"""
Approach B: Task Singular Vector (TSV) subspace overlap.

For each source, per layer, per weight matrix:
1. Reshape task vector delta into a 2D matrix
2. Compute SVD -> top-k right singular vectors
3. Compute pairwise subspace overlap via principal angles

Reference: "Task Singular Vectors: Reducing Task Interference
in Model Merging" (CVPR 2025, Gargiulo et al.)

Only requires model weights — no data, no inference.
"""

from __future__ import annotations

import re
from itertools import combinations
import numpy as np
import torch

LAYER_PATTERN = re.compile(r"roberta\.encoder\.layer\.(\d+)\.")
# Only weight matrices (2D), skip biases and LayerNorm
WEIGHT_PATTERN = re.compile(
    r"roberta\.encoder\.layer\.(\d+)\."
    r"(attention\.self\.(query|key|value)|attention\.output\.dense|"
    r"intermediate\.dense|output\.dense)\.weight"
)
NUM_LAYERS = 12


def _is_attention_param(name: str) -> bool:
    return "attention" in name


def compute_task_singular_vectors(
    delta_matrix: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Compute top-k right singular vectors of a task vector delta matrix.

    Returns matrix of shape (top_k, n_cols) or fewer if rank < top_k.
    """
    U, S, Vh = torch.linalg.svd(delta_matrix.float(), full_matrices=False)
    k = min(top_k, Vh.shape[0])
    return Vh[:k]


def compute_subspace_overlap(
    V1: torch.Tensor,
    V2: torch.Tensor,
) -> float:
    """
    Subspace overlap via principal angles.

    overlap = ||V1 @ V2^T||_F^2 / min(k1, k2)

    Range: [0, 1]. Higher = more overlap = more similar subspaces.
    """
    k = min(V1.shape[0], V2.shape[0])
    if k == 0:
        return 0.0
    cross = V1 @ V2.T
    return (cross**2).sum().item() / k


def extract_weight_matrices(
    task_vector: dict[str, torch.Tensor],
) -> dict[int, list[tuple[str, torch.Tensor]]]:
    """
    Extract 2D weight matrices grouped by layer.

    Returns dict: layer_id -> [(param_name, delta_matrix), ...]
    """
    layers: dict[int, list[tuple[str, torch.Tensor]]] = {
        i: [] for i in range(NUM_LAYERS)
    }

    for name, tensor in task_vector.items():
        match = WEIGHT_PATTERN.match(name)
        if match and tensor.dim() == 2:
            layer_id = int(match.group(1))
            layers[layer_id].append((name, tensor))

    return layers


def compute_subspace_overlap_features(
    task_vectors: dict[str, dict[str, torch.Tensor]],
    source_locales: list[str],
    top_k: int = 10,
) -> dict[str, float]:
    """
    Compute all TSV subspace overlap features for a source set.

    For each pair of sources, for each layer, for each weight matrix:
    1. Compute TSVs (top-k singular vectors of delta)
    2. Compute pairwise subspace overlap
    3. Aggregate across matrices, layers, and pairs

    Returns flat dict of feature_name -> value.
    """
    tvs = {loc: task_vectors[loc] for loc in source_locales}
    pairs = list(combinations(source_locales, 2))

    # Pre-compute TSVs for all sources, all layers, all matrices
    # Structure: locale -> layer -> [(param_name, TSV_matrix)]
    all_tsvs: dict[str, dict[int, list[tuple[str, torch.Tensor]]]] = {}
    for loc in source_locales:
        weight_matrices = extract_weight_matrices(tvs[loc])
        all_tsvs[loc] = {}
        for layer_id, matrices in weight_matrices.items():
            all_tsvs[loc][layer_id] = [
                (name, compute_task_singular_vectors(mat, top_k))
                for name, mat in matrices
            ]

    # Compute pairwise overlaps
    layer_overlaps = {i: [] for i in range(NUM_LAYERS)}
    attn_overlaps = []
    ffn_overlaps = []
    all_overlaps = []

    for loc_a, loc_b in pairs:
        for layer_id in range(NUM_LAYERS):
            mats_a = all_tsvs[loc_a].get(layer_id, [])
            mats_b = all_tsvs[loc_b].get(layer_id, [])

            # Match by param name
            dict_a = {name: tsv for name, tsv in mats_a}
            dict_b = {name: tsv for name, tsv in mats_b}

            for name in dict_a:
                if name in dict_b:
                    overlap = compute_subspace_overlap(dict_a[name], dict_b[name])
                    layer_overlaps[layer_id].append(overlap)
                    all_overlaps.append(overlap)

                    if _is_attention_param(name):
                        attn_overlaps.append(overlap)
                    else:
                        ffn_overlaps.append(overlap)

    features = {
        # Global overlap stats
        "subspace_overlap_mean": float(np.mean(all_overlaps)) if all_overlaps else 0.0,
        "subspace_overlap_min": float(np.min(all_overlaps)) if all_overlaps else 0.0,
        "subspace_overlap_std": float(np.std(all_overlaps)) if all_overlaps else 0.0,
        # Per-layer overlap
        **{
            f"subspace_overlap_layer_{i}": float(np.mean(layer_overlaps[i]))
            for i in range(NUM_LAYERS)
            if layer_overlaps[i]
        },
        # Layer groups
        "subspace_overlap_bottom": float(
            np.mean([np.mean(layer_overlaps[i]) for i in range(4) if layer_overlaps[i]])
        )
        if any(layer_overlaps[i] for i in range(4))
        else 0.0,
        "subspace_overlap_middle": float(
            np.mean([np.mean(layer_overlaps[i]) for i in range(4, 8) if layer_overlaps[i]])
        )
        if any(layer_overlaps[i] for i in range(4, 8))
        else 0.0,
        "subspace_overlap_top": float(
            np.mean([np.mean(layer_overlaps[i]) for i in range(8, 12) if layer_overlaps[i]])
        )
        if any(layer_overlaps[i] for i in range(8, 12))
        else 0.0,
        # Component split
        "subspace_overlap_attn_mean": float(np.mean(attn_overlaps)) if attn_overlaps else 0.0,
        "subspace_overlap_ffn_mean": float(np.mean(ffn_overlaps)) if ffn_overlaps else 0.0,
    }

    return features
