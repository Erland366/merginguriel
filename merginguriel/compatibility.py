"""
Source Compatibility Analysis Module

Computes pairwise compatibility between source models using:
1. Task Vector Cosine Similarity (parameter space) - fast, no data needed
2. CKA (representation space) - requires shared input data

These metrics measure how well sources work together when merged,
independent of any specific target language.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Resolve repository root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, "submodules/auto_merge_llm")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from auto_merge_llm.utils.task_vector import TaskVector

# All 49 MASSIVE locales
ALL_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB",
    "da-DK", "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI",
    "fr-FR", "he-IL", "hi-IN", "hu-HU", "hy-AM", "id-ID", "is-IS",
    "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH", "kn-IN", "ko-KR",
    "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL",
    "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SI", "sq-AL", "sv-SE",
    "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "uk-UA",
    "ur-PK", "vi-VN", "zh-CN", "zh-TW",
]

# Default model directories
DEFAULT_MODEL_DIRS = {
    "xlm-roberta-base": Path(project_root) / "haryos_model",
    "xlm-roberta-large": Path(project_root) / "haryos_model_large",
}


@dataclass
class CompatibilityConfig:
    """Configuration for compatibility computation."""

    metric: str = "task_vector_cosine"  # "task_vector_cosine" or "cka"
    pretrained_model_name: str = "xlm-roberta-base"
    models_dir: Optional[Path] = None
    exclude_param_patterns: Optional[List[str]] = None
    # CKA-specific options
    cka_num_samples: int = 500
    cka_dataset: str = "AmazonScience/massive"
    cka_split: str = "train"
    cka_layer: int = -1  # -1 = last layer, or specific layer index
    device: str = "cuda"

    def __post_init__(self):
        if self.models_dir is None:
            self.models_dir = DEFAULT_MODEL_DIRS.get(
                self.pretrained_model_name, Path(project_root) / "haryos_model"
            )
        # Default: exclude classifier head (different dims between pretrained and finetuned)
        if self.exclude_param_patterns is None:
            self.exclude_param_patterns = ["classifier"]


# =============================================================================
# Task Vector Cosine Similarity (Parameter Space)
# =============================================================================


def compute_task_vector_cosine(
    task_vector_a: Dict[str, torch.Tensor],
    task_vector_b: Dict[str, torch.Tensor],
) -> float:
    """
    Compute cosine similarity between two task vectors.

    Task vectors are parameter differences: (finetuned - pretrained).
    This measures alignment in parameter space direction.

    Args:
        task_vector_a: Dict mapping param names to delta tensors for model A
        task_vector_b: Dict mapping param names to delta tensors for model B

    Returns:
        Cosine similarity in [-1, 1] range
    """
    # Get common parameters
    common_params = set(task_vector_a.keys()) & set(task_vector_b.keys())
    if not common_params:
        raise ValueError("No common parameters between task vectors")

    # Flatten all parameters into single vectors
    vec_a_parts = []
    vec_b_parts = []
    for param_name in sorted(common_params):
        vec_a_parts.append(task_vector_a[param_name].flatten().float())
        vec_b_parts.append(task_vector_b[param_name].flatten().float())

    vec_a = torch.cat(vec_a_parts)
    vec_b = torch.cat(vec_b_parts)

    # Compute cosine similarity
    similarity = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1)
    return similarity.item()


def extract_task_vector(
    finetuned_model_path: Path,
    pretrained_model: torch.nn.Module,
    exclude_param_names_regex: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract task vector from a finetuned model.

    Args:
        finetuned_model_path: Path to finetuned model directory
        pretrained_model: The pretrained base model
        exclude_param_names_regex: Patterns to exclude from task vector

    Returns:
        Dict mapping param names to delta tensors (finetuned - pretrained)
    """
    from transformers import AutoModelForSequenceClassification

    finetuned_model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_path
    )

    # Ensure exclude_param_names_regex is a list, not None
    if exclude_param_names_regex is None:
        exclude_param_names_regex = []

    task_vector = TaskVector(
        pretrained_model=pretrained_model,
        finetuned_model=finetuned_model,
        exclude_param_names_regex=exclude_param_names_regex,
    )

    return task_vector.task_vector_param_dict


# =============================================================================
# CKA (Representation Space)
# =============================================================================


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment) between two activation matrices.

    CKA measures similarity between representations - higher values mean
    the two models encode similar information structure.

    Reference: Kornblith et al. "Similarity of Neural Network Representations Revisited" (2019)

    Args:
        X: Activation matrix of shape (n_samples, n_features_x)
        Y: Activation matrix of shape (n_samples, n_features_y)

    Returns:
        CKA similarity in [0, 1] range
    """
    X = X.float()
    Y = Y.float()

    # Center the activations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (linear kernel)
    # K = X @ X.T, L = Y @ Y.T
    # But we use the Frobenius inner product form for efficiency:
    # CKA = ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)

    YtX = Y.T @ X  # (n_features_y, n_features_x)
    XtX = X.T @ X  # (n_features_x, n_features_x)
    YtY = Y.T @ Y  # (n_features_y, n_features_y)

    # Frobenius norms
    hsic_xy = (YtX * YtX).sum()  # ||Y.T @ X||_F^2
    hsic_xx = (XtX * XtX).sum()  # ||X.T @ X||_F^2
    hsic_yy = (YtY * YtY).sum()  # ||Y.T @ Y||_F^2

    # CKA
    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    cka = hsic_xy / denominator
    return cka.item()


def extract_hidden_states(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    layer: int = -1,
    device: str = "cuda",
    max_length: int = 128,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Extract hidden state representations from a model.

    Args:
        model: The transformer model
        tokenizer: Associated tokenizer
        texts: Input texts to process
        layer: Which layer to extract (-1 = last layer)
        device: Device to run on
        max_length: Max sequence length
        batch_size: Batch size for processing

    Returns:
        Tensor of shape (n_samples, hidden_dim) using [CLS] token representations
    """
    model = model.to(device)
    model.eval()

    all_hidden_states = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)

        # Use [CLS] token representation (index 0)
        cls_hidden = hidden_states[:, 0, :]  # (batch, hidden_dim)
        all_hidden_states.append(cls_hidden.cpu())

    return torch.cat(all_hidden_states, dim=0)


def load_cka_input_samples(
    dataset_name: str = "AmazonScience/massive",
    split: str = "train",
    num_samples: int = 500,
    locale: str = "en-US",
) -> List[str]:
    """
    Load input samples for CKA computation.

    Uses a single locale (default: en-US) so all models process the same inputs.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        num_samples: Number of samples to load
        locale: Locale to use for input texts

    Returns:
        List of input text strings
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, locale, split=split)

    # Sample deterministically
    indices = list(range(min(num_samples, len(dataset))))
    samples = [dataset[i]["utt"] for i in indices]

    return samples


# =============================================================================
# Matrix Computation
# =============================================================================


def compute_pairwise_tv_cosine_matrix(
    config: CompatibilityConfig,
    locales: List[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise Task Vector Cosine similarity matrix for all source locales.

    Args:
        config: Compatibility configuration
        locales: List of locale codes to include
        verbose: Print progress information

    Returns:
        Symmetric DataFrame where entry (i,j) is TV cosine similarity between i and j
    """
    from transformers import AutoModelForSequenceClassification

    n = len(locales)
    matrix = np.zeros((n, n))

    if verbose:
        print(f"Computing Task Vector Cosine matrix for {n} locales")
        print(f"Models directory: {config.models_dir}")

    # Load pretrained model once
    if verbose:
        print(f"Loading pretrained model: {config.pretrained_model_name}")

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_name
    )

    # Pre-compute task vectors for all locales
    task_vectors = {}
    if verbose:
        print("Computing task vectors...")
        locale_iter = tqdm(locales, desc="Task vectors")
    else:
        locale_iter = locales

    for locale in locale_iter:
        model_path = config.models_dir / f"xlm-roberta-base_massive_k_{locale}"
        if not model_path.exists():
            print(f"  Warning: Model not found for {locale}, skipping")
            continue

        task_vectors[locale] = extract_task_vector(
            model_path,
            pretrained_model,
            config.exclude_param_patterns,
        )

    # Compute pairwise similarities
    computed_locales = list(task_vectors.keys())
    if verbose:
        print(f"Computing pairwise similarities for {len(computed_locales)} locales...")

    for i, locale_a in enumerate(computed_locales):
        for j, locale_b in enumerate(computed_locales):
            if i <= j:  # Only compute upper triangle (symmetric)
                if locale_a == locale_b:
                    score = 1.0  # Perfect self-similarity
                else:
                    score = compute_task_vector_cosine(
                        task_vectors[locale_a],
                        task_vectors[locale_b],
                    )
                # Get indices in original locales list
                idx_a = locales.index(locale_a)
                idx_b = locales.index(locale_b)
                matrix[idx_a, idx_b] = score
                matrix[idx_b, idx_a] = score  # Mirror for symmetry

                if verbose and i != j and (i * n + j) % 100 == 0:
                    print(f"  {locale_a} <-> {locale_b}: {score:.4f}")

    return pd.DataFrame(matrix, index=locales, columns=locales)


def compute_pairwise_cka_matrix(
    config: CompatibilityConfig,
    locales: List[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise CKA similarity matrix for all source locales.

    Args:
        config: Compatibility configuration
        locales: List of locale codes to include
        verbose: Print progress information

    Returns:
        Symmetric DataFrame where entry (i,j) is CKA similarity between i and j
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    n = len(locales)
    matrix = np.zeros((n, n))

    if verbose:
        print(f"Computing CKA matrix for {n} locales")
        print(f"Using {config.cka_num_samples} shared input samples from en-US")

    # Load shared input samples (same for all models)
    input_texts = load_cka_input_samples(
        config.cka_dataset,
        config.cka_split,
        config.cka_num_samples,
    )
    if verbose:
        print(f"Loaded {len(input_texts)} input samples")

    # Pre-compute hidden states for all locales
    hidden_states = {}
    if verbose:
        print("Extracting hidden states...")
        locale_iter = tqdm(locales, desc="Hidden states")
    else:
        locale_iter = locales

    for locale in locale_iter:
        model_path = config.models_dir / f"xlm-roberta-base_massive_k_{locale}"
        if not model_path.exists():
            print(f"  Warning: Model not found for {locale}, skipping")
            continue

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        hidden_states[locale] = extract_hidden_states(
            model,
            tokenizer,
            input_texts,
            layer=config.cka_layer,
            device=config.device,
        )

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute pairwise CKA
    computed_locales = list(hidden_states.keys())
    if verbose:
        print(f"Computing pairwise CKA for {len(computed_locales)} locales...")

    for i, locale_a in enumerate(computed_locales):
        for j, locale_b in enumerate(computed_locales):
            if i <= j:  # Only compute upper triangle (symmetric)
                if locale_a == locale_b:
                    score = 1.0
                else:
                    score = linear_cka(
                        hidden_states[locale_a],
                        hidden_states[locale_b],
                    )
                idx_a = locales.index(locale_a)
                idx_b = locales.index(locale_b)
                matrix[idx_a, idx_b] = score
                matrix[idx_b, idx_a] = score

                if verbose and i != j and (i * n + j) % 50 == 0:
                    print(f"  {locale_a} <-> {locale_b}: {score:.4f}")

    return pd.DataFrame(matrix, index=locales, columns=locales)


def compute_pairwise_compatibility_matrix(
    config: CompatibilityConfig,
    locales: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise compatibility matrix using the configured metric.

    Args:
        config: Compatibility configuration (metric determines which method to use)
        locales: List of locale codes (defaults to ALL_LOCALES)
        verbose: Print progress information

    Returns:
        Symmetric DataFrame where entry (i,j) is compatibility between i and j
    """
    if locales is None:
        locales = ALL_LOCALES

    if config.metric == "task_vector_cosine":
        return compute_pairwise_tv_cosine_matrix(config, locales, verbose)
    elif config.metric == "cka":
        return compute_pairwise_cka_matrix(config, locales, verbose)
    else:
        raise ValueError(f"Unknown metric: {config.metric}")


# =============================================================================
# Loading and Using Compatibility Matrices
# =============================================================================


def load_compatibility_matrix(matrix_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load pre-computed compatibility matrix from CSV.

    Args:
        matrix_path: Path to CSV file
        verbose: Print loading information

    Returns:
        Compatibility matrix as DataFrame
    """
    if verbose:
        print(f"Loading compatibility matrix from {matrix_path}")

    if not matrix_path.exists():
        raise FileNotFoundError(f"Compatibility matrix not found: {matrix_path}")

    df = pd.read_csv(matrix_path, index_col=0)

    if verbose:
        print(f"Loaded matrix with shape: {df.shape}")

    return df


def get_source_compatibility_scores(
    compatibility_matrix: pd.DataFrame,
    source_locales: List[str],
) -> Dict[str, float]:
    """
    Get compatibility scores for a set of source locales.

    For each source, computes the average pairwise compatibility with all
    other sources in the set. Higher score = more compatible with group.

    Args:
        compatibility_matrix: Pre-computed compatibility matrix
        source_locales: List of source locale codes

    Returns:
        Dict mapping locale to average compatibility score
    """
    scores = {}
    for locale in source_locales:
        if locale not in compatibility_matrix.index:
            scores[locale] = 0.0
            continue

        other_locales = [l for l in source_locales if l != locale]
        if other_locales:
            # Filter to only locales that exist in matrix
            valid_others = [
                l for l in other_locales if l in compatibility_matrix.columns
            ]
            if valid_others:
                avg_compat = compatibility_matrix.loc[locale, valid_others].mean()
            else:
                avg_compat = 1.0
        else:
            avg_compat = 1.0  # Only one source

        scores[locale] = float(avg_compat)

    return scores


def apply_compatibility_weights(
    similarity_weights: List[Tuple[str, float]],
    compatibility_matrix: pd.DataFrame,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    Apply compatibility-based adjustment to similarity weights.

    Uses multiplicative weighting: final_weight = similarity × avg_compatibility

    Args:
        similarity_weights: List of (locale, similarity_weight) tuples
        compatibility_matrix: Pre-computed compatibility matrix
        normalize: Whether to renormalize weights to sum to 1

    Returns:
        List of (locale, adjusted_weight) tuples
    """
    source_locales = [locale for locale, _ in similarity_weights]
    compat_scores = get_source_compatibility_scores(compatibility_matrix, source_locales)

    # Apply multiplicative weighting
    adjusted = []
    for locale, sim_weight in similarity_weights:
        compat = compat_scores.get(locale, 1.0)
        adjusted.append((locale, sim_weight * compat))

    # Normalize if requested
    if normalize:
        total = sum(w for _, w in adjusted)
        if total > 0:
            adjusted = [(locale, w / total) for locale, w in adjusted]

    return adjusted


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "CompatibilityConfig",
    "ALL_LOCALES",
    # Task Vector Cosine
    "compute_task_vector_cosine",
    "extract_task_vector",
    # CKA
    "linear_cka",
    "extract_hidden_states",
    "load_cka_input_samples",
    # Matrix computation
    "compute_pairwise_compatibility_matrix",
    "compute_pairwise_tv_cosine_matrix",
    "compute_pairwise_cka_matrix",
    # Loading and using
    "load_compatibility_matrix",
    "get_source_compatibility_scores",
    "apply_compatibility_weights",
]
