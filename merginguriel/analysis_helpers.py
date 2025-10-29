from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from merginguriel.run_merging_pipeline_refactored import (
    MergeConfig,
    ModelMerger,
    WeightCalculatorFactory,
)


@dataclass
class ModelArtifacts:
    """Convenience container for a loaded model + tokenizer."""

    model: AutoModelForSequenceClassification
    tokenizer: Any
    model_path: Path


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """
    Pick an available device. Accepts strings like 'cuda', 'cpu', or None.
    Falls back to CPU if CUDA is unavailable.
    """
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested, torch.device):
        return requested
    requested = requested.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_model_artifacts(
    model_path: str | Path,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
) -> ModelArtifacts:
    """
    Load a Hugging Face classification model and tokenizer from disk.

    Parameters
    ----------
    model_path:
        Path to the directory containing the model weights/config.
    device:
        Optional device spec (e.g., 'cpu', 'cuda'). Defaults to auto.
    trust_remote_code:
        Forwarded to Hugging Face loaders if custom code is required.
    """
    model_path = Path(model_path).expanduser().resolve()
    device = resolve_device(device)

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return ModelArtifacts(model=model, tokenizer=tokenizer, model_path=model_path)


def merge_models_in_memory(
    config: MergeConfig,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
) -> tuple[ModelArtifacts, dict[str, Any]]:
    """
    Execute a merge using the refactored pipeline components without writing to disk.

    Returns both the loaded artifacts and metadata (models, weights) for inspection.
    """
    weight_calculator = WeightCalculatorFactory.create_calculator(config.mode)
    models_and_weights, base_model_info = weight_calculator.calculate_weights(config)
    merger = ModelMerger(config)
    merged_model, tokenizer = merger.merge_models(models_and_weights, base_model_info)

    device = resolve_device(device)
    merged_model.to(device)
    merged_model.eval()

    if trust_remote_code:
        tokenizer.init_kwargs["trust_remote_code"] = True

    artifacts = ModelArtifacts(
        model=merged_model,
        tokenizer=tokenizer,
        model_path=Path(f"{config.mode}:{config.target_lang}"),
    )
    metadata = {
        "models_and_weights": models_and_weights,
        "base_model": base_model_info,
    }
    return artifacts, metadata


def _extract_layer_id(parameter_name: str) -> str:
    """
    Attempt to pull a stable layer identifier out of a parameter name.
    Example: 'roberta.encoder.layer.3.attention.self.query.weight' -> 'layer.3.attention.self.query'.
    """
    parts = parameter_name.split(".")
    if "layer" in parts:
        idx = parts.index("layer")
        # include 'layer', the numeric index, and the next two descriptors for context
        tail = parts[idx : min(idx + 4, len(parts))]
        return ".".join(tail)
    # fallback to the immediate module scope
    return ".".join(parts[:-1]) if len(parts) > 1 else parts[0]


def iter_named_tensors(
    model: torch.nn.Module,
    filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Yield (name, tensor) pairs from a model, optionally filtering.
    """
    for name, param in model.named_parameters():
        if filter_fn and not filter_fn(name, param):
            continue
        yield name, param.detach()


def compute_weight_deltas(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> pd.DataFrame:
    """
    Compare parameters between two models and return summary statistics.

    The output frame includes:
    - parameter: full parameter name
    - layer: coarse layer identifier
    - delta_l2: L2 norm of the parameter difference
    - delta_mean_abs: mean absolute difference
    - cosine_with_reference: cosine similarity with the reference tensor
    - reference_norm: L2 norm of the reference tensor
    - candidate_norm: L2 norm of the candidate tensor
    """
    ref_state = dict(iter_named_tensors(reference_model, filter_fn))
    rows: list[dict[str, Any]] = []

    for name, candidate_tensor in iter_named_tensors(candidate_model, filter_fn):
        if name not in ref_state:
            continue
        ref_tensor = ref_state[name]
        delta = (candidate_tensor - ref_tensor).float()
        candidate_tensor = candidate_tensor.float()
        ref_tensor = ref_tensor.float()

        rows.append(
            {
                "parameter": name,
                "layer": _extract_layer_id(name),
                "delta_l2": delta.norm().item(),
                "delta_mean_abs": delta.abs().mean().item(),
                "cosine_with_reference": F.cosine_similarity(
                    candidate_tensor.view(-1), ref_tensor.view(-1), dim=0
                ).item(),
                "reference_norm": ref_tensor.norm().item(),
                "candidate_norm": candidate_tensor.norm().item(),
            }
        )

    return pd.DataFrame(rows)


def aggregate_parameter_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate parameter delta statistics at the layer granularity.
    """
    if df.empty:
        return df
    grouped = (
        df.groupby("layer")
        .agg(
            delta_l2_sum=("delta_l2", "sum"),
            delta_l2_mean=("delta_l2", "mean"),
            delta_mean_abs=("delta_mean_abs", "mean"),
            cosine_mean=("cosine_with_reference", "mean"),
            reference_norm_mean=("reference_norm", "mean"),
            candidate_norm_mean=("candidate_norm", "mean"),
        )
        .reset_index()
        .sort_values("layer")
    )
    return grouped


def _safe_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy with numerical stability."""
    eps = 1e-12
    return -(probs * (probs + eps).log()).sum(dim=dim)


def collect_model_signals(
    artifacts: ModelArtifacts,
    texts: Sequence[str],
    device: str | torch.device | None = None,
    max_length: int = 128,
) -> dict[str, Any]:
    """
    Run the model on a batch of texts and capture logits, attentions, and hidden states.

    Returns a dictionary containing:
    - inputs: tokenized batch (on CPU)
    - logits: raw logits (CPU tensor)
    - attentions: tuple of attention tensors (CPU)
    - hidden_states: tuple of hidden state tensors (CPU)
    """
    device = resolve_device(device) if device else artifacts.model.device
    enc = artifacts.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = artifacts.model(
            **enc,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    cpu_inputs = {k: v.cpu() for k, v in enc.items()}
    attentions = tuple(t.detach().cpu() for t in outputs.attentions or [])
    hidden_states = tuple(t.detach().cpu() for t in outputs.hidden_states or [])
    logits = outputs.logits.detach().cpu()

    return {
        "inputs": cpu_inputs,
        "logits": logits,
        "attentions": attentions,
        "hidden_states": hidden_states,
    }


def summarize_attentions(attentions: Sequence[torch.Tensor]) -> pd.DataFrame:
    """
    Convert a collection of attention tensors into head-level summary statistics.

    Each row captures:
    - layer
    - head
    - mean_prob: overall mean attention probability
    - entropy: token-level entropy averaged across batch and query positions
    - cls_focus: average attention mass placed on the first token (assumed CLS)
    - diagonal_focus: average mass placed on the diagonal (token attending to itself)
    """
    rows: list[dict[str, Any]] = []
    for layer_idx, layer_tensor in enumerate(attentions):
        # layer_tensor: [batch, heads, seq_len, seq_len]
        batch_size, num_heads, *_ = layer_tensor.shape
        for head_idx in range(num_heads):
            head_tensor = layer_tensor[:, head_idx, :, :]  # [batch, seq, seq]
            mean_prob = head_tensor.mean().item()
            entropy = _safe_entropy(head_tensor, dim=-1).mean().item()
            cls_focus = head_tensor[:, :, 0].mean().item()
            diagonal = head_tensor.diagonal(dim1=-2, dim2=-1)
            diag_focus = diagonal.mean().item()
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "mean_prob": mean_prob,
                    "entropy": entropy,
                    "cls_focus": cls_focus,
                    "diagonal_focus": diag_focus,
                }
            )
    return pd.DataFrame(rows)


def summarize_hidden_states(hidden_states: Sequence[torch.Tensor]) -> pd.DataFrame:
    """
    Produce layer-level activation statistics from hidden states.

    Hidden states are expected to include the embedding layer at index 0.
    """
    rows: list[dict[str, Any]] = []
    for layer_idx, tensor in enumerate(hidden_states):
        if tensor.dim() != 3:
            continue
        norms = tensor.norm(dim=-1)  # [batch, seq]
        rows.append(
            {
                "layer": layer_idx,
                "mean_token_norm": norms.mean().item(),
                "max_token_norm": norms.max().item(),
                "std_token_norm": norms.std().item(),
                "sequence_mean_norm": norms.mean(dim=-1).mean().item(),
            }
        )
    return pd.DataFrame(rows)


def summarize_logits(logits: torch.Tensor) -> pd.Series:
    """
    Capture simple distributional statistics over the output logits.
    """
    probs = logits.softmax(dim=-1)
    confidence = probs.max(dim=-1).values
    entropy = _safe_entropy(probs, dim=-1)
    return pd.Series(
        {
            "logit_mean": logits.mean().item(),
            "logit_std": logits.std().item(),
            "confidence_mean": confidence.mean().item(),
            "confidence_std": confidence.std().item(),
            "entropy_mean": entropy.mean().item(),
        }
    )


def ensure_text_samples(
    default_samples: Optional[Sequence[str]] = None,
    file_hint: Optional[str | Path] = None,
    limit: int = 16,
) -> list[str]:
    """
    Utility to source textual prompts for probing.

    Priority:
    1. Provided `default_samples`.
    2. Lines from a `file_hint` (one utterance per line).
    3. Fallback to a small built-in English list.
    """
    if default_samples:
        return list(default_samples)[:limit]

    if file_hint:
        file_path = Path(file_hint)
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            if lines:
                return lines[:limit]

    return [
        "How can I upgrade my flight booking?",
        "Show me the weather forecast for tomorrow evening.",
        "I need to reset the password for my online banking.",
        "Find vegetarian restaurants near my location.",
        "Translate this sentence into French.",
        "Remind me to call my mom at 6 PM.",
    ][:limit]


__all__ = [
    "ModelArtifacts",
    "aggregate_parameter_stats",
    "collect_model_signals",
    "compute_weight_deltas",
    "ensure_text_samples",
    "load_model_artifacts",
    "resolve_device",
    "summarize_attentions",
    "summarize_hidden_states",
    "summarize_logits",
]
