"""
Approach C: Prediction agreement on unlabeled target data.

Runs all source models on the target locale's test set (without labels)
and measures functional consensus: pairwise agreement, ensemble entropy,
and majority vote confidence.

Requires: fine-tuned models + unlabeled target data (MASSIVE, public).
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_target_utterances(
    locale: str,
    split: str = "test",
    max_samples: int = 2000,
) -> list[str]:
    """Load unlabeled utterances from MASSIVE for a target locale."""
    from datasets import load_dataset

    dataset = load_dataset(
        "AmazonScience/massive", locale, split=split, trust_remote_code=True
    )
    n = min(max_samples, len(dataset))
    return [dataset[i]["utt"] for i in range(n)]


def get_source_predictions(
    model_path: str,
    utterances: list[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a source model on utterances.

    Returns:
        predicted_labels: (n_samples,) int array
        logits: (n_samples, n_classes) float array
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_logits = []

    for i in range(0, len(utterances), batch_size):
        batch = utterances[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.cpu()

        all_preds.append(torch.argmax(logits, dim=1).numpy())
        all_logits.append(logits.numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_preds), np.concatenate(all_logits, axis=0)


def compute_prediction_agreement_features(
    source_locales: list[str],
    target_locale: str,
    models_dir: str,
    device: str = "cuda",
    max_samples: int = 2000,
    base_model_name: str = "xlm-roberta-base",
) -> dict[str, float]:
    """
    Compute all prediction agreement features.

    Args:
        source_locales: source locale codes
        target_locale: target locale for unlabeled data
        models_dir: directory containing model folders
        device: torch device
        max_samples: max utterances to use
        base_model_name: model naming prefix

    Returns:
        Flat dict of feature_name -> value.
    """
    utterances = load_target_utterances(target_locale, max_samples=max_samples)

    # Get predictions from each source
    predictions = {}
    logits = {}
    for loc in tqdm(source_locales, desc=f"Predictions for {target_locale}"):
        model_path = os.path.join(
            models_dir, f"{base_model_name}_massive_k_{loc}"
        )
        preds, lgts = get_source_predictions(model_path, utterances, device)
        predictions[loc] = preds
        logits[loc] = lgts

    pairs = list(combinations(source_locales, 2))

    # Pairwise prediction agreement
    agreements = []
    for a, b in pairs:
        agree = (predictions[a] == predictions[b]).mean()
        agreements.append(agree)

    # Ensemble entropy: average softmax then compute entropy
    softmax_stack = np.stack(
        [F.softmax(torch.tensor(logits[loc]), dim=1).numpy() for loc in source_locales]
    )
    avg_softmax = softmax_stack.mean(axis=0)  # (n_samples, n_classes)
    # Per-sample entropy
    sample_entropy = -(avg_softmax * np.log(avg_softmax + 1e-10)).sum(axis=1)

    # Majority vote confidence
    pred_matrix = np.stack([predictions[loc] for loc in source_locales])  # (n_sources, n_samples)
    majority_counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=1).max(), axis=0, arr=pred_matrix
    )
    majority_frac = majority_counts / len(source_locales)

    # Per-source confidence (max softmax prob)
    source_confidences = []
    for loc in source_locales:
        probs = F.softmax(torch.tensor(logits[loc]), dim=1).numpy()
        source_confidences.append(probs.max(axis=1).mean())

    features = {
        # Pairwise agreement
        "pred_agreement_mean": float(np.mean(agreements)),
        "pred_agreement_min": float(np.min(agreements)),
        "pred_agreement_std": float(np.std(agreements)),
        # Ensemble entropy
        "ensemble_entropy_mean": float(np.mean(sample_entropy)),
        "ensemble_entropy_median": float(np.median(sample_entropy)),
        "ensemble_entropy_p90": float(np.percentile(sample_entropy, 90)),
        "ensemble_high_entropy_frac": float((sample_entropy > np.log(10)).mean()),
        # Majority vote
        "majority_vote_confidence": float(majority_frac.mean()),
        "majority_unanimous_frac": float((majority_frac == 1.0).mean()),
        # Source confidence
        "source_confidence_mean": float(np.mean(source_confidences)),
        "source_confidence_std": float(np.std(source_confidences)),
    }

    return features
