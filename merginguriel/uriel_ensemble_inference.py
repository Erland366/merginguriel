#!/usr/bin/env python3
"""
URIEL-Guided Ensemble Inference Script

This script implements the URIEL-guided ensemble inference approach described in section 7.3.
Instead of traditional voting, it combines the logits of multiple source models using
URIEL similarity scores as weights.

The proposed workflow:
1. For a given input, get the logit outputs from each of the K source models
2. For each source model, multiply its logit tensor by its URIEL similarity score
   to the target language
3. Sum the weighted logits from all source models to produce a final, ensembled
   logit distribution
4. The final prediction is the argmax of this combined distribution
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# --- Add submodules and project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
# --- End Path Setup ---




def get_subfolder_for_language(locale: str, subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9"):
    """Generate subfolder pattern based on locale."""
    return subfolder_pattern.format(locale=locale)


def load_similarity_weights(similarity_matrix_path: str, available_locales: List[str], target_lang: str,
                          num_languages: int = None, top_k: int = 20, sinkhorn_iterations: int = 20):
    """
    Load similarity weights and create model-to-weight mapping for target language.

    Uses language_similarity_matrix_unified.csv and local model directory.
    Direct locale matching - no mapping needed!
    """
    from merginguriel.similarity_utils import load_and_process_similarity

    try:
        print(f"Processing similarity weights for {len(available_locales)} available locales")

        # Get similarity weights using the utility
        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang, num_languages, top_k, sinkhorn_iterations, verbose=True
        )

        if not similar_languages:
            print(f"No similar languages found for target '{target_lang}'")
            return None

        # Create model-to-weight mapping - direct matching by locale
        models_and_weights = {}
        available_locales_set = set(available_locales)

        for locale, weight in similar_languages:
            # Direct locale match - no mapping needed!
            if locale in available_locales_set:
                models_and_weights[locale] = {
                    'weight': weight,
                    'locale': locale
                }
                print(f"  ✓ {locale}: {weight:.6f}")
            else:
                print(f"  ✗ Locale '{locale}' not in available models")

        # Normalize weights to sum to 1.0 after filtering for available models
        if models_and_weights:
            total_weight = sum(info['weight'] for info in models_and_weights.values())
            if total_weight > 0:
                normalization_factor = 1.0 / total_weight
                for info in models_and_weights.values():
                    info['weight'] *= normalization_factor
                print(f"\nNormalized weights to sum to 1.0 (total was: {total_weight:.6f})")
                for locale, info in models_and_weights.items():
                    print(f"  {locale}: {info['weight']:.6f}")

        print(f"\nFound {len(models_and_weights)} available models for {target_lang}")
        return models_and_weights

    except Exception as e:
        print(f"Error loading similarity weights: {e}")
        return None


def load_model_from_local(locale: str):
    """Load model directly from local haryos_model directory."""
    try:
        print(f"Loading model for locale: {locale}")

        # Construct the local model path
        model_dir = os.path.join("/home/coder/Python_project/MergingUriel/haryos_model",
                                f"xlm-roberta-base_massive_k_{locale}")

        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            return None, None

        print(f"Loading from local path: {model_dir}")

        # Load model and tokenizer from local directory
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model for locale {locale}: {e}")
        return None, None


def majority_vote(predictions: List[int]) -> int:
    """Simple majority voting."""
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]


def weighted_majority_vote(predictions: List[int], weights: List[float]) -> int:
    """Weighted majority voting."""
    # Count weighted votes for each class
    weighted_counts = {}
    for pred, weight in zip(predictions, weights):
        if pred not in weighted_counts:
            weighted_counts[pred] = 0
        weighted_counts[pred] += weight

    # Return the class with highest weighted count
    return max(weighted_counts.items(), key=lambda x: x[1])[0]


def soft_vote(probabilities: List[torch.Tensor], weights: List[float] = None) -> int:
    """Soft voting using probability averages."""
    if weights is None:
        weights = [1.0] * len(probabilities)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Average the probabilities
    avg_prob = torch.zeros_like(probabilities[0])
    for prob, weight in zip(probabilities, weights):
        avg_prob += prob * weight

    # Return the class with highest average probability
    return avg_prob.argmax().item()


def uriel_weighted_logits(logits_list: List[torch.Tensor], weights: List[float],
                         normalize_weights: bool = True) -> int:
    """
    Apply URIEL-guided weighting to logits and combine them.

    This implements the core algorithm from section 7.3:
    1. For each source model, multiply its logit tensor by its URIEL similarity score
    2. Sum the weighted logits to produce final ensembled logit distribution
    3. Use argmax for final prediction

    Args:
        logits_list: List of logits tensors from each source model
        weights: List of URIEL similarity scores for each model
        normalize_weights: Whether to normalize weights before applying

    Returns:
        Final prediction class (argmax of combined logits)
    """
    if not logits_list or not weights:
        raise ValueError("Empty logits list or weights")

    if len(logits_list) != len(weights):
        raise ValueError("Number of logits and weights must match")

    # Normalize weights if requested
    if normalize_weights:
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        weights = weights.tolist()

    # Start with zeros tensor of the same shape as the first logits
    weighted_logits = torch.zeros_like(logits_list[0])

    # Apply URIEL weights to each model's logits and sum them
    for logits, weight in zip(logits_list, weights):
        weighted_logits += logits * weight

    # Return the class with highest weighted logit
    return weighted_logits.argmax().item()


def run_ensemble_inference(models_and_weights: Dict, test_data: List[str],
                          voting_method: str = "majority") -> Tuple[List[int], Dict]:
    """
    Run ensemble inference on multiple models using various voting methods.

    Args:
        models_and_weights: Dictionary mapping model paths to their info
        test_data: List of text examples to classify
        voting_method: "majority", "weighted_majority", "soft", or "uriel_logits"

    Returns:
        Tuple of (predictions, metadata)
    """
    print(f"\n--- Running Ensemble Inference (Method: {voting_method}) ---")
    print(f"Number of models: {len(models_and_weights)}")
    print(f"Number of test examples: {len(test_data)}")

    # Load all models
    models = []
    tokenizers = []
    model_names = []
    weights = []

    for model_info in models_and_weights.values():
        locale = model_info['locale']
        weight = model_info['weight']

        print(f"\nLoading model for locale: {locale} (weight: {weight:.6f})")

        model, tokenizer = load_model_from_local(locale)

        if model is not None and tokenizer is not None:
            models.append(model)
            tokenizers.append(tokenizer)
            model_names.append(locale)
            weights.append(weight)
            print(f"✓ Successfully loaded model for {locale}")

            # Put model in eval mode
            model.eval()
        else:
            print(f"✗ Failed to load model for locale {locale}")
            raise RuntimeError(f"Failed to load model for locale: {locale}")

    if not models:
        raise ValueError("No models could be loaded successfully")

    print(f"\nSuccessfully loaded {len(models)} models for ensemble")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model in models:
        model.to(device)

    # Run inference
    all_predictions = []
    all_probabilities = []
    all_logits = []

    for i, text in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}")

        model_predictions = []
        model_probabilities = []
        model_logits = []

        # Get predictions from each model
        for model, tokenizer in zip(models, tokenizers):
            try:
                # Tokenize input
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get model prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    prediction = logits.argmax(dim=-1).item()

                model_predictions.append(prediction)
                model_probabilities.append(probabilities.squeeze())
                model_logits.append(logits.squeeze())

            except Exception as e:
                print(f"Error in model inference: {e}")
                # Use a default prediction if model fails
                model_predictions.append(0)
                model_probabilities.append(torch.zeros(60))  # Assuming 60 classes
                model_logits.append(torch.zeros(60))

        # Store individual model predictions
        all_predictions.append(model_predictions)
        all_probabilities.append(model_probabilities)
        all_logits.append(model_logits)

    # Apply voting to get final predictions
    print(f"\nApplying {voting_method} voting...")
    final_predictions = []

    for i, (predictions, probabilities, logits) in enumerate(zip(all_predictions, all_probabilities, all_logits)):
        if voting_method == "majority":
            final_pred = majority_vote(predictions)
        elif voting_method == "weighted_majority":
            final_pred = weighted_majority_vote(predictions, weights)
        elif voting_method == "soft":
            final_pred = soft_vote(probabilities, weights)
        elif voting_method == "uriel_logits":
            final_pred = uriel_weighted_logits(logits, weights, normalize_weights=True)
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")

        final_predictions.append(final_pred)

    # Prepare metadata
    metadata = {
        "voting_method": voting_method,
        "num_models": len(models),
        "model_names": model_names,
        "weights": weights,
        "device": str(device),
        "individual_predictions": all_predictions,
        "individual_probabilities": [[prob.tolist() for prob in probs] for probs in all_probabilities],
        "individual_logits": [[logit.tolist() for logit in logits] for logits in all_logits]
    }

    return final_predictions, metadata


def evaluate_ensemble(predictions: List[int], true_labels: List[int]) -> Dict:
    """Evaluate ensemble predictions against true labels."""
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0

    # Calculate per-class accuracy
    per_class_correct = {}
    per_class_total = {}

    for pred, true in zip(predictions, true_labels):
        if true not in per_class_total:
            per_class_total[true] = 0
            per_class_correct[true] = 0

        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1

    per_class_accuracy = {}
    for class_id in per_class_total:
        per_class_accuracy[class_id] = per_class_correct[class_id] / per_class_total[class_id]

    return {
        "accuracy": accuracy,
        "correct_predictions": correct,
        "total_predictions": total,
        "error_rate": 1 - accuracy,
        "per_class_accuracy": per_class_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="URIEL-Guided Ensemble Inference")
    parser.add_argument(
        "--target-lang",
        type=str,
        required=True,
        help="Target language/locale (e.g., 'en-US', 'sq-AL')"
    )
    parser.add_argument(
        "--subfolder-pattern",
        type=str,
        default="alpha_0.5_{locale}_epoch-9",
        help="Subfolder pattern for model loading"
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=5,
        help="Number of models to include in ensemble"
    )
    parser.add_argument(
        "--voting-method",
        type=str,
        choices=["majority", "weighted_majority", "soft", "uriel_logits"],
        default="majority",
        help="Voting method to use"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of test examples to evaluate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top similar languages to consider"
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn normalization iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="urie_ensemble_results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    print("*****************************************************")
    print("*      URIEL-Guided Ensemble Inference Pipeline      *")
    print("*****************************************************")

    # Use target language directly as locale code
    print(f"Target language: {args.target_lang}")

    # Load models and weights using similarity matrix and local models
    similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

    # Check what models are actually available locally
    local_models_dir = "/home/coder/Python_project/MergingUriel/haryos_model"
    available_locales = []
    for item in os.listdir(local_models_dir):
        if os.path.isdir(os.path.join(local_models_dir, item)) and item.startswith("xlm-roberta-base_massive_k_"):
            locale = item.replace("xlm-roberta-base_massive_k_", "")
            available_locales.append(locale)

    print(f"Found {len(available_locales)} local models: {sorted(available_locales)}")

    models_and_weights = load_similarity_weights(
        similarity_matrix_path, available_locales, args.target_lang,
        args.num_languages, args.top_k, args.sinkhorn_iters
    )

    if not models_and_weights:
        print("Could not load models. Aborting.")
        return

    # Load test data from MASSIVE dataset
    print(f"\n--- Loading Test Data for {args.target_lang} ---")
    dataset = load_dataset("AmazonScience/massive", f"{args.target_lang}", split="test", trust_remote_code=True)

    # Use all examples if num_examples is None, otherwise use specified amount
    if args.num_examples is None:
        test_texts = dataset["utt"]
        test_labels = dataset["intent"]
        print(f"Loaded all {len(test_texts)} test examples")
    else:
        # Dataset slicing returns a dict, so we need to access fields correctly
        subset = dataset[:args.num_examples]
        test_texts = subset["utt"]
        test_labels = subset["intent"]
        print(f"Loaded {len(test_texts)} test examples")

    # Run ensemble inference
    predictions, metadata = run_ensemble_inference(
        models_and_weights, test_texts, args.voting_method
    )

    # Evaluate predictions
    print(f"\n--- Evaluating Ensemble Performance ---")
    results = evaluate_ensemble(predictions, test_labels)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct_predictions']}/{results['total_predictions']}")
    print(f"Error rate: {results['error_rate']:.4f}")

    # Save results
    output_dir = os.path.join(project_root, args.output_dir, f"urie_{args.voting_method}_{args.target_lang}")
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    detailed_results = {
        "experiment_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "target_language": args.target_lang,
            "voting_method": args.voting_method,
            "num_models": len(models_and_weights),
            "num_examples": len(test_texts),
            "subfolder_pattern": args.subfolder_pattern
        },
        "models": models_and_weights,
        "metadata": metadata,
        "performance": results,
        "examples": [
            {
                "text": text,
                "true_label": true_label,
                "predicted_label": pred,
                "correct": true_label == pred
            }
            for text, true_label, pred in zip(test_texts, test_labels, predictions)
        ]
    }

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n*****************************************************")
    print("*                  Ensemble Finished                 *")
    print("*****************************************************")


if __name__ == "__main__":
    main()