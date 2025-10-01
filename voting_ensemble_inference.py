#!/usr/bin/env python3
"""
Voting Ensemble Inference Script

Instead of merging model weights, this script runs inference on multiple models
and combines their predictions through various voting mechanisms.
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
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
# --- End Path Setup ---

def map_locale_to_language_code(locale: str):
    """Map MASSIVE locale format to language code using model_mapping.csv."""
    try:
        model_mapping_path = os.path.join(project_root, "model_mapping.csv")
        if os.path.exists(model_mapping_path):
            model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)

            # If locale is already a language code, return as-is
            if locale in model_mapping_df.index:
                return locale

            # If locale is in MASSIVE format, find the corresponding language code
            if locale in model_mapping_df['locale'].values:
                # Find the row where locale matches and return the index (language code)
                language_code = model_mapping_df[model_mapping_df['locale'] == locale].index[0]
                print(f"Mapped locale '{locale}' to language code '{language_code}'")
                return language_code

        # If no mapping found, return original locale
        print(f"No mapping found for locale '{locale}', using as-is")
        return locale
    except Exception as e:
        print(f"Could not load locale mapping: {e}")
        return locale

def get_subfolder_for_language(locale: str, subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9"):
    """Generate subfolder pattern based on locale."""
    return subfolder_pattern.format(locale=locale)

def load_similarity_weights(sparsed_matrix_path: str, model_mapping_path: str, target_lang: str,
                          subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9", num_languages: int = None):
    """Load similarity weights and create model-to-weight mapping for target language."""
    try:
        # Load the sparsed (normalized) similarity matrix
        sparsed_df = pd.read_csv(sparsed_matrix_path, index_col=0)

        # Load model mapping
        model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)

        print(f"Loaded similarity matrix with shape: {sparsed_df.shape}")
        print(f"Loaded model mapping with {len(model_mapping_df)} models")
        print(f"Using subfolder pattern: {subfolder_pattern}")

        # Find the target language row
        if target_lang not in sparsed_df.index:
            print(f"Target language '{target_lang}' not found in similarity matrix")
            print(f"Available languages: {list(sparsed_df.index)}")
            return None

        # Get weights for target language
        target_weights = sparsed_df.loc[target_lang]

        # Filter languages with non-zero weights and valid model mappings
        valid_languages = []
        for lang, weight in target_weights.items():
            if weight > 0 and lang in model_mapping_df.index:
                valid_languages.append((lang, weight))

        # Sort by weight in descending order and limit to num_languages
        valid_languages.sort(key=lambda x: x[1], reverse=True)
        if num_languages is not None and num_languages > 0:
            valid_languages = valid_languages[:num_languages]
            print(f"Limiting to top {num_languages} languages by similarity weight")

        # Create model-to-weight mapping
        models_and_weights = {}
        for lang, weight in valid_languages:
            model_info = model_mapping_df.loc[lang]
            model_name = model_info['model_name']
            locale = model_info['locale']

            # Generate language-specific subfolder
            subfolder = get_subfolder_for_language(locale, subfolder_pattern)

            # Use the format expected by auto_merge_llm: model_name@subfolder
            model_with_subfolder = f"{model_name}@{subfolder}"
            models_and_weights[model_with_subfolder] = {
                'weight': weight,
                'subfolder': subfolder,
                'language': lang,
                'locale': locale,
                'base_model_name': model_name
            }
            print(f"  - {model_with_subfolder}: {weight:.6f} (language: {lang})")

        return models_and_weights

    except Exception as e:
        print(f"Error loading similarity weights: {e}")
        return None

def load_model_from_cache(model_name: str, subfolder: str = None):
    """Load model from HuggingFace cache with offline mode."""
    try:
        print(f"Loading model: {model_name} (subfolder: {subfolder})")

        # Try to construct the cache path
        cache_model_name = model_name.replace('/', '--')
        if subfolder:
            model_path = os.path.join(project_root, ".cache", "huggingface", "hub",
                                    f"models--{cache_model_name}", "snapshots", "*", subfolder)
        else:
            model_path = os.path.join(project_root, ".cache", "huggingface", "hub",
                                    f"models--{cache_model_name}")

        # First try the cache path
        if subfolder:
            # Find the actual path with wildcard
            import glob
            matching_paths = glob.glob(model_path)
            if matching_paths:
                actual_path = matching_paths[0]
                print(f"Loading from cache path: {actual_path}")
                model = AutoModelForSequenceClassification.from_pretrained(actual_path)
                tokenizer = AutoTokenizer.from_pretrained(actual_path)
                return model, tokenizer
            else:
                print(f"Cache path not found: {model_path}")

        # Fallback to normal loading with local_files_only
        print(f"Attempting to load with local_files_only=True")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, subfolder=subfolder, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, subfolder=subfolder, local_files_only=True
        )

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model {model_name}@{subfolder}: {e}")
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

def run_voting_ensemble(models_and_weights: Dict, test_data: List[str], voting_method: str = "majority") -> Tuple[List[int], Dict]:
    """
    Run voting ensemble inference on multiple models.

    Args:
        models_and_weights: Dictionary mapping model paths to their info
        test_data: List of text examples to classify
        voting_method: "majority", "weighted_majority", or "soft"

    Returns:
        Tuple of (predictions, metadata)
    """
    print(f"\n--- Running Voting Ensemble Inference (Method: {voting_method}) ---")
    print(f"Number of models: {len(models_and_weights)}")
    print(f"Number of test examples: {len(test_data)}")

    # Load all models
    models = []
    tokenizers = []
    model_names = []
    weights = []

    for model_path, model_info in models_and_weights.items():
        print(f"\nLoading model: {model_path}")
        model_name = model_info['base_model_name']
        subfolder = model_info['subfolder']
        weight = model_info['weight']

        model, tokenizer = load_model_from_cache(model_name, subfolder)

        if model is not None and tokenizer is not None:
            models.append(model)
            tokenizers.append(tokenizer)
            model_names.append(model_path)
            weights.append(weight)
            print(f"✓ Successfully loaded {model_path} (weight: {weight:.6f})")

            # Put model in eval mode
            model.eval()
        else:
            print(f"✗ Failed to load {model_path}")

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

    for i, text in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}")

        model_predictions = []
        model_probabilities = []

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

            except Exception as e:
                print(f"Error in model inference: {e}")
                # Use a default prediction if model fails
                model_predictions.append(0)
                model_probabilities.append(torch.zeros(60))  # Assuming 60 classes

        # Store individual model predictions
        all_predictions.append(model_predictions)
        all_probabilities.append(model_probabilities)

    # Apply voting to get final predictions
    print(f"\nApplying {voting_method} voting...")
    final_predictions = []

    for i, (predictions, probabilities) in enumerate(zip(all_predictions, all_probabilities)):
        if voting_method == "majority":
            final_pred = majority_vote(predictions)
        elif voting_method == "weighted_majority":
            final_pred = weighted_majority_vote(predictions, weights)
        elif voting_method == "soft":
            final_pred = soft_vote(probabilities, weights)
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
        "individual_probabilities": [[prob.tolist() for prob in probs] for probs in all_probabilities]
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
    parser = argparse.ArgumentParser(description="Voting Ensemble Inference")
    parser.add_argument(
        "--target-lang",
        type=str,
        required=True,
        help="Target language/locale (e.g., 'af-ZA', 'ar-SA')"
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
        choices=["majority", "weighted_majority", "soft"],
        default="weighted_majority",
        help="Voting method to use"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of test examples to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="voting_results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    print("*****************************************************")
    print("*        Voting Ensemble Inference Pipeline        *")
    print("*****************************************************")

    # Map target language
    target_lang_code = map_locale_to_language_code(args.target_lang)
    print(f"Target language: {args.target_lang} (mapped to: {target_lang_code})")

    # Load models and weights using similarity matrix
    sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix.csv")
    model_mapping_path = os.path.join(project_root, "model_mapping.csv")

    models_and_weights = load_similarity_weights(
        sparsed_matrix_path, model_mapping_path, target_lang_code,
        args.subfolder_pattern, args.num_languages
    )

    if not models_and_weights:
        print("Could not load models. Aborting.")
        return

    # Load test data from MASSIVE dataset
    print(f"\n--- Loading Test Data for {args.target_lang} ---")
    try:
        dataset = load_dataset("AmazonScience/massive", f"{args.target_lang}", split="test")
        test_texts = [example["utt"] for example in dataset[:args.num_examples]]
        test_labels = [example["intent"] for example in dataset[:args.num_examples]]
        print(f"Loaded {len(test_texts)} test examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to dummy data if dataset loading fails
        print("Using dummy test data")
        test_texts = ["Test text " + str(i) for i in range(args.num_examples)]
        test_labels = [0] * args.num_examples

    # Run voting ensemble
    predictions, metadata = run_voting_ensemble(
        models_and_weights, test_texts, args.voting_method
    )

    # Evaluate predictions
    print(f"\n--- Evaluating Ensemble Performance ---")
    results = evaluate_ensemble(predictions, test_labels)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct_predictions']}/{results['total_predictions']}")
    print(f"Error rate: {results['error_rate']:.4f}")

    # Save results
    output_dir = os.path.join(project_root, args.output_dir, f"voting_{args.voting_method}_{args.target_lang}")
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    detailed_results = {
        "experiment_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "target_language": args.target_lang,
            "target_lang_code": target_lang_code,
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
    print("*                Ensemble Finished                 *")
    print("*****************************************************")

if __name__ == "__main__":
    main()