#!/usr/bin/env python3
"""
Ensemble Runner Module

This module provides functions to run individual ensemble experiments
and can be imported by other scripts like the comparison runner.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.uriel_ensemble_inference import run_ensemble_inference, map_locale_to_language_code, load_similarity_weights, evaluate_ensemble


def run_single_ensemble_experiment(target_lang: str,
                                 voting_method: str = "uriel_logits",
                                 num_examples: int = 100,
                                 num_languages: int = 5,
                                 output_dir: str = "ensemble_results",
                                 subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9") -> Dict[str, Any]:
    """
    Run a single ensemble experiment and return detailed results.

    Args:
        target_lang: Target language/locale
        voting_method: Voting method to use
        num_examples: Number of test examples
        num_languages: Number of source models to include
        output_dir: Output directory
        subfolder_pattern: Model subfolder pattern

    Returns:
        Dictionary containing detailed experiment results
    """
    from datasets import load_dataset

    # Map target language
    target_lang_code = map_locale_to_language_code(target_lang)

    # Load models and weights using similarity matrix
    sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix.csv")
    model_mapping_path = os.path.join(project_root, "model_mapping.csv")

    models_and_weights = load_similarity_weights(
        sparsed_matrix_path, model_mapping_path, target_lang_code,
        subfolder_pattern, num_languages
    )

    if not models_and_weights:
        raise ValueError(f"Could not load models for target language {target_lang}")

    # Load test data from MASSIVE dataset
    dataset = load_dataset("AmazonScience/massive", f"{target_lang}", split="test", trust_remote_code=True)

    # Dataset slicing returns a dict, so we need to access fields correctly
    subset = dataset[:num_examples]
    test_texts = subset["utt"]
    test_labels = subset["intent"]
    dataset_source = "massive"

    # Run ensemble inference
    predictions, metadata = run_ensemble_inference(
        models_and_weights, test_texts, voting_method
    )

    # Evaluate predictions
    results = evaluate_ensemble(predictions, test_labels)

    # Prepare detailed results
    detailed_results = {
        "experiment_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "target_language": target_lang,
            "target_lang_code": target_lang_code,
            "voting_method": voting_method,
            "num_models": len(models_and_weights),
            "num_examples": len(test_texts),
            "subfolder_pattern": subfolder_pattern,
            "dataset_source": dataset_source
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

    # Save results to file
    output_path = os.path.join(project_root, output_dir, f"urie_{voting_method}_{target_lang}")
    os.makedirs(output_path, exist_ok=True)

    results_file = os.path.join(output_path, "results.json")
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    return detailed_results


if __name__ == "__main__":
    # Test function
    result = run_single_ensemble_experiment(
        target_lang="en-US",
        voting_method="uriel_logits",
        num_examples=10,
        num_languages=3
    )
    print("Test experiment completed!")
    print(f"Accuracy: {result['performance']['accuracy']:.4f}")