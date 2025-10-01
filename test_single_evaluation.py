#!/usr/bin/env python3
"""
Test script to debug a single evaluation before running the full NxN evaluation.
"""

import sys
import torch
from evaluate_specific_model import evaluate_specific_model

def test_single_evaluation():
    """Test a single model evaluation to debug CUDA issues."""

    # Test parameters - using a simple example
    model_name = "haryoaw/xlm-roberta-base_massive_k_en-US"
    model_dir = "alpha_0.5_en-US_epoch-9"
    locale = "en-US"

    print(f"Testing single evaluation:")
    print(f"Model: {model_name}")
    print(f"Model dir: {model_dir}")
    print(f"Locale: {locale}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test evaluation
    try:
        results = evaluate_specific_model(
            model_name=model_name,
            subfolder=model_dir,
            locale=locale,
            eval_folder="test_results"
        )

        if results:
            print(f"✓ Evaluation successful!")
            print(f"Accuracy: {results['performance']['accuracy']:.4f}")
        else:
            print("✗ Evaluation failed")

    except Exception as e:
        print(f"✗ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_evaluation()