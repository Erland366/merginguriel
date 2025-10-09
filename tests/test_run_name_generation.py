#!/usr/bin/env python
"""
Test script to verify the run name generation function works correctly.
"""

import sys
import os

# Add the current directory to the path so we can import the function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the run name generation function
from training_bert import generate_wandb_run_name

from dataclasses import dataclass

@dataclass
class MockModelArgs:
    model_name_or_path: str

@dataclass
class MockDataArgs:
    dataset_name: str
    dataset_config_name: str

@dataclass
class MockTrainingArgs:
    learning_rate: float
    num_train_epochs: int

def test_run_name_generation():
    """Test various configurations for run name generation."""

    print("=== Testing Run Name Generation ===\n")

    test_cases = [
        {
            "name": "Default RoBERTa configuration",
            "model_args": MockModelArgs("FacebookAI/roberta-base"),
            "data_args": MockDataArgs("AmazonScience/massive", "en-US"),
            "training_args": MockTrainingArgs(5e-5, 3),
            "expected": "roberta-base_massive_en-US_lr5e-5_ep3"
        },
        {
            "name": "BERT configuration with different LR",
            "model_args": MockModelArgs("bert-base-uncased"),
            "data_args": MockDataArgs("AmazonScience/massive", "en-US"),
            "training_args": MockTrainingArgs(3e-5, 5),
            "expected": "bert-base-uncased_massive_en-US_lr3e-5_ep5"
        },
        {
            "name": "French dataset configuration",
            "model_args": MockModelArgs("FacebookAI/roberta-base"),
            "data_args": MockDataArgs("AmazonScience/massive", "fr-FR"),
            "training_args": MockTrainingArgs(1e-4, 2),
            "expected": "roberta-base_massive_fr-FR_lr1e-4_ep2"
        },
        {
            "name": "Large learning rate",
            "model_args": MockModelArgs("bert-base-uncased"),
            "data_args": MockDataArgs("AmazonScience/massive", "en-US"),
            "training_args": MockTrainingArgs(0.01, 1),
            "expected": "bert-base-uncased_massive_en-US_lr0.0100_ep1"
        },
        {
            "name": "Zero learning rate (edge case)",
            "model_args": MockModelArgs("roberta-base"),
            "data_args": MockDataArgs("massive", "en-US"),
            "training_args": MockTrainingArgs(0.0, 1),
            "expected": "roberta-base_massive_en-US_lr0_ep1"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")

        try:
            generated_name = generate_wandb_run_name(
                test_case['model_args'],
                test_case['data_args'],
                test_case['training_args']
            )

            print(f"   Expected: {test_case['expected']}")
            print(f"   Generated: {generated_name}")

            if generated_name == test_case['expected']:
                print("   ✅ PASS")
            else:
                print("   ❌ FAIL")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print()

if __name__ == "__main__":
    test_run_name_generation()