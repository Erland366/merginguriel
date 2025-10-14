#!/usr/bin/env python3
"""
Script to generate realistic test data for testing the aggregate_results.py system.
"""

import json
import os
from datetime import datetime

def create_test_results_folder(folder_name, locale, experiment_type, accuracy, source_languages=None, weights=None):
    """Create a test results folder with results.json and merge_details.txt."""

    folder_path = os.path.join("results", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Create results.json
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_name': f'{experiment_type}_merge_{locale}',
            'subfolder': '',
            'locale': locale,
            'massive_locale': locale,
            'dataset': 'AmazonScience/massive',
            'split': 'test',
        },
        'model_info': {
            'num_labels': 60,
            'model_type': 'xlm-roberta',
            'architecture': 'XLMRobertaForSequenceClassification',
            'id2label': {str(i): f'intent_{i}' for i in range(60)},
            'label2id': {f'intent_{i}': str(i) for i in range(60)},
        },
        'dataset_info': {
            'total_examples': 1000,
            'unique_intents': 60,
            'intent_labels': [f'intent_{i}' for i in range(60)],
        },
        'performance': {
            'accuracy': accuracy,
            'correct_predictions': int(accuracy * 1000),
            'total_predictions': 1000,
            'error_rate': 1.0 - accuracy,
        },
        'examples': []
    }

    with open(os.path.join(folder_path, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Create merge_details.txt for merging experiments
    if experiment_type in ['similarity', 'average', 'fisher_simple', 'fisher_dataset']:
        merge_details = f"""Merge Mode: {experiment_type}
Timestamp (UTC): {datetime.utcnow().isoformat()}

Base Model (for architecture): xlm-roberta-base
Target Language: {locale}

--- Merged Models and Weights ---
"""

        if source_languages and weights:
            for i, (source_lang, weight) in enumerate(zip(source_languages, weights), 1):
                model_path = f"./haryos_model/xlm-roberta-base_massive_k_{source_lang}"
                merge_details += f"""{i}. Model: {model_path}
   - Locale: {source_lang}
   - Weight: {weight:.6f} ({weight*100:.2f}% of total)

"""

        with open(os.path.join(folder_path, "merge_details.txt"), 'w') as f:
            f.write(merge_details)

    print(f"Created test data in: {folder_path}")

def main():
    """Generate comprehensive test data."""

    # Clean up existing test data
    if os.path.exists("results"):
        import shutil
        shutil.rmtree("results")

    # Create test results for different locales and experiment types
    test_cases = [
        # Albanian (sq-AL)
        {
            'locale': 'sq-AL',
            'baseline_accuracy': 0.8234,
            'similarity_accuracy': 0.8456,
            'average_accuracy': 0.8312,
            'source_languages': ['hr-HR', 'bs-Latn', 'sr-Cyrl', 'mt-MT', 'sl-SI'],
            'similarity_weights': [0.35, 0.25, 0.20, 0.10, 0.10],
            'average_weights': [0.20, 0.20, 0.20, 0.20, 0.20]
        },
        # Thai (th-TH)
        {
            'locale': 'th-TH',
            'baseline_accuracy': 0.7890,
            'similarity_accuracy': 0.8123,
            'average_accuracy': 0.7967,
            'source_languages': ['lo-LA', 'km-KH', 'my-MM', 'vi-VN'],
            'similarity_weights': [0.30, 0.25, 0.25, 0.20],
            'average_weights': [0.25, 0.25, 0.25, 0.25]
        },
        # Croatian (hr-HR)
        {
            'locale': 'hr-HR',
            'baseline_accuracy': 0.8567,
            'similarity_accuracy': 0.8789,
            'average_accuracy': 0.8623,
            'source_languages': ['sl-SI', 'bs-Latn', 'sr-Cyrl', 'sq-AL'],
            'similarity_weights': [0.30, 0.25, 0.25, 0.20],
            'average_weights': [0.25, 0.25, 0.25, 0.25]
        }
    ]

    for case in test_cases:
        locale = case['locale']

        # Create baseline experiment
        create_test_results_folder(
            f"baseline_{locale}",
            locale,
            'baseline',
            case['baseline_accuracy']
        )

        # Create similarity experiment
        create_test_results_folder(
            f"similarity_{locale}",
            locale,
            'similarity',
            case['similarity_accuracy'],
            case['source_languages'],
            case['similarity_weights']
        )

        # Create average experiment
        create_test_results_folder(
            f"average_{locale}",
            locale,
            'average',
            case['average_accuracy'],
            case['source_languages'],
            case['average_weights']
        )

    # Create nxn_results directory with evaluation matrix
    os.makedirs("nxn_results/nxn_eval_test", exist_ok=True)

    # Copy sample evaluation matrix
    import shutil
    shutil.copy("test_data/sample_evaluation_matrix.csv",
                "nxn_results/nxn_eval_test/evaluation_matrix.csv")

    print("\n✅ Test data generation complete!")
    print("\n📁 Created test structure:")
    print("   results/")
    print("   ├── baseline_sq-AL/")
    print("   │   ├── results.json")
    print("   │   └── merge_details.txt (not created for baseline)")
    print("   ├── similarity_sq-AL/")
    print("   │   ├── results.json")
    print("   │   └── merge_details.txt")
    print("   ├── average_sq-AL/")
    print("   │   ├── results.json")
    print("   │   └── merge_details.txt")
    print("   └── ... (same for th-TH and hr-HR)")
    print("   ")
    print("   nxn_results/nxn_eval_test/")
    print("   └── evaluation_matrix.csv")

    print("\n🧪 Now you can test the system with:")
    print("   python merginguriel/aggregate_results.py --verbose")
    print("   python merginguriel/aggregate_results.py --evaluation-matrix test_data/sample_evaluation_matrix.csv")
    print("   python merginguriel/aggregate_results.py --locales sq-AL th-TH --verbose")

if __name__ == "__main__":
    main()