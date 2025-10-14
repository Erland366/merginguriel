#!/usr/bin/env python3
"""
Simple test script to verify the similarity_utils module works correctly.
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.similarity_utils import load_and_process_similarity

def test_similarity_utils():
    """Test the similarity utilities module."""
    print("Testing similarity_utils module...")

    # Test with a few target languages
    test_languages = ["en-US", "fr-FR", "de-DE"]

    similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

    for target_lang in test_languages:
        print(f"\n--- Testing {target_lang} ---")
        try:
            # Test with 3 languages
            similar_languages = load_and_process_similarity(
                similarity_matrix_path,
                target_lang,
                num_languages=3,
                top_k=20,
                sinkhorn_iterations=20,
                verbose=True
            )

            print(f"✓ Successfully processed {target_lang}")
            print(f"  Found {len(similar_languages)} similar languages:")
            for locale, weight in similar_languages:
                print(f"    - {locale}: {weight:.6f}")

        except Exception as e:
            print(f"✗ Error processing {target_lang}: {e}")

    print("\n--- Testing model mapping integration ---")

    # Test integration with haryo models
    haryo_models_path = os.path.join(project_root, "haryoaw_k_models.csv")

    if os.path.exists(haryo_models_path):
        import pandas as pd
        haryo_models_df = pd.read_csv(haryo_models_path)

        print(f"Loaded {len(haryo_models_df)} models from haryoaw_k_models.csv")
        print(f"Available locales: {list(haryo_models_df['locale'].values)}")

        # Test finding models for multiple target languages
        for target_lang in ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT"]:
            similar_languages = load_and_process_similarity(
                similarity_matrix_path,
                target_lang,
                num_languages=15,  # Get more languages to find matches
                top_k=20,
                sinkhorn_iterations=20,
                verbose=False
            )

            print(f"\nTarget: {target_lang}")
            print("Similar languages that have available models:")

            found_models = []
            available_locales = set(haryo_models_df['locale'].values)

            for locale, weight in similar_languages:
                # Direct locale match - just like in the ensemble inference
                if locale in available_locales:
                    model_row = haryo_models_df[haryo_models_df['locale'] == locale].iloc[0]
                    found_models.append((locale, weight, model_row['model_name']))
                    print(f"  ✓ {locale}: {model_row['model_name']} (weight: {weight:.6f})")
                else:
                    print(f"  ✗ {locale}: No model available")

            print(f"\nFound {len(found_models)} available models for {target_lang}")

    else:
        print("✗ haryoaw_k_models.csv not found")

    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_similarity_utils()