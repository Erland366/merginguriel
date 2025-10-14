#!/usr/bin/env python3
"""
Integration tests for Merging Pipeline with similarity matrix processing.
"""

import os
import sys
import pytest
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.similarity_utils import (
    load_and_process_similarity,
    process_similarity_matrix,
    get_similarity_weights
)


class TestMergingPipelineIntegration:
    """Integration tests for merging pipeline similarity processing."""

    def test_similarity_matrix_loading(self):
        """Test loading and processing of the actual similarity matrix."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        if not os.path.exists(similarity_matrix_path):
            pytest.skip("Similarity matrix not found")

        # Test loading for a few languages
        test_languages = ["en-US", "fr-FR", "de-DE"]

        for target_lang in test_languages:
            result = load_and_process_similarity(
                similarity_matrix_path, target_lang,
                num_languages=5, top_k=10, sinkhorn_iterations=5,
                verbose=False
            )

            assert result is not None
            assert len(result) <= 5

            # Check weights are properly normalized (sum to 1.0)
            total_weight = sum(weight for _, weight in result)
            assert abs(total_weight - 1.0) < 1e-6

            # Check format
            for locale, weight in result:
                assert isinstance(locale, str)
                assert isinstance(weight, float)
                assert 0.0 <= weight <= 1.0

    def test_top_k_filtering(self):
        """Test top-K filtering with different values."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        if not os.path.exists(similarity_matrix_path):
            pytest.skip("Similarity matrix not found")

        target_lang = "en-US"
        top_k_values = [5, 10, 20]

        for top_k in top_k_values:
            result = load_and_process_similarity(
                similarity_matrix_path, target_lang,
                num_languages=10, top_k=top_k, sinkhorn_iterations=5,
                verbose=False
            )

            assert result is not None
            assert len(result) <= 10  # Limited by num_languages

            # Higher top_k should generally give us more options
            # (though this depends on model availability)

    def test_sinkhorn_normalization(self):
        """Test Sinkhorn normalization with different iterations."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        if not os.path.exists(similarity_matrix_path):
            pytest.skip("Similarity matrix not found")

        target_lang = "en-US"
        iterations = [1, 5, 10, 20]

        for iter_count in iterations:
            result = load_and_process_similarity(
                similarity_matrix_path, target_lang,
                num_languages=5, top_k=10, sinkhorn_iterations=iter_count,
                verbose=False
            )

            assert result is not None
            if result:  # Only check if we got results
                total_weight = sum(weight for _, weight in result)
                assert abs(total_weight - 1.0) < 1e-6

    def test_model_availability_filtering(self):
        """Test filtering based on available models."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")
        haryo_models_path = os.path.join(project_root, "haryoaw_k_models.csv")

        if not os.path.exists(similarity_matrix_path):
            pytest.skip("Similarity matrix not found")
        if not os.path.exists(haryo_models_path):
            pytest.skip("Haryo models list not found")

        # Load available models
        import pandas as pd
        haryo_models_df = pd.read_csv(haryo_models_path)
        available_locales = set(haryo_models_df['locale'].values)

        target_lang = "en-US"

        # Get all similar languages
        all_similar = load_and_process_similarity(
            similarity_matrix_path, target_lang,
            num_languages=50, top_k=50, sinkhorn_iterations=5,
            verbose=False
        )

        # Filter by available models
        available_similar = [(locale, weight) for locale, weight in all_similar
                            if locale in available_locales]

        # Should have some available models
        assert len(available_similar) > 0

        # Check that all found locales are actually available
        for locale, _ in available_similar:
            assert locale in available_locales

    def test_merging_pipeline_compatibility(self):
        """Test compatibility with merging pipeline expected format."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        if not os.path.exists(similarity_matrix_path):
            pytest.skip("Similarity matrix not found")

        target_lang = "en-US"
        available_locales = ["en-US", "de-DE", "fr-FR", "es-ES"]

        # Test the format expected by merging pipeline
        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang,
            num_languages=5, top_k=10, sinkhorn_iterations=5,
            verbose=False
        )

        # Convert to model-weight mapping format
        models_and_weights = {}
        available_locales_set = set(available_locales)

        for locale, weight in similar_languages:
            if locale in available_locales_set:
                models_and_weights[locale] = {
                    'weight': weight,
                    'locale': locale
                }

        # Should find at least one available model
        assert len(models_and_weights) > 0

        # Check weights are properly normalized
        total_weight = sum(info['weight'] for info in models_and_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_merged_model_structure(self):
        """Test that merged models have expected structure."""
        merged_models_dir = os.path.join(project_root, "merged_models")

        if not os.path.exists(merged_models_dir):
            pytest.skip("No merged models directory found")

        # Look for similarity merge results
        similarity_merge_dirs = [d for d in os.listdir(merged_models_dir)
                               if "similarity" in d]

        if not similarity_merge_dirs:
            pytest.skip("No similarity merge results found")

        # Check one of the merge results
        merge_dir = os.path.join(merged_models_dir, similarity_merge_dirs[0])
        assert os.path.isdir(merge_dir)

        # Should have merge details
        merge_details = os.path.join(merge_dir, "merge_details.txt")
        if os.path.exists(merge_details):
            with open(merge_details, 'r') as f:
                content = f.read()

            # Should contain key information
            assert "Merge Mode:" in content
            assert "Target Language:" in content
            assert "Total Weight:" in content

            # Extract and verify weights
            lines = content.split('\n')
            weights = []
            for line in lines:
                if 'Weight:' in line and 'Total Weight:' not in line and '%' in line:
                    try:
                        weight_str = line.split('Weight:')[1].strip().split(' ')[0]
                        weights.append(float(weight_str))
                    except (ValueError, IndexError):
                        continue

            if weights:  # Only check if we found weights
                total_weight = sum(weights)
                assert abs(total_weight - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])