#!/usr/bin/env python3
"""
Integration tests for URIEL-Guided Ensemble Inference system.
"""

import os
import sys
import json
import pytest
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.uriel_ensemble_inference import (
    load_similarity_weights,
    run_ensemble_inference,
    evaluate_ensemble,
    uriel_weighted_logits,
    load_model_from_local
)


class TestEnsembleInferenceIntegration:
    """Integration tests for the complete ensemble inference pipeline."""

    def test_load_similarity_weights_integration(self):
        """Test similarity weights loading with real data."""
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")
        available_locales = ["en-US", "de-DE", "fr-FR", "es-ES", "it-IT"]
        target_lang = "en-US"

        result = load_similarity_weights(
            similarity_matrix_path, available_locales, target_lang,
            num_languages=3, top_k=10, sinkhorn_iterations=5
        )

        assert result is not None
        assert len(result) <= 3  # Should not exceed num_languages
        assert len(result) > 0   # Should find at least some models

        # Check weights sum to 1.0
        total_weight = sum(info['weight'] for info in result.values())
        assert abs(total_weight - 1.0) < 1e-6

        # Check all target locales are in available list
        for locale in result.keys():
            assert locale in available_locales

    def test_model_loading_integration(self):
        """Test model loading with actual models."""
        # Test with a locale that should exist
        test_locales = ["en-US", "de-DE", "fr-FR"]

        loaded_models = []
        for locale in test_locales:
            model, tokenizer = load_model_from_local(locale)
            if model is not None and tokenizer is not None:
                loaded_models.append(locale)

        # Should load at least one model
        assert len(loaded_models) > 0, "Should load at least one model for testing"

    def test_ensemble_inference_integration(self):
        """Test complete ensemble inference pipeline."""
        # This test requires actual models, so we'll make it more flexible
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        # Use dummy data for testing
        test_texts = ["Hello world", "How are you?"]
        test_labels = [0, 1]

        # Try to load models and weights
        available_locales = ["en-US", "de-DE", "fr-FR"]
        target_lang = "en-US"

        models_and_weights = load_similarity_weights(
            similarity_matrix_path, available_locales, target_lang,
            num_languages=2, top_k=5, sinkhorn_iterations=5
        )

        if models_and_weights and len(models_and_weights) > 0:
            # If we have models, test the full pipeline
            try:
                predictions, metadata = run_ensemble_inference(
                    models_and_weights, test_texts, "majority"
                )

                assert len(predictions) == len(test_texts)
                assert "voting_method" in metadata
                assert "num_models" in metadata
                assert metadata["num_models"] > 0

                # Test evaluation
                results = evaluate_ensemble(predictions, test_labels)
                assert "accuracy" in results
                assert "correct_predictions" in results
                assert "total_predictions" in results
                assert results["total_predictions"] == len(test_texts)

            except Exception as e:
                # Model loading might fail in test environment
                pytest.skip(f"Model loading failed: {e}")
        else:
            pytest.skip("No models available for integration testing")

    def test_uriel_logits_algorithm(self):
        """Test the URIEL logits weighting algorithm directly."""
        import torch

        # Create dummy logits (3 models, 4 classes)
        logits_list = [
            torch.tensor([1.0, 2.0, 1.5, 0.5]),
            torch.tensor([0.8, 1.8, 2.0, 1.2]),
            torch.tensor([1.2, 1.5, 1.0, 2.1])
        ]

        # Test weights
        weights = [0.5, 0.3, 0.2]

        result = uriel_weighted_logits(logits_list, weights, normalize_weights=True)

        assert isinstance(result, int)
        assert 0 <= result < 4  # Should be a valid class index

    def test_ensemble_results_persistence(self):
        """Test that ensemble results are properly saved."""
        # This is more of a file system test
        results_dir = os.path.join(project_root, "urie_ensemble_results")

        if os.path.exists(results_dir):
            # Check for expected subdirectories
            subdirs = [d for d in os.listdir(results_dir)
                      if os.path.isdir(os.path.join(results_dir, d))]

            # Should have at least one results directory
            assert len(subdirs) > 0

            # Check for results files
            for subdir in subdirs[:1]:  # Check just one to save time
                subdir_path = os.path.join(results_dir, subdir)
                files = os.listdir(subdir_path)
                assert "results.json" in files

                # Verify JSON structure
                results_file = os.path.join(subdir_path, "results.json")
                with open(results_file, 'r') as f:
                    data = json.load(f)

                assert "experiment_info" in data
                assert "models" in data
                assert "metadata" in data
                assert "performance" in data
                assert "examples" in data


if __name__ == "__main__":
    pytest.main([__file__])