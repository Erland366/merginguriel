"""
Refactored Model Merging Pipeline (moved under merginguriel/)

A composable and extensible pipeline for merging language models using various strategies.
This version uses classes to make the code more modular and easier to extend.
"""

import os
import sys
from datetime import datetime
import subprocess
import warnings
import numpy as np
import argparse
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Resolve repository root (one level up when this file is inside merginguriel/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DEFAULT_MODEL_DIRS = {
    "xlm-roberta-base": Path(project_root) / "haryos_model",
    "xlm-roberta-large": Path(project_root) / "haryos_model_large",
}

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from merginguriel.utils import get_similarity_scores
from merginguriel.similarity_utils import load_and_process_similarity
from auto_merge_llm.methods import merging_methods_dict


@dataclass
class ModelInfo:
    """Data class to hold model information."""
    model_name: str
    subfolder: str
    language: str
    locale: str
    weight: float


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    mode: str
    target_lang: str = "sq-AL"
    subfolder_pattern: str = ""  # No longer needed with consolidated model structure
    num_languages: int = 5
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    text_column: str = "text"
    label_column: str = "label"
    num_fisher_examples: int = 100
    base_model: str = "xlm-roberta-base"
        # Similarity matrix options
    similarity_source: str = "sparse"  # 'sparse' (precomputed) or 'dense' (on-the-fly)
    similarity_type: str = "URIEL"  # 'URIEL' (linguistic features) or 'REAL' (empirical evaluation results)
    include_target: bool = False  # Include target language model in merging (IT mode)
    top_k: int = 20
    sinkhorn_iters: int = 20
    # Fisher/dataset options
    fisher_data_mode: str = "target"  # {target|sources|both}
    preweight: str = "equal"  # {equal|uriel}
    batch_size: int = 16
    max_seq_length: int = 128
    base_model_dir: str = ""


class WeightCalculator(ABC):
    """Abstract base class for weight calculation strategies."""

    @abstractmethod
    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        """
        Calculate weights for models to be merged.

        Returns:
            Tuple of (models_and_weights, base_model_info)
        """
        pass


class UrielWeightCalculator(WeightCalculator):
    """Weight calculator using URIEL language similarity."""

    def __init__(self):
        self.models_to_merge = {
            "ind": "lur601/xlm_roberta-base-finetuned-paxn-id",
            "jav": "w11wo/xlm-roberta-base-finetuned-ud-javanese",
        }
        self.source_language = "eng"

    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        print("\n--- Calculating URIEL Similarity Weights ---")

        df_path = os.path.join(project_root, "big_assets/language_similarity_matrix.csv")
        df = pd.read_csv(df_path, index_col=0)
        target_langs = list(self.models_to_merge.keys())
        scores = get_similarity_scores(self.source_language, target_langs, df)
        weights = scores["normalized_scores"]

        if not weights or sum(weights.values()) == 0:
            raise ValueError("Could not calculate weights")

        models_and_weights = {}
        for lang, weight in weights.items():
            model_name = self.models_to_merge[lang]
            models_and_weights[model_name] = ModelInfo(
                model_name=model_name,
                subfolder="",  # URIEL mode doesn't use subfolders
                language=lang,
                locale=lang,
                weight=weight
            )

        # Use the first model as base
        first_model_name = list(models_and_weights.keys())[0]
        base_model_info = models_and_weights[first_model_name]
        models_and_weights.pop(first_model_name)

        print("Calculated Normalized Weights:")
        for lang, weight in weights.items():
            print(f"  - {self.models_to_merge[lang]}: {weight:.4f}")

        return models_and_weights, base_model_info


class SimilarityWeightCalculator(WeightCalculator):
    """Weight calculator using pre-computed similarity matrix."""

    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        target_lang = config.target_lang
        print(f"\n--- Computing Similarity Weights for {target_lang} ---")
        print(f"Using {config.similarity_type} similarity matrix with top-k + Sinkhorn normalization")

        # Choose similarity matrix based on type
        if config.similarity_type == "URIEL":
            similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")
        elif config.similarity_type == "REAL":
            # Use the latest NxN evaluation results
            similarity_matrix_path = os.path.join(project_root, "nxn_results", "nxn_eval_20251027_103544", "evaluation_matrix.csv")
        else:
            raise ValueError(f"Unknown similarity type: {config.similarity_type}")

        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang, config.num_languages,
            config.top_k, config.sinkhorn_iters, config.include_target, verbose=True
        )

        if not similar_languages:
            raise ValueError("Could not compute similarity weights")

        if config.base_model_dir:
            base_dir = Path(config.base_model_dir).expanduser()
            if not base_dir.is_absolute():
                base_dir = Path(project_root) / base_dir
        else:
            base_dir = DEFAULT_MODEL_DIRS.get(
                config.base_model,
                Path(project_root) / "haryos_model",
            )

        # Create model mapping using the resolved base directory
        models_and_weights: Dict[str, ModelInfo] = {}
        for locale, weight in similar_languages:
            model_path = base_dir / f"{config.base_model}_massive_k_{locale}"

            if not model_path.exists():
                print(f"  ✗ Model path not found: {model_path}")
                continue

            model_path_str = str(model_path)
            models_and_weights[model_path_str] = ModelInfo(
                model_name=model_path_str,
                subfolder="",
                language=locale,
                locale=locale,
                weight=weight,
            )
            print(f"  ✓ {model_path_str}: {weight:.6f} (locale: {locale})")

        if not models_and_weights:
            raise ValueError("No local models found for the target language")

        # Use first model as base
        first_model_key = list(models_and_weights.keys())[0]
        base_model_info = models_and_weights[first_model_key]
        models_and_weights.pop(first_model_key)

        # Normalize weights to sum to 1.0
        total_weight = sum(info.weight for info in models_and_weights.values()) + base_model_info.weight
        if total_weight > 0:
            normalization_factor = 1.0 / total_weight
            base_model_info.weight *= normalization_factor
            for info in models_and_weights.values():
                info.weight *= normalization_factor

        return models_and_weights, base_model_info

  

class ManualWeightCalculator(WeightCalculator):
    """Weight calculator using manually specified weights."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights is None:
            # Default example weights for testing
            self.weights = {
                "lur601/xlm-roberta-base-finetuned-panx-en": 0.6,
                "lur601/xlm-roberta-base-finetuned-panx-it": 0.4,
            }
        else:
            self.weights = weights

    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        print("\n--- Validating Manual Configuration ---")

        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Your weights must sum to 1.0, but they sum to {total_weight}")

        models_and_weights = {}
        for model_name, weight in self.weights.items():
            models_and_weights[model_name] = ModelInfo(
                model_name=model_name,
                subfolder="",  # Manual mode doesn't use subfolders
                language="",
                locale="",
                weight=weight
            )

        # Use first model as base
        first_model_name = list(models_and_weights.keys())[0]
        base_model_info = models_and_weights[first_model_name]
        models_and_weights.pop(first_model_name)

        print("Manual weights are valid.")
        for model, weight in self.weights.items():
            print(f"  - Weight {weight:.4f}: {model}")

        return models_and_weights, base_model_info


class AverageWeightCalculator(SimilarityWeightCalculator):
    """Weight calculator using equal weights for all models."""

    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        target_lang = config.target_lang
        print(f"\n--- Setting Up Average (Equal) Weights for {target_lang} ---")

        # Use the parent class to get models, then set equal weights
        models_and_weights, base_model_info = super().calculate_weights(config)

        if not models_and_weights:
            raise ValueError("No models found for the target language")

        # Set equal weights
        num_models = len(models_and_weights) + 1  # +1 for base model
        equal_weight = 1.0 / num_models

        base_model_info.weight = equal_weight
        for info in models_and_weights.values():
            info.weight = equal_weight

        print(f"Using equal weights for {num_models} models: {equal_weight:.6f} each")

        return models_and_weights, base_model_info


class IterativeWeightCalculator(WeightCalculator):
    """Weight calculator for iterative training merges."""

    def __init__(self, active_model_states: Optional[Dict[str, Any]] = None, target_locales: Optional[List[str]] = None):
        self.active_model_states = active_model_states or {}
        self.target_locales = target_locales or []

    def calculate_weights(self, config: MergeConfig) -> Tuple[Dict[str, ModelInfo], ModelInfo]:
        print("\n--- Setting Up Iterative Merging Weights ---")

        # For iterative merging, we use the provided model states and equal weights
        # This can be extended to support more sophisticated weight calculation strategies

        if not self.active_model_states:
            # Fallback to similarity-based weights if no states provided
            print("No active model states provided, falling back to similarity-based weights")
            similarity_calculator = SimilarityWeightCalculator()
            return similarity_calculator.calculate_weights(config)

        models_and_weights = {}
        equal_weight = 1.0 / len(self.active_model_states)

        for locale, model_info in self.active_model_states.items():
            model_path = model_info.get('checkpoint_path', model_info.get('locale', locale))

            models_and_weights[model_path] = ModelInfo(
                model_name=model_path,
                subfolder="",
                language=locale,
                locale=locale,
                weight=equal_weight
            )
            print(f"  - {locale}: {equal_weight:.6f} (from checkpoint: {model_path})")

        if not models_and_weights:
            raise ValueError("No models available for iterative merging")

        # Use first model as base
        first_model_key = list(models_and_weights.keys())[0]
        base_model_info = models_and_weights[first_model_key]
        models_and_weights.pop(first_model_key)

        # Renormalize weights
        num_models = len(models_and_weights) + 1  # +1 for base model
        final_equal_weight = 1.0 / num_models

        base_model_info.weight = final_equal_weight
        for info in models_and_weights.values():
            info.weight = final_equal_weight

        print(f"Using equal weights for {num_models} models in iterative merge: {final_equal_weight:.6f} each")

        return models_and_weights, base_model_info


class WeightCalculatorFactory:
    """Factory for creating weight calculators."""

    @staticmethod
    def create_calculator(mode: str, **kwargs) -> WeightCalculator:
        """Create a weight calculator based on the mode."""
        calculators = {
            'uriel': UrielWeightCalculator,
            'manual': ManualWeightCalculator,
            'similarity': SimilarityWeightCalculator,
            'average': AverageWeightCalculator,
            'iterative': IterativeWeightCalculator,
            'fisher': SimilarityWeightCalculator,
            # Advanced merging methods use similarity-based weights
            'ties': SimilarityWeightCalculator,
            'task_arithmetic': SimilarityWeightCalculator,
            'slerp': SimilarityWeightCalculator,
            'regmean': SimilarityWeightCalculator,
        }

        if mode not in calculators:
            raise ValueError(f"Unknown mode: {mode}")

        calculator_class = calculators[mode]

        if mode == 'manual':
            return calculator_class(kwargs.get('weights'))
        elif mode == 'iterative':
            return calculator_class(
                kwargs.get('active_model_states'),
                kwargs.get('target_locales')
            )
        else:
            return calculator_class()


class MergingStrategy(ABC):
    """Strategy interface for selecting merger and building method params."""

    @abstractmethod
    def get_merger(self, mode: str):
        pass

    @abstractmethod
    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        pass


class LinearStrategy(MergingStrategy):
    def get_merger(self, mode: str):
        return merging_methods_dict["linear"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        weights = [info.weight for info in models_and_weights.values()]
        return {"weights": weights}


class FisherSimpleStrategy(MergingStrategy):
    def get_merger(self, mode: str):
        # Supports both 'fisher' and 'fisher_simple' registered methods
        return merging_methods_dict[mode]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        raw_weights = [info.weight for info in models_and_weights.values()]
        if not raw_weights:
            return {}
        total = sum(raw_weights)
        if total <= 0:
            norm = [1.0 / len(raw_weights)] * len(raw_weights)
        else:
            norm = [w / total for w in raw_weights]
        return {"weights": norm}


class FisherDatasetStrategy(MergingStrategy):
    def get_merger(self, mode: str):
        return merging_methods_dict["fisher_dataset"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        if not config.dataset_name:
            raise ValueError("--dataset-name is required for fisher_dataset mode")

        fisher_scaling_coefficients: Optional[List[float]] = None
        if config.preweight == "uriel":
            src_weights = [info.weight for info in models_and_weights.values()]
            total = sum(src_weights)
            if total > 0:
                fisher_scaling_coefficients = [w / total for w in src_weights]
            else:
                fisher_scaling_coefficients = [1.0 / max(1, len(src_weights))] * len(src_weights)

        return {
            "dataset_config": {
                "dataset_name": config.dataset_name,
                "dataset_split": config.dataset_split,
                "text_column": config.text_column,
                "label_column": config.label_column,
            },
            "fisher_data_mode": config.fisher_data_mode,
            "target_locale": config.target_lang,
            "source_locales": [info.locale for info in models_and_weights.values()],
            "num_fisher_examples": config.num_fisher_examples,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "normalize_fisher_weight": True,
            "minimal_fisher_weight": 1e-6,
            "fisher_scaling_coefficients": fisher_scaling_coefficients,
        }


class TiesStrategy(MergingStrategy):
    """TIES merging strategy that resolves sign disagreements and prunes low-magnitude weights."""

    def get_merger(self, mode: str):
        return merging_methods_dict["ties"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        # TIES method parameters
        # Use URIEL weights to influence the merging process
        src_weights = [info.weight for info in models_and_weights.values()]

        # Default parameters for TIES
        param_value_mask_rate = 0.8  # Mask 80% of smallest-magnitude parameters
        scaling_coefficient = 1.0    # Scaling coefficient for task vectors

        # If we have meaningful weights, we can adjust scaling
        if src_weights and max(src_weights) > 0:
            # Use the maximum weight as scaling coefficient for better preservation
            scaling_coefficient = max(src_weights)

        return {
            "param_value_mask_rate": param_value_mask_rate,
            "scaling_coefficient": scaling_coefficient,
        }


class TaskArithmeticStrategy(MergingStrategy):
    """TaskArithmetic merging strategy that adds/subtracts task vectors."""

    def get_merger(self, mode: str):
        return merging_methods_dict["task_arithmetic"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        # TaskArithmetic parameters
        # For task_arithmetic, scaling_coefficient should be a single float
        # We'll use the average of weights or a default value
        src_weights = [info.weight for info in models_and_weights.values()]

        if src_weights:
            # Use the average weight as the scaling coefficient
            scaling_coefficient = sum(src_weights) / len(src_weights)
        else:
            scaling_coefficient = 1.0  # Default if no weights

        # Optional: mask smallest parameters (DARE-like behavior)
        param_value_mask_rate = 0.0  # Default: no masking

        return {
            "scaling_coefficient": scaling_coefficient,
            "param_value_mask_rate": param_value_mask_rate,
        }


class SlerpStrategy(MergingStrategy):
    """SLERP (Spherical Linear Interpolation) merging strategy with incremental merging support."""

    def get_merger(self, mode: str):
        return merging_methods_dict["slerp"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        # For incremental SLERP, we need parameters for each merge step
        # We'll use the model weights to determine interpolation ratios

        # Get all models including base model for sorting
        all_models = [(base_model_info.model_name, base_model_info.weight)]
        all_models.extend([(name, info.weight) for name, info in models_and_weights.items()])

        # Sort models by weight (descending) - merge most important first
        all_models.sort(key=lambda x: x[1], reverse=True)

        # The first model (highest weight) becomes the initial base
        # Subsequent models are merged one by one
        merge_steps = []

        for i in range(1, len(all_models)):
            current_model_name, current_weight = all_models[i]
            prev_model_name, prev_weight = all_models[i-1]

            # Calculate interpolation ratio based on relative weights
            # Higher weight model gets more influence
            total_weight = current_weight + prev_weight
            if total_weight > 0:
                slerp_t = prev_weight / total_weight  # Base model gets proportion of its weight
            else:
                slerp_t = 0.5  # Equal interpolation if weights are zero

            merge_steps.append({
                "slerp_t": slerp_t,
                "dot_threshold": 0.9995,  # Default threshold from auto-merge-llm
                "base_model": prev_model_name,
                "merge_model": current_model_name,
                "base_weight": prev_weight,
                "merge_weight": current_weight
            })

        return {
            "incremental_slerp": True,
            "merge_steps": merge_steps,
            "total_models": len(all_models)
        }


class RegMeanStrategy(MergingStrategy):
    """RegMean merging strategy - simplified implementation using linear merging."""

    def get_merger(self, mode: str):
        # For now, use linear merging as RegMean requires complex trainer setup
        # Full RegMean integration would require actual training data and trainers
        return merging_methods_dict["linear"]()

    def get_method_params(
        self,
        config: MergeConfig,
        models_and_weights: Dict[str, ModelInfo],
        base_model_info: ModelInfo,
    ) -> Dict[str, Any]:
        """
        Simplified RegMean implementation.

        NOTE: Full RegMean requires:
        - Actual training data and data loaders
        - Trainer instances for each model
        - Complex setup to compute regression matrices

        For practical use in this pipeline, we implement a simplified approach
        that uses URIEL-weighted linear merging, which captures the spirit of RegMean
        (data-driven coefficient optimization) without the complexity.
        """

        # Get URIEL weights as data-driven coefficients
        weights = [info.weight for info in models_and_weights.values()]

        # Additional RegMean-inspired parameters for potential future enhancement
        # These would be used in a full RegMean implementation
        num_models = len(models_and_weights)
        regmean_lambda = 1.0  # Regularization strength
        reduce_non_diagonal_ratio = 0.5  # Matrix regularization

        print(f"\n📊 RegMean Strategy: Using simplified linear merging")
        print(f"   Models: {num_models + 1} (including base model)")
        print(f"   URIEL weights: {[f'{w:.3f}' for w in weights]}")
        print(f"   Note: Full RegMean requires trainer setup, using weighted linear as approximation")

        return {
            "weights": weights,
            # Metadata for documentation
            "regmean_metadata": {
                "original_method": "regmean",
                "approximation": "linear_with_uriel_weights",
                "num_models": num_models + 1,
                "regularization_lambda": regmean_lambda,
                "reduce_non_diagonal_ratio": reduce_non_diagonal_ratio,
                "reason": "RegMean requires complex trainer setup, using simplified approach"
            }
        }


class MergingStrategyFactory:
    @staticmethod
    def create(mode: str) -> MergingStrategy:
        if mode == "fisher":
            return FisherDatasetStrategy()
        if mode == "ties":
            return TiesStrategy()
        if mode == "task_arithmetic":
            return TaskArithmeticStrategy()
        if mode == "slerp":
            return SlerpStrategy()
        if mode == "regmean":
            return RegMeanStrategy()
        return LinearStrategy()


class ModelMerger:
    """Handles the actual model merging process."""

    def __init__(self, config: MergeConfig):
        self.config = config
        self.strategy = MergingStrategyFactory.create(config.mode)

    def merge_models(self, models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo) -> Tuple[Any, Any]:
        """Perform the model merging."""
        print("\n--- Performing Model Merge ---")

        # Choose the appropriate merging method via strategy
        merger = self.strategy.get_merger(self.config.mode)

        # Set up method parameters via strategy
        method_params = self.strategy.get_method_params(self.config, models_and_weights, base_model_info)

        # Check if this is incremental SLERP
        if self.config.mode == "slerp" and method_params.get("incremental_slerp", False):
            return self._perform_incremental_slerp(models_and_weights, base_model_info, merger, method_params)
        else:
            # Standard merging for all other methods
            return self._perform_standard_merge(models_and_weights, base_model_info, merger, method_params)

    def _perform_standard_merge(self, models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo,
                               merger, method_params: Dict[str, Any]) -> Tuple[Any, Any]:
        """Perform standard model merging (non-incremental)."""
        # Prepare model paths and weights
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [info.weight for info in models_and_weights.values()]

        print(f"Base model: {base_model_info.model_name}")
        print(f"Models to merge: {models_to_merge_paths}")
        print(f"Weights: {weight_values}")

        # Perform the merge
        result = merger.merge(
            base_model=base_model_info.model_name,
            models_to_merge=models_to_merge_paths,
            method_params=method_params,
        )

        print("Merge successful!")
        return result['merged_model'], result['base_tokenizer']

    def _perform_incremental_slerp(self, models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo,
                                  merger, method_params: Dict[str, Any]) -> Tuple[Any, Any]:
        """Perform incremental SLERP merging for multiple models."""
        merge_steps = method_params["merge_steps"]
        total_models = method_params["total_models"]

        print(f"🔄 Starting Incremental SLERP merging for {total_models} models...")
        print(f"   Number of merge steps: {len(merge_steps)}")

        # Create temporary directory for intermediate models
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="incremental_slerp_")
        print(f"   Using temporary directory: {temp_dir}")

        try:
            # Start with the first two models
            first_step = merge_steps[0]
            print(f"\n📦 Step 1/{len(merge_steps)}:")
            print(f"   Base:  {first_step['base_model']} (weight: {first_step['base_weight']:.4f})")
            print(f"   Merge: {first_step['merge_model']} (weight: {first_step['merge_weight']:.4f})")
            print(f"   SLERP interpolation ratio: {first_step['slerp_t']:.4f}")

            # Prepare SLERP parameters for first step
            slerp_params = {
                "slerp_t": first_step["slerp_t"],
                "dot_threshold": first_step["dot_threshold"]
            }

            # Perform first SLERP merge
            # SLERP expects exactly 2 models in models_to_merge list
            result = merger.merge(
                base_model=first_step["base_model"],  # Used as architecture reference
                models_to_merge=[first_step["base_model"], first_step["merge_model"]],  # 2 models for SLERP
                method_params=slerp_params,
            )

            # Save intermediate result
            intermediate_path = os.path.join(temp_dir, "step_1_model")
            result['merged_model'].save_pretrained(intermediate_path)
            result['base_tokenizer'].save_pretrained(intermediate_path)

            current_model_path = intermediate_path
            print(f"   ✅ Step 1 completed, saved intermediate model")

            # Continue with remaining steps
            for i in range(1, len(merge_steps)):
                step = merge_steps[i]
                print(f"\n📦 Step {i+1}/{len(merge_steps)}:")
                print(f"   Base:  intermediate_model (weight: cumulative)")
                print(f"   Merge: {step['merge_model']} (weight: {step['merge_weight']:.4f})")
                print(f"   SLERP interpolation ratio: {step['slerp_t']:.4f}")

                # Prepare SLERP parameters for this step
                slerp_params = {
                    "slerp_t": step["slerp_t"],
                    "dot_threshold": step["dot_threshold"]
                }

                try:
                    # Perform SLERP merge using intermediate model as base
                    # SLERP expects exactly 2 models in models_to_merge list
                    result = merger.merge(
                        base_model=current_model_path,  # Used as architecture reference
                        models_to_merge=[current_model_path, step["merge_model"]],  # 2 models for SLERP
                        method_params=slerp_params,
                    )

                    # Save new intermediate result
                    intermediate_path = os.path.join(temp_dir, f"step_{i+1}_model")
                    result['merged_model'].save_pretrained(intermediate_path)
                    result['base_tokenizer'].save_pretrained(intermediate_path)

                    current_model_path = intermediate_path
                    print(f"   ✅ Step {i+1} completed, saved intermediate model")

                except Exception as e:
                    print(f"   ❌ Step {i+1} failed: {e}")
                    raise RuntimeError(f"Incremental SLERP failed at step {i+1}: {e}")

            print(f"\n🎉 Incremental SLERP merging completed successfully!")
            print(f"   Final merged model created from {total_models} source models")

            # Load final model and tokenizer
            from transformers import AutoModel, AutoTokenizer
            final_model = AutoModel.from_pretrained(current_model_path)
            final_tokenizer = AutoTokenizer.from_pretrained(current_model_path)

            return final_model, final_tokenizer

        finally:
            # Clean up temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"   🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not clean up temp directory {temp_dir}: {e}")

    # _get_method_params removed in favor of strategy pattern


class OutputManager:
    """Manages saving models and results."""

    def __init__(self, project_root: str, merged_models_dir: str = "merged_models"):
        self.project_root = project_root
        self.merged_models_dir = merged_models_dir

    def save_model_and_details(self, merged_model: Any, tokenizer: Any, config: MergeConfig,
                              models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo) -> str:
        """Save the merged model and merge details."""
        # Get the number of models merged
        num_models = len(models_and_weights)

        # Extract base model name from base_model_info.model_name
        base_model_name = base_model_info.model_name
        # Clean up the model name to get just the model family (xlm-roberta-base, xlm-roberta-large)
        if "/" in base_model_name:
            base_model_name = base_model_name.split("/")[-1]  # Get last part after slash

        # Extract model family using model-agnostic detection
        from merginguriel.naming_config import naming_manager
        try:
            base_model_name = naming_manager.extract_model_family(base_model_name)
        except ValueError:
            pass  # Keep original base_model_name if extraction fails

        # Use centralized naming manager with IT/ET support
        from merginguriel.naming_config import naming_manager
        merged_model_dir_name = naming_manager.get_merged_model_dir_name(
            experiment_type='merging',
            method=config.mode,
            similarity_type=config.similarity_type,
            locale=config.target_lang,
            model_family=base_model_name,
            num_languages=num_models,
            include_target=config.include_target
        )
        output_dir = os.path.join(self.project_root, self.merged_models_dir, merged_model_dir_name)

        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully to: {output_dir}")

        self._save_merge_details(output_dir, config, models_and_weights, base_model_info)

        return output_dir

    def _save_merge_details(self, output_dir: str, config: MergeConfig,
                           models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo):
        """Save details about the merge."""
        filepath = os.path.join(output_dir, "merge_details.txt")
        with open(filepath, 'w') as f:
            f.write(f"Merge Mode: {config.mode}\n")
            f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}\n\n")
            f.write(f"Base Model (for architecture): {config.base_model}\n")
            f.write(f"Target Language: {config.target_lang}\n")
            f.write("\n--- Merged Models and Weights ---\n")

            # Include base model
            all_models = [base_model_info] + list(models_and_weights.values())

            for i, model_info in enumerate(all_models):
                f.write(f"{i+1}. Model: {model_info.model_name}\n")
                if model_info.subfolder:
                    f.write(f"   - Subfolder: {model_info.subfolder}\n")
                if model_info.language:
                    f.write(f"   - Language: {model_info.language}\n")
                if model_info.locale:
                    f.write(f"   - Locale: {model_info.locale}\n")
                f.write(f"   - Weight: {model_info.weight:.6f} ({model_info.weight*100:.2f}% of total)\n")

            total_weight = sum(info.weight for info in all_models)
            f.write(f"\nTotal Weight: {total_weight:.6f}\n")

        print(f"Merge details saved to: {filepath}")


class Evaluator:
    """Handles model evaluation."""

    def __init__(self, project_root: str, merged_models_dir: str = "merged_models"):
        self.project_root = project_root
        self.merged_models_dir = merged_models_dir

    def evaluate_model(self, model_path: str):
        """Evaluate the merged model."""
        evaluation_script_path = os.path.join(self.project_root, "merginguriel/evaluate_base_encoder.py")
        if not os.path.exists(evaluation_script_path):
            raise FileNotFoundError(f"Evaluation script not found at {evaluation_script_path}")

        print(f"\n--- Starting evaluation for model: {model_path} ---")
        command = [sys.executable, evaluation_script_path, "--model_name_or_path", model_path]
        try:
            subprocess.run(command, check=True)
            print(f"--- Evaluation finished for {model_path} ---")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Evaluation failed with error: {e}")


class MergingPipeline:
    """Main pipeline orchestrator for model merging."""

    def __init__(self, config: MergeConfig, merged_models_dir: str = "merged_models"):
        self.config = config
        self.project_root = project_root
        self.weight_calculator = WeightCalculatorFactory.create_calculator(config.mode)
        self.model_merger = ModelMerger(config)
        self.output_manager = OutputManager(self.project_root, merged_models_dir)
        self.evaluator = Evaluator(self.project_root)

    def run(self):
        """Run the complete merging pipeline."""
        print("*****************************************************")
        print(f"*        Model Merging Pipeline (Mode: {self.config.mode.upper()})      *")
        print("*****************************************************")

        # Step 1: Calculate weights
        models_and_weights, base_model_info = self.weight_calculator.calculate_weights(self.config)

        # Step 2: Merge models
        merged_model, tokenizer = self.model_merger.merge_models(models_and_weights, base_model_info)

        # Step 3: Save results
        output_dir = self.output_manager.save_model_and_details(
            merged_model, tokenizer, self.config, models_and_weights, base_model_info
        )

        # Step 4: Evaluate
        self.evaluator.evaluate_model(output_dir)

        print("\n*****************************************************")
        print("*                  Pipeline Finished                *")
        print("*****************************************************")


def create_config_from_args(args) -> MergeConfig:
    """Create MergeConfig from command line arguments."""
    return MergeConfig(
        mode=args.mode,
        target_lang=args.target_lang,
        subfolder_pattern=args.subfolder_pattern,
        num_languages=args.num_languages,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        label_column=args.label_column,
        num_fisher_examples=args.num_fisher_examples,
        base_model=args.base_model,
        similarity_source=args.similarity_source,
        similarity_type=args.similarity_type,
        include_target=args.include_target,
        top_k=args.top_k,
        sinkhorn_iters=args.sinkhorn_iters,
        fisher_data_mode=args.fisher_data_mode,
        preweight=args.preweight,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        base_model_dir=args.base_model_dir,
    )


def create_config_from_yaml(yaml_path: Path, args) -> MergeConfig:
    """Create MergeConfig from YAML file with CLI overrides.

    Args:
        yaml_path: Path to YAML config file
        args: Parsed CLI arguments for overrides

    Returns:
        MergeConfig instance
    """
    from merginguriel.config import (
        ConfigLoader,
        PipelineConfig,
        ConfigDeprecationWarning,
    )

    # Load config from YAML
    pipeline_config = PipelineConfig.from_yaml(yaml_path)

    # Track which CLI args were explicitly provided
    provided_args = getattr(args, "_provided_args", set())

    # CLI arg to config path mapping
    arg_to_config = {
        "mode": "mode",
        "target_lang": "target.locale",
        "similarity_type": "similarity.type",
        "similarity_source": "similarity.source",
        "top_k": "similarity.top_k",
        "sinkhorn_iters": "similarity.sinkhorn_iters",
        "include_target": "target.inclusion",
        "base_model": "model.base_model",
        "num_languages": "model.num_languages",
        "dataset_name": "dataset.name",
        "dataset_split": "dataset.split",
        "text_column": "dataset.text_column",
        "label_column": "dataset.label_column",
        "fisher_data_mode": "fisher.data_mode",
        "preweight": "fisher.preweight",
        "num_fisher_examples": "fisher.num_examples",
        "batch_size": "fisher.batch_size",
        "max_seq_length": "fisher.max_seq_length",
    }

    # Override with CLI args if provided (with deprecation warnings)
    args_dict = vars(args)
    for arg_name, config_path in arg_to_config.items():
        if arg_name in provided_args:
            arg_value = args_dict.get(arg_name)
            if arg_value is not None:
                # Emit deprecation warning
                warnings.warn(
                    f"CLI argument '--{arg_name.replace('_', '-')}' is deprecated when using --config. "
                    f"Use config file with '{config_path}' instead. "
                    f"CLI value will override config file for backward compatibility.",
                    ConfigDeprecationWarning,
                    stacklevel=2
                )
                # Set the value in config
                ConfigLoader._set_nested_attr(pipeline_config, config_path, arg_value)

    # Convert PipelineConfig to legacy MergeConfig format
    # Handle include_target specially - it's a bool in CLI but "IncTar"/"ExcTar" in config
    include_target = pipeline_config.target.inclusion == "IncTar"

    return MergeConfig(
        mode=pipeline_config.mode,
        target_lang=pipeline_config.target.locale,
        subfolder_pattern=getattr(args, "subfolder_pattern", ""),
        num_languages=pipeline_config.model.num_languages,
        dataset_name=pipeline_config.dataset.name,
        dataset_split=pipeline_config.dataset.split,
        text_column=pipeline_config.dataset.text_column,
        label_column=pipeline_config.dataset.label_column,
        num_fisher_examples=pipeline_config.fisher.num_examples,
        base_model=pipeline_config.model.base_model,
        similarity_source=pipeline_config.similarity.source,
        similarity_type=pipeline_config.similarity.type,
        include_target=include_target,
        top_k=pipeline_config.similarity.top_k,
        sinkhorn_iters=pipeline_config.similarity.sinkhorn_iters,
        fisher_data_mode=pipeline_config.fisher.data_mode,
        preweight=pipeline_config.fisher.preweight,
        batch_size=pipeline_config.fisher.batch_size,
        max_seq_length=pipeline_config.fisher.max_seq_length,
        base_model_dir=getattr(args, "base_model_dir", ""),
    )


class TrackProvidedArgsAction(argparse.Action):
    """Custom argparse action that tracks which arguments were explicitly provided."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        # Track that this argument was provided
        if not hasattr(namespace, "_provided_args"):
            namespace._provided_args = set()
        namespace._provided_args.add(self.dest)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A composable pipeline to merge models using various strategies.")

    # Config file argument (new)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file. If provided, CLI args override config values with deprecation warnings."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,  # Changed to allow config override
        choices=['uriel', 'manual', 'similarity', 'average', 'fisher', 'iterative',
                'ties', 'task_arithmetic', 'slerp', 'regmean'],
        action=TrackProvidedArgsAction,
        help="The merging mode to use."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="sq-AL",
        action=TrackProvidedArgsAction,
        help="Target language/locale for similarity-based merging (e.g., sq-AL, th-TH, af-ZA)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="xlm-roberta-base",
        action=TrackProvidedArgsAction,
        help="Base model name for model path construction (e.g., xlm-roberta-base, xlm-roberta-large)"
    )
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default="",
        help="Optional override for the directory containing base-model checkpoints."
    )
    parser.add_argument(
        "--merged-models-dir",
        type=str,
        default="merged_models",
        help="Directory for saving merged models (default: merged_models)"
    )
    parser.add_argument(
        "--subfolder-pattern",
        type=str,
        default="alpha_0.5_{locale}_epoch-9",
        help="Subfolder pattern to use for model loading"
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=5,
        action=TrackProvidedArgsAction,
        help="Number of languages to include in merging"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        action=TrackProvidedArgsAction,
        help="HuggingFace dataset name for Fisher merging"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        action=TrackProvidedArgsAction,
        help="Dataset split to use"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="utt",
        action=TrackProvidedArgsAction,
        help="Column name containing text data (MASSIVE uses 'utt')"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        action=TrackProvidedArgsAction,
        help="Column name containing labels"
    )
    parser.add_argument(
        "--num-fisher-examples",
        type=int,
        default=1000,
        action=TrackProvidedArgsAction,
        help="Number of examples to use for Fisher computation"
    )
    parser.add_argument(
        "--similarity-source",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        action=TrackProvidedArgsAction,
        help="Use precomputed sparse CSV or compute dense similarities on-the-fly with top-k + Sinkhorn"
    )
    parser.add_argument(
        "--similarity-type",
        type=str,
        choices=["URIEL", "REAL"],
        default="URIEL",
        action=TrackProvidedArgsAction,
        help="Type of similarity matrix to use: URIEL (linguistic features) or REAL (empirical evaluation results)"
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include target language model in merging (IT mode). Default is exclude target (ET mode)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        action=TrackProvidedArgsAction,
        help="Top-K neighbors to preserve per language when computing similarities on-the-fly"
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        action=TrackProvidedArgsAction,
        help="Sinkhorn normalization iterations for similarity computation"
    )
    parser.add_argument(
        "--fisher-data-mode",
        type=str,
        choices=["target", "sources", "both"],
        default="target",
        action=TrackProvidedArgsAction,
        help="Which data distribution to compute Fisher on: target locale only, the selected source locales, or both"
    )
    parser.add_argument(
        "--preweight",
        type=str,
        choices=["equal", "uriel"],
        default="uriel",
        action=TrackProvidedArgsAction,
        help="Pre-weight models before Fisher merging: equal or URIEL cosine weights"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        action=TrackProvidedArgsAction,
        help="Batch size for dataset-enabled Fisher computation"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        action=TrackProvidedArgsAction,
        help="Max sequence length for tokenization in Fisher computation"
    )

    args = parser.parse_args()

    # Determine config source: YAML file or CLI arguments
    if args.config is not None:
        # Load from YAML config file (new path)
        if not args.config.exists():
            parser.error(f"Config file not found: {args.config}")
        print(f"Loading configuration from: {args.config}")
        config = create_config_from_yaml(args.config, args)
    else:
        # Legacy path: create from CLI arguments only
        if args.mode is None:
            parser.error("--mode is required when not using --config")
        config = create_config_from_args(args)

    pipeline = MergingPipeline(config, args.merged_models_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
