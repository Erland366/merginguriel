"""
Refactored Model Merging Pipeline (moved under merginguriel/)

A composable and extensible pipeline for merging language models using various strategies.
This version uses classes to make the code more modular and easier to extend.
"""

import os
import sys
from datetime import datetime
import subprocess
import numpy as np
import argparse
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Resolve repository root (one level up when this file is inside merginguriel/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from merginguriel.utils import get_similarity_scores
from merginguriel.similarity import sinkhorn_normalize, filter_top_k, locale_to_uriel_code
import lang2vec.lang2vec as l2v
from sklearn.metrics.pairwise import cosine_similarity
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
    top_k: int = 20
    sinkhorn_iters: int = 20
    # Fisher/dataset options
    fisher_data_mode: str = "target"  # {target|sources|both}
    preweight: str = "equal"  # {equal|uriel}
    batch_size: int = 16
    max_seq_length: int = 128


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
        print(f"\n--- Loading Similarity Weights for {target_lang} ---")
        if config.similarity_source == "dense":
            print("Using on-the-fly dense similarity with top-k + Sinkhorn normalization")
            df = self._compute_similarity_onfly(config.top_k, config.sinkhorn_iters)
            models_and_weights = self._build_mapping_from_df(df, target_lang, config.num_languages)
        else:
            sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix_unified.csv")
            models_and_weights = self._load_similarity_weights(
                sparsed_matrix_path, target_lang,
                config.subfolder_pattern, config.num_languages
            )

        if not models_and_weights:
            raise ValueError("Could not load similarity weights")

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

    
    def _load_similarity_weights(self, sparsed_matrix_path: str,
                                target_lang: str, subfolder_pattern: str, num_languages: int) -> Dict[str, ModelInfo]:
        """Load similarity weights and create model-to-weight mapping."""
        sparsed_df = pd.read_csv(sparsed_matrix_path, index_col=0)

        print(f"Loaded similarity matrix with shape: {sparsed_df.shape}")

        if target_lang not in sparsed_df.index:
            raise ValueError(f"Target language '{target_lang}' not found in similarity matrix")

        target_weights = sparsed_df.loc[target_lang]

        # Filter languages with non-zero weights
        valid_languages = []
        for locale, weight in target_weights.items():
            if weight > 0:
                valid_languages.append((locale, weight))

        # Sort by weight and limit
        valid_languages.sort(key=lambda x: x[1], reverse=True)
        if num_languages > 0:
            valid_languages = valid_languages[:num_languages]

        models_and_weights = {}
        for locale, weight in valid_languages:
            # Use consolidated model directory structure
            model_path = f"./haryos_model/xlm-roberta-base_massive_k_{locale}"

            models_and_weights[model_path] = ModelInfo(
                model_name=model_path,
                subfolder="",  # No subfolder needed with consolidated structure
                language=locale,
                locale=locale,
                weight=weight
            )
            print(f"  - {model_path}: {weight:.6f} (locale: {locale})")

        return models_and_weights

    def _compute_similarity_onfly(self, top_k: int, sinkhorn_iters: int) -> pd.DataFrame:
        """Compute dense similarity across locales on the fly, then sparsify + Sinkhorn.
        Returns a DataFrame indexed/columned by MASSIVE locales with normalized weights.
        """
        mapping_path = os.path.join(project_root, "model_mapping_unified.csv")
        map_df = pd.read_csv(mapping_path)
        if "locale" not in map_df.columns:
            raise ValueError(f"Missing 'locale' column in {mapping_path}")
        locales = map_df["locale"].tolist()
        uriel_codes = []
        kept_locales = []
        for loc in locales:
            code = locale_to_uriel_code(loc)
            if code:
                uriel_codes.append(code)
                kept_locales.append(loc)

        feats = l2v.get_features(uriel_codes, "syntax_knn")
        if isinstance(feats, dict):
            X = np.stack([feats[c] for c in uriel_codes])
        else:
            X = np.asarray(feats)

        sim = cosine_similarity(X)
        sim = (sim + 1.0) / 2.0
        sparse = filter_top_k(sim, top_k)
        norm = sinkhorn_normalize(sparse, iterations=sinkhorn_iters)
        df = pd.DataFrame(norm, index=kept_locales, columns=kept_locales)
        return df

    def _build_mapping_from_df(self, df: pd.DataFrame, target_lang: str, num_languages: int) -> Dict[str, ModelInfo]:
        if target_lang not in df.index:
            raise ValueError(f"Target language '{target_lang}' not found in computed similarity matrix")
        target_weights = df.loc[target_lang]
        valid = [(loc, float(w)) for loc, w in target_weights.items() if w > 0]
        valid.sort(key=lambda x: x[1], reverse=True)
        if num_languages > 0:
            valid = valid[:num_languages]

        # Renormalize to sum to 1 over the chosen subset
        s = sum(w for _, w in valid) or 1.0
        models_and_weights: Dict[str, ModelInfo] = {}
        for locale, w in valid:
            model_path = f"./haryos_model/xlm-roberta-base_massive_k_{locale}"
            models_and_weights[model_path] = ModelInfo(
                model_name=model_path,
                subfolder="",
                language=locale,
                locale=locale,
                weight=w / s,
            )
        return models_and_weights

    def _get_subfolder_for_language(self, locale: str, subfolder_pattern: str) -> str:
        """Generate subfolder pattern based on locale."""
        return subfolder_pattern.format(locale=locale)


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

        sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix_unified.csv")

        models_and_weights = self._load_similarity_weights(
            sparsed_matrix_path, target_lang,
            config.subfolder_pattern, config.num_languages
        )

        if not models_and_weights:
            raise ValueError("No models found for the target language")

        # Use first model as base
        first_model_key = list(models_and_weights.keys())[0]
        base_model_info = models_and_weights[first_model_key]
        models_and_weights.pop(first_model_key)

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
            'fisher_dataset': SimilarityWeightCalculator,
            'iterative': IterativeWeightCalculator,
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


class MergingStrategyFactory:
    @staticmethod
    def create(mode: str) -> MergingStrategy:
        if mode in {"fisher", "fisher_simple"}:
            return FisherSimpleStrategy()
        if mode == "fisher_dataset":
            return FisherDatasetStrategy()
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

        # Prepare model paths and weights
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [info.weight for info in models_and_weights.values()]

        # Set up method parameters via strategy
        method_params = self.strategy.get_method_params(self.config, models_and_weights, base_model_info)

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

    # _get_method_params removed in favor of strategy pattern


class OutputManager:
    """Manages saving models and results."""

    def __init__(self, project_root: str):
        self.project_root = project_root

    def save_model_and_details(self, merged_model: Any, tokenizer: Any, config: MergeConfig,
                              models_and_weights: Dict[str, ModelInfo], base_model_info: ModelInfo):
        """Save the merged model and merge details."""
        output_dir = os.path.join(self.project_root, "merged_models",
                                 f"{config.mode}_merge_{config.target_lang}")

        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully to: {output_dir}")

        self._save_merge_details(output_dir, config, models_and_weights, base_model_info)

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

    def __init__(self, project_root: str):
        self.project_root = project_root

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

    def __init__(self, config: MergeConfig):
        self.config = config
        self.project_root = project_root
        self.weight_calculator = WeightCalculatorFactory.create_calculator(config.mode)
        self.model_merger = ModelMerger(config)
        self.output_manager = OutputManager(self.project_root)
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
        self.output_manager.save_model_and_details(
            merged_model, tokenizer, self.config, models_and_weights, base_model_info
        )

        # Step 4: Evaluate
        output_dir = os.path.join(self.project_root, "merged_models",
                                 f"{self.config.mode}_merge_{self.config.target_lang}")
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
        base_model="xlm-roberta-base",
        similarity_source=args.similarity_source,
        top_k=args.top_k,
        sinkhorn_iters=args.sinkhorn_iters,
        fisher_data_mode=args.fisher_data_mode,
        preweight=args.preweight,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A composable pipeline to merge models using various strategies.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['uriel', 'manual', 'similarity', 'average', 'fisher', 'fisher_simple', 'fisher_dataset', 'iterative'],
        help="The merging mode to use."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="sq-AL",
        help="Target language/locale for similarity-based merging (e.g., sq-AL, th-TH, af-ZA)"
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
        help="Number of languages to include in merging"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name for Fisher merging"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="utt",
        help="Column name containing text data (MASSIVE uses 'utt')"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name containing labels"
    )
    parser.add_argument(
        "--num-fisher-examples",
        type=int,
        default=100,
        help="Number of examples to use for Fisher computation"
    )
    parser.add_argument(
        "--similarity-source",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        help="Use precomputed sparse CSV or compute dense similarities on-the-fly with top-k + Sinkhorn"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K neighbors to preserve per language when computing similarities on-the-fly"
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Sinkhorn normalization iterations for on-the-fly similarity"
    )
    parser.add_argument(
        "--fisher-data-mode",
        type=str,
        choices=["target", "sources", "both"],
        default="target",
        help="Which data distribution to compute Fisher on: target locale only, the selected source locales, or both"
    )
    parser.add_argument(
        "--preweight",
        type=str,
        choices=["equal", "uriel"],
        default="equal",
        help="Pre-weight models before Fisher merging: equal or URIEL cosine weights"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dataset-enabled Fisher computation"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Max sequence length for tokenization in Fisher computation"
    )

    args = parser.parse_args()
    config = create_config_from_args(args)

    pipeline = MergingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
