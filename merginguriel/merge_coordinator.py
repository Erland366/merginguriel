"""
Merge coordination for iterative training and merging.

This module provides the MergeCoordinator class that handles the coordination
of merge operations during training, including scheduling, execution, and
state redistribution.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from merginguriel.training_state import ModelState, TrainingStateManager
from merginguriel.iterative_config import IterativeMergeConfig
from merginguriel.run_merging_pipeline_refactored import (
    MergingPipeline, MergeConfig, WeightCalculatorFactory, ModelMerger
)
from auto_merge_llm.methods import merging_methods_dict

logger = logging.getLogger(__name__)


class MergeCoordinator:
    """
    Coordinates merge operations during iterative training.

    This class handles:
    - Scheduling merge operations
    - Executing merges using the existing pipeline
    - Managing state redistribution after merges
    - Error handling and recovery
    """

    def __init__(
        self,
        merge_config: IterativeMergeConfig,
        training_state_manager: TrainingStateManager,
        base_output_dir: str
    ):
        self.merge_config = merge_config
        self.state_manager = training_state_manager
        self.base_output_dir = base_output_dir

        # Merge execution state
        self.merge_in_progress = False
        self.merge_lock = threading.Lock()
        self.merge_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.merge_start_time: Optional[float] = None
        self.total_merge_time: float = 0.0
        self.successful_merges: int = 0
        self.failed_merges: int = 0

    def should_merge(
        self,
        current_epoch: int,
        current_step: int,
        metrics_history: List[Dict[str, float]],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Determine if a merge should be triggered based on configuration and conditions.

        Note: Merging always occurs per epoch for consistent training dynamics.

        Args:
            current_epoch: Current training epoch
            current_step: Current training step (ignored for merge timing)
            metrics_history: History of training metrics
            performance_metrics: Current performance metrics

        Returns:
            True if merge should be triggered
        """
        # Check if merge is already in progress
        if self.merge_in_progress:
            return False

        # Epoch-based merging (always per epoch)
        if current_epoch % self.merge_config.merge_frequency == 0 and current_epoch > 0:
            logger.info(f"Merge triggered by epoch frequency: epoch {current_epoch}")
            return True

        # Performance-based merging (can still trigger between scheduled epochs)
        if self.merge_config.performance_merge_trigger and performance_metrics:
            if self._should_merge_based_on_performance(metrics_history, performance_metrics):
                logger.info("Merge triggered by performance metrics")
                return True

        return False

    def _should_merge_based_on_performance(
        self,
        metrics_history: List[Dict[str, float]],
        current_metrics: Dict[str, float]
    ) -> bool:
        """Determine if merge should be triggered based on performance stagnation."""
        if len(metrics_history) < 3:
            return False

        # Check for performance plateau in validation accuracy
        recent_accuracies = [
            m.get('eval_accuracy', 0) for m in metrics_history[-3:]
            if 'eval_accuracy' in m
        ]

        if len(recent_accuracies) < 3:
            return False

        # Calculate improvement rate
        improvements = [
            recent_accuracies[i+1] - recent_accuracies[i]
            for i in range(len(recent_accuracies)-1)
        ]

        avg_improvement = np.mean(improvements)
        return avg_improvement < self.merge_config.convergence_threshold

    def execute_merge(
        self,
        active_locales: List[str],
        target_locales: List[str],
        merge_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute a merge operation using current model states.

        Args:
            active_locales: List of locales currently being trained
            target_locales: Target locales for the merge
            merge_metadata: Additional metadata for the merge

        Returns:
            Tuple of (success, merged_model_path)
        """
        with self.merge_lock:
            if self.merge_in_progress:
                logger.warning("Merge already in progress, skipping")
                return False, None

            self.merge_in_progress = True
            self.merge_start_time = time.time()

        try:
            logger.info(f"Starting merge operation for locales: {active_locales}")

            # Collect current model states
            model_states = self._collect_model_states(active_locales)
            if not model_states:
                raise ValueError("No valid model states found for merging")

            # Create checkpoints before merge
            if self.merge_config.checkpoint_before_merge:
                self._create_pre_merge_checkpoints(model_states)

            # Execute the merge using existing pipeline
            merged_model_path = self._perform_merge(model_states, target_locales)

            if merged_model_path:
                # Redistribute merged weights back to active models
                self._redistribute_merged_weights(merged_model_path, active_locales)

                # Record successful merge
                self._record_merge_success(active_locales, target_locales, merge_metadata)
                return True, merged_model_path
            else:
                raise RuntimeError("Merge operation returned None")

        except Exception as e:
            logger.error(f"Merge operation failed: {e}")
            self._record_merge_failure(active_locales, target_locales, str(e))
            return False, None

        finally:
            with self.merge_lock:
                self.merge_in_progress = False
                if self.merge_start_time:
                    merge_duration = time.time() - self.merge_start_time
                    self.total_merge_time += merge_duration
                    logger.info(f"Merge operation completed in {merge_duration:.2f} seconds")

    def _collect_model_states(self, locales: List[str]) -> Dict[str, ModelState]:
        """Collect current model states for the specified locales."""
        model_states = {}

        for locale in locales:
            state = self.state_manager.get_state(locale)
            if state and state.model_state_dict:
                # Validate state integrity
                if state.validate_integrity():
                    model_states[locale] = state
                else:
                    logger.warning(f"Skipping {locale} due to state integrity check failure")
            else:
                logger.warning(f"No valid state found for locale: {locale}")

        return model_states

    def _create_pre_merge_checkpoints(self, model_states: Dict[str, ModelState]):
        """Create checkpoints before performing the merge."""
        logger.info("Creating pre-merge checkpoints")

        for locale, state in model_states.items():
            try:
                checkpoint_path = self.state_manager.create_checkpoint(
                    locale,
                    metadata={
                        "merge_type": "pre_merge",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"Created pre-merge checkpoint for {locale}: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to create pre-merge checkpoint for {locale}: {e}")

    def _perform_merge(
        self,
        model_states: Dict[str, ModelState],
        target_locales: List[str]
    ) -> Optional[str]:
        """Perform the actual merge using the existing pipeline."""
        try:
            # Create temporary merge configuration
            merge_config = self._create_merge_config(target_locales)

            # Create model merger
            model_merger = ModelMerger(merge_config)

            # Prepare model paths and weights for merging
            models_and_weights, base_model_info = self._prepare_merge_inputs(
                model_states, target_locales
            )

            if not models_and_weights:
                raise ValueError("No models prepared for merging")

            # Execute merge
            merged_model, tokenizer = model_merger.merge_models(
                models_and_weights, base_model_info
            )

            # Save merged model
            merged_model_path = self._save_merged_model(
                merged_model, tokenizer, target_locales
            )

            return merged_model_path

        except Exception as e:
            logger.error(f"Merge execution failed: {e}")
            return None

    def _create_merge_config(self, target_locales: List[str]) -> MergeConfig:
        """Create a MergeConfig for the current merge operation."""
        return MergeConfig(
            mode=self.merge_config.merge_algorithm,
            target_lang=target_locales[0] if target_locales else "unknown",
            num_languages=self.merge_config.num_languages,
            similarity_source=self.merge_config.similarity_source,
            top_k=self.merge_config.top_k,
            sinkhorn_iters=self.merge_config.sinkhorn_iters,
            fisher_data_mode=self.merge_config.fisher_data_mode,
            preweight=self.merge_config.preweight,
            num_fisher_examples=self.merge_config.num_fisher_examples,
            batch_size=self.merge_config.fisher_batch_size,
            max_seq_length=self.merge_config.fisher_max_seq_length
        )

    def _prepare_merge_inputs(
        self,
        model_states: Dict[str, ModelState],
        target_locales: List[str]
    ) -> Tuple[Dict[str, Any], Any]:
        """Prepare model paths and weights for the merge operation."""
        from merginguriel.run_merging_pipeline_refactored import ModelInfo
        from merginguriel.similarity_utils import load_and_process_similarity

        logger.info("Preparing merge inputs using cosine similarity matrix")

        # Use the same logic as run_merging_pipeline_refactored.py
        # Calculate similarity weights for the target locales
        target_lang = target_locales[0] if target_locales else list(model_states.keys())[0]

        # Load similarity matrix and compute weights
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

        try:
            similar_languages = load_and_process_similarity(
                similarity_matrix_path, target_lang, self.merge_config.num_languages,
                self.merge_config.top_k, self.merge_config.sinkhorn_iters, verbose=False
            )

            # Create a mapping from locale to similarity weight
            similarity_weights = {locale: weight for locale, weight in similar_languages}
            logger.info(f"Computed similarity weights for {len(similarity_weights)} languages")

        except Exception as e:
            logger.warning(f"Failed to compute similarity weights: {e}. Using equal weights.")
            similarity_weights = {}

        # Prepare models using haryos_model structure like in the main pipeline
        actual_models_and_weights = {}

        for locale, state in model_states.items():
            # Use haryos_model path structure like in run_merging_pipeline_refactored.py (line 148)
            model_path = f"/home/coder/Python_project/MergingUriel/haryos_model/xlm-roberta-base_massive_k_{locale}"

            # Check if model exists locally
            if not os.path.exists(model_path):
                logger.warning(f"  ✗ Model path not found: {model_path}")
                # Fallback to checkpoint path if available
                if state.checkpoint_path and os.path.exists(state.checkpoint_path):
                    model_path = state.checkpoint_path
                    logger.info(f"  ✓ Using checkpoint path: {model_path}")
                else:
                    logger.warning(f"  ✗ No valid model found for {locale}")
                    continue
            else:
                logger.info(f"  ✓ Found model: {model_path}")

            # Use similarity weight if available, otherwise equal weight
            if similarity_weights and locale in similarity_weights:
                weight = similarity_weights[locale]
                logger.info(f"  - {locale}: similarity weight {weight:.6f}")
            else:
                weight = 1.0 / len(model_states)  # Equal weight fallback
                logger.info(f"  - {locale}: equal weight {weight:.6f}")

            actual_models_and_weights[model_path] = ModelInfo(
                model_name=model_path,
                subfolder="",  # No subfolder needed with consolidated structure
                language=locale,
                locale=locale,
                weight=weight
            )

        if not actual_models_and_weights:
            raise ValueError(f"No valid models found for locales: {list(model_states.keys())}")

        # Set base model (use first model like in main pipeline)
        base_key = list(actual_models_and_weights.keys())[0]
        base_model_info = actual_models_and_weights[base_key]
        actual_models_and_weights.pop(base_key)

        # Normalize weights to sum to 1.0 (like in main pipeline)
        total_weight = sum(info.weight for info in actual_models_and_weights.values()) + base_model_info.weight
        if total_weight > 0:
            normalization_factor = 1.0 / total_weight
            base_model_info.weight *= normalization_factor
            for info in actual_models_and_weights.values():
                info.weight *= normalization_factor

        logger.info(f"Using {len(actual_models_and_weights) + 1} models for merge")
        logger.info(f"Base model: {base_model_info.model_name} (weight: {base_model_info.weight:.6f})")

        return actual_models_and_weights, base_model_info

    def _prepare_equal_weight_merge(
        self,
        model_states: Dict[str, ModelState]
    ) -> Tuple[Dict[str, Any], Any]:
        """Prepare merge inputs with equal weights as fallback."""
        from merginguriel.run_merging_pipeline_refactored import ModelInfo

        models_and_weights = {}
        equal_weight = 1.0 / len(model_states)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        for locale, state in model_states.items():
            # Use the same path resolution logic as in _prepare_merge_inputs
            if state.checkpoint_path and os.path.exists(state.checkpoint_path):
                model_path = state.checkpoint_path
            else:
                # Fallback to haryos_model structure
                model_path = os.path.join(project_root, "haryos_model", f"xlm-roberta-base_massive_k_{locale}")

                # If haryos_model doesn't exist, use the checkpoint path or locale
                if not os.path.exists(model_path):
                    model_path = state.checkpoint_path or locale

            models_and_weights[model_path] = ModelInfo(
                model_name=model_path,
                subfolder="",  # No subfolder needed with consolidated structure
                language=locale,
                locale=locale,
                weight=equal_weight
            )

        # Set base model
        if models_and_weights:
            base_key = list(models_and_weights.keys())[0]
            base_model_info = models_and_weights[base_key]
            models_and_weights.pop(base_key)
            return models_and_weights, base_model_info
        else:
            raise ValueError("No models available for equal weight merge")

    def _save_merged_model(
        self,
        merged_model: Any,
        tokenizer: Any,
        target_locales: List[str]
    ) -> str:
        """Save the merged model to disk."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_suffix = "_".join(target_locales[:3])  # Limit suffix length
        output_dir = os.path.join(
            self.merge_config.merge_output_base_dir,
            f"iterative_merge_{target_suffix}_{timestamp}"
        )

        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save merge details
        self._save_merge_details(output_dir, target_locales)

        logger.info(f"Merged model saved to: {output_dir}")
        return output_dir

    def _save_merge_details(self, output_dir: str, target_locales: List[str]):
        """Save details about the merge operation."""
        details = {
            "merge_type": "iterative",
            "timestamp": datetime.utcnow().isoformat(),
            "target_locales": target_locales,
            "merge_algorithm": self.merge_config.merge_algorithm,
            "weight_calculation": self.merge_config.weight_calculation,
            "merge_frequency": self.merge_config.merge_frequency,
            "successful_merges_so_far": self.successful_merges,
            "total_merge_time_so_far": self.total_merge_time
        }

        details_path = os.path.join(output_dir, "iterative_merge_details.json")
        import json
        with open(details_path, 'w') as f:
            json.dump(details, f, indent=2)

    def _redistribute_merged_weights(self, merged_model_path: str, active_locales: List[str]):
        """Redistribute merged weights back to the active training models."""
        try:
            logger.info(f"Redistributing merged weights from {merged_model_path} to {active_locales}")

            # Load the merged model
            merged_model = AutoModelForSequenceClassification.from_pretrained(merged_model_path)
            merged_state_dict = merged_model.state_dict()

            # Update each active model's state with merged weights
            for locale in active_locales:
                state = self.state_manager.get_state(locale)
                if state:
                    # Update the model state dict with merged weights
                    state.model_state_dict = merged_state_dict.copy()
                    logger.info(f"Updated {locale} with merged weights")
                else:
                    logger.warning(f"No state found for locale {locale} during weight redistribution")

        except Exception as e:
            logger.error(f"Failed to redistribute merged weights: {e}")
            # Continue without redistribution - training can proceed with original weights

    def _record_merge_success(
        self,
        active_locales: List[str],
        target_locales: List[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Record a successful merge operation."""
        self.successful_merges += 1

        merge_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "active_locales": active_locales,
            "target_locales": target_locales,
            "merge_algorithm": self.merge_config.merge_algorithm,
            "weight_calculation": self.merge_config.weight_calculation,
            "merge_duration": time.time() - self.merge_start_time if self.merge_start_time else 0,
            "metadata": metadata or {}
        }

        self.merge_history.append(merge_record)
        logger.info(f"Recorded successful merge #{self.successful_merges}")

    def _record_merge_failure(
        self,
        active_locales: List[str],
        target_locales: List[str],
        error_message: str
    ):
        """Record a failed merge operation."""
        self.failed_merges += 1

        merge_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed",
            "active_locales": active_locales,
            "target_locales": target_locales,
            "merge_algorithm": self.merge_config.merge_algorithm,
            "weight_calculation": self.merge_config.weight_calculation,
            "error_message": error_message,
            "merge_duration": time.time() - self.merge_start_time if self.merge_start_time else 0
        }

        self.merge_history.append(merge_record)
        logger.error(f"Recorded failed merge #{self.failed_merges}: {error_message}")

    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get statistics about merge operations."""
        total_merges = self.successful_merges + self.failed_merges
        success_rate = (self.successful_merges / total_merges * 100) if total_merges > 0 else 0

        return {
            "total_merges": total_merges,
            "successful_merges": self.successful_merges,
            "failed_merges": self.failed_merges,
            "success_rate": success_rate,
            "total_merge_time": self.total_merge_time,
            "average_merge_time": (self.total_merge_time / self.successful_merges) if self.successful_merges > 0 else 0,
            "merge_in_progress": self.merge_in_progress,
            "recent_merges": self.merge_history[-5:]  # Last 5 merges
        }

    def save_merge_history(self, output_path: str):
        """Save the complete merge history to a file."""
        import json
        history_data = {
            "merge_statistics": self.get_merge_statistics(),
            "merge_history": self.merge_history,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Merge history saved to: {output_path}")