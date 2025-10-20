"""
Iterative Training Orchestrator for MergingUriel.

This module provides the IterativeTrainingOrchestrator class that manages
multiple simultaneous language model training processes with periodic
merge operations during training.
"""

import os
import sys
import time
import logging
import threading
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path

import torch
import transformers
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.iterative_config import (
    IterativeOrchestratorConfig, IterativeTrainingConfig, IterativeMergeConfig
)
from merginguriel.training_state import TrainingStateManager, TrainingMetrics
from merginguriel.merge_coordinator import MergeCoordinator
from merginguriel.training_bert import generate_wandb_run_name

logger = logging.getLogger(__name__)


class IterativeTrainingOrchestrator:
    """
    Orchestrates iterative training and merging of multiple language models.

    This class manages:
    - Multiple simultaneous training processes
    - Periodic merge operations during training
    - State synchronization and checkpointing
    - Resource management and monitoring
    - Error handling and recovery
    """

    def __init__(self, config: IterativeOrchestratorConfig):
        self.config = config
        self.project_root = project_root

        # Initialize logging
        self._setup_logging()

        # Core components
        self.state_manager = TrainingStateManager(config.base_output_dir)
        self.merge_coordinator = MergeCoordinator(
            config.merge_config,
            self.state_manager,
            config.base_output_dir
        )

        # Training state
        self.trainers: Dict[str, Trainer] = {}
        self.training_threads: Dict[str, threading.Thread] = {}
        self.training_futures: Dict[str, Future] = {}
        self.stop_event = threading.Event()

        # Synchronization
        self.merge_barrier = threading.Barrier(len(config.training_configs) + 1)  # +1 for orchestrator
        self.training_complete = threading.Event()

        # Performance tracking
        self.start_time: Optional[float] = None
        self.epoch_progress: Dict[str, int] = {}
        self.step_progress: Dict[str, int] = {}

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Initialized IterativeTrainingOrchestrator with {len(config.training_configs)} models")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(self.config.base_output_dir, "iterative_training.log")
                )
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()

    def setup_trainers(self):
        """Initialize all trainer instances for the configured locales."""
        logger.info("Setting up trainers for all locales")

        for training_config in self.config.training_configs:
            locale = training_config.locale
            logger.info(f"Setting up trainer for {locale}")

            try:
                # Register model with state manager
                self.state_manager.register_model(locale, vars(training_config))

                # Create output directory
                os.makedirs(training_config.output_dir, exist_ok=True)

                # Initialize trainer
                trainer = self._create_trainer(training_config)
                self.trainers[locale] = trainer

                # Initialize progress tracking
                self.epoch_progress[locale] = 0
                self.step_progress[locale] = 0

                logger.info(f"Successfully set up trainer for {locale}")

            except Exception as e:
                logger.error(f"Failed to setup trainer for {locale}: {e}")
                raise

    def _create_trainer(self, training_config: IterativeTrainingConfig) -> Trainer:
        """Create a trainer instance for the given configuration."""
        # Load dataset
        logger.info(f"Loading dataset for {training_config.locale}")
        raw_datasets = load_dataset(
            "AmazonScience/massive",
            training_config.dataset_config_name,
            cache_dir=None,
            trust_remote_code=True
        )

        # Rename 'intent' column to 'labels'
        raw_datasets = raw_datasets.rename_column("intent", "labels")

        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer for {training_config.locale}")
        model = AutoModelForSequenceClassification.from_pretrained(
            training_config.model_name_or_path,
            num_labels=len(raw_datasets["train"].features["labels"].names)
        )
        tokenizer = AutoTokenizer.from_pretrained(training_config.model_name_or_path)

        # Preprocessing function
        def preprocess_function(examples):
            # Tokenize the text
            tokenized_inputs = tokenizer(
                examples["utt"],
                padding="max_length",
                max_length=training_config.max_seq_length,
                truncation=True,
            )
            # Add labels
            tokenized_inputs["labels"] = examples["labels"]
            return tokenized_inputs

        # Preprocess datasets
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
        )

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.max_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            logging_dir=f"{training_config.output_dir}/logs",
            logging_steps=self.config.log_frequency,
            eval_strategy=training_config.eval_strategy,
            eval_steps=training_config.eval_steps,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            fp16=training_config.fp16,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            report_to="wandb" if self.config.enable_wandb else "none",
            disable_tqdm=False,
        )

        # Setup wandb if enabled
        if self.config.enable_wandb:
            self._setup_wandb(training_config, training_args)

        # Metrics function
        def compute_metrics(eval_pred):
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score

            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted")

            return {
                "accuracy": accuracy,
                "f1": f1
            }

        # Create custom trainer with callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                IterativeTrainingCallback(
                    locale=training_config.locale,
                    orchestrator=self,
                    state_manager=self.state_manager
                )
            ]
        )

        return trainer

    def _setup_wandb(self, training_config: IterativeTrainingConfig, training_args: TrainingArguments):
        """Setup wandb logging for the trainer."""
        try:
            import wandb

            # Generate run name
            run_name = generate_wandb_run_name(
                type('ModelArgs', (), {
                    'model_name_or_path': training_config.model_name_or_path
                })(),
                type('DataArgs', (), {
                    'dataset_name': 'AmazonScience/massive',
                    'dataset_config_name': training_config.dataset_config_name
                })(),
                training_args
            )

            # Initialize wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{run_name}_{training_config.locale}",
                config={
                    "locale": training_config.locale,
                    "iterative_training": True,
                    "merge_frequency": self.config.merge_config.merge_frequency,
                    "merge_algorithm": self.config.merge_config.merge_algorithm
                }
            )

        except Exception as e:
            logger.warning(f"Failed to setup wandb for {training_config.locale}: {e}")

    def run_training_loop(self):
        """Execute the main training loop with periodic merges."""
        logger.info("Starting iterative training loop")
        self.start_time = time.time()

        try:
            if self.config.sequential_training:
                self._run_sequential_training()
            else:
                self._run_parallel_training()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            raise
        finally:
            self.training_complete.set()
            self._cleanup()

        logger.info("Iterative training loop completed")

    def _run_sequential_training(self):
        """Run training models with proper epoch-by-epoch iterative merging."""
        logger.info("Running iterative training with epoch-level merging")

        locales = list(self.trainers.keys())
        max_epochs = self.config.training_configs[0].max_epochs
        merge_frequency = self.config.merge_config.merge_frequency

        logger.info(f"Training {len(locales)} models for {max_epochs} epochs with merge every {merge_frequency} epochs")

        # Train epoch by epoch with merging
        for epoch in range(max_epochs):
            logger.info(f"=== EPOCH {epoch + 1}/{max_epochs} ===")

            # Train each model for one epoch
            for locale in locales:
                logger.info(f"Training {locale} for epoch {epoch}")
                self._run_single_epoch(locale, epoch)

            # Check if we should merge after this epoch
            if (epoch + 1) % merge_frequency == 0:
                logger.info(f"Triggering merge after epoch {epoch + 1}")
                success = self._execute_merge_cycle_after_epoch(locales, epoch + 1)
                if success:
                    logger.info(f"✅ Merge #{(epoch + 1) // merge_frequency} completed successfully")
                else:
                    logger.error(f"❌ Merge #{(epoch + 1) // merge_frequency} failed")

        logger.info("All iterative training completed")

    def _run_parallel_training(self):
        """Run training models in parallel (legacy method - may cause OOM)."""
        logger.warning("Running parallel training - this may cause GPU memory issues!")

        # Start training for all models
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit training tasks
            for locale, trainer in self.trainers.items():
                future = executor.submit(self._run_single_trainer, locale, trainer)
                self.training_futures[locale] = future
                logger.info(f"Submitted training task for {locale}")

            # Monitor progress and handle merges
            self._monitor_training_and_merges()

            # Wait for all training to complete
            for locale, future in self.training_futures.items():
                try:
                    future.result()
                    logger.info(f"Training completed for {locale}")
                except Exception as e:
                    logger.error(f"Training failed for {locale}: {e}")

    def _run_single_trainer(self, locale: str, trainer: Trainer):
        """Run training for a single model (legacy method for compatibility)."""
        return self._run_single_trainer_sequential(locale, trainer)

    def _run_single_trainer_sequential(self, locale: str, trainer: Trainer):
        """Run training for a single model sequentially with memory management."""
        logger.info(f"Starting sequential training for {locale}")

        try:
            # Clear GPU memory before training
            self._clear_gpu_memory()

            # Set model to training device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer.model.to(device)

            logger.info(f"Training {locale} on device: {device}")

            # Train the model
            trainer.train()

            # Save final model
            trainer.save_model()
            trainer.save_state()

            # Update state manager with final training state for merging
            try:
                # Get final training metrics
                log_history = trainer.state.log_history
                final_metrics = None
                if log_history:
                    final_log = log_history[-1]  # Get the last log entry
                    final_metrics = TrainingMetrics(
                        epoch=final_log.get('epoch', 0),
                        step=trainer.state.global_step,
                        train_loss=final_log.get('train_loss', 0.0),
                        eval_loss=final_log.get('eval_loss'),
                        eval_accuracy=final_log.get('eval_accuracy'),
                        eval_f1=final_log.get('eval_f1'),
                        learning_rate=final_log.get('learning_rate', 0.0)
                    )

                # Update the state manager with model weights and metrics
                self.state_manager.update_state(
                    locale=locale,
                    epoch=trainer.state.epoch,
                    step=trainer.state.global_step,
                    model_state_dict=trainer.model.state_dict(),
                    optimizer_state_dict=trainer.optimizer.state_dict(),
                    metrics=final_metrics
                )
                logger.info(f"Updated training state for {locale}: epoch={trainer.state.epoch}, step={trainer.state.global_step}")

            except Exception as state_error:
                logger.warning(f"Failed to update training state for {locale}: {state_error}")
                # Continue anyway - merge coordinator can use checkpoint fallback

            # Clear GPU memory after training
            self._clear_gpu_memory()

            logger.info(f"Training completed successfully for {locale}")

        except Exception as e:
            logger.error(f"Training failed for {locale}: {e}")
            # Clear GPU memory on error
            self._clear_gpu_memory()
            raise

    def _run_single_epoch(self, locale: str, target_epoch: int):
        """Run training for a single model for one specific epoch."""
        logger.info(f"Starting epoch {target_epoch} for {locale}")
        try:
            # Clear GPU memory before training
            self._clear_gpu_memory()

            trainer = self.trainers[locale]

            # Set model to training device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer.model.to(device)
            logger.info(f"Training {locale} on device: {device}")

            # Calculate how many epochs to train (from current to target)
            current_epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
            epochs_to_train = target_epoch - current_epoch + 1  # +1 because epochs are 0-indexed

            if epochs_to_train <= 0:
                logger.info(f"Skipping {locale} - already at epoch {current_epoch}")
                return

            # Modify training arguments for single epoch
            original_max_epochs = trainer.args.num_train_epochs
            trainer.args.num_train_epochs = target_epoch + 1  # Train up to target epoch

            # Train for the required number of epochs
            logger.info(f"Training {locale} from epoch {current_epoch} to {target_epoch} ({epochs_to_train} epochs)")
            trainer.train()

            # Restore original max epochs
            trainer.args.num_train_epochs = original_max_epochs

            # Save model state after this epoch
            trainer.save_model()
            trainer.save_state()

            # Update state manager with training state for merging
            try:
                # Get latest training metrics
                log_history = trainer.state.log_history
                current_metrics = None
                if log_history:
                    # Find the log entry for the completed epoch (use the most recent)
                    latest_log = log_history[-1]
                    current_metrics = TrainingMetrics(
                        epoch=latest_log.get('epoch', target_epoch),
                        step=trainer.state.global_step,
                        train_loss=latest_log.get('train_loss', 0.0),
                        eval_loss=latest_log.get('eval_loss'),
                        eval_accuracy=latest_log.get('eval_accuracy'),
                        eval_f1=latest_log.get('eval_f1'),
                        learning_rate=latest_log.get('learning_rate', 0.0)
                    )

                # Update the state manager with model weights and metrics
                self.state_manager.update_state(
                    locale=locale,
                    epoch=trainer.state.epoch,
                    step=trainer.state.global_step,
                    model_state_dict=trainer.model.state_dict(),
                    optimizer_state_dict=trainer.optimizer.state_dict(),
                    metrics=current_metrics
                )
                logger.info(f"Updated training state for {locale}: epoch={trainer.state.epoch}, step={trainer.state.global_step}")

            except Exception as state_error:
                logger.warning(f"Failed to update training state for {locale}: {state_error}")

            # Update progress tracking
            self.epoch_progress[locale] = trainer.state.epoch
            self.step_progress[locale] = trainer.state.global_step

            # Clear GPU memory after training
            self._clear_gpu_memory()

            logger.info(f"Completed epoch {target_epoch} for {locale}")

        except Exception as e:
            logger.error(f"Training failed for {locale} epoch {target_epoch}: {e}")
            # Clear GPU memory on error
            self._clear_gpu_memory()
            raise

    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared GPU cache")

    def _should_merge_after_model(self, locale: str, model_index: int, total_models: int) -> bool:
        """Determine if we should merge after training a specific model."""
        # Merge after every N models (based on merge frequency)
        models_per_merge = max(1, total_models // max(1, self.config.merge_config.merge_frequency))

        # Always merge after the last model
        if model_index == total_models - 1:
            return True

        # Merge after every models_per_merge models
        return (model_index + 1) % models_per_merge == 0

    def _execute_merge_cycle_after_training(self, trained_locales: List[str]):
        """Execute a merge cycle after training specific models."""
        logger.info(f"Starting merge cycle for trained models: {trained_locales}")

        try:
            # Get active locales (those that have been trained)
            active_locales = trained_locales.copy()
            target_languages = self.config.merge_config.target_languages

            if not target_languages:
                target_languages = active_locales

            # Execute merge
            success, merged_model_path = self.merge_coordinator.execute_merge(
                active_locales=active_locales,
                target_locales=target_languages,
                merge_metadata={
                    "trained_models": len(active_locales),
                    "merge_type": "sequential_training",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            if success:
                logger.info(f"Merge cycle completed successfully: {merged_model_path}")

                # Optional: Update remaining trainers with merged weights
                self._update_remaining_trainers_with_merge(merged_model_path, active_locales, target_languages)
            else:
                logger.error("Merge cycle failed")

        except Exception as e:
            logger.error(f"Merge cycle failed: {e}")

    def _execute_merge_cycle_after_epoch(self, trained_locales: List[str], epoch: int) -> bool:
        """Execute a merge cycle after completing a specific epoch."""
        logger.info(f"Starting merge cycle after epoch {epoch} for models: {trained_locales}")

        try:
            # Clear GPU memory before merging
            self._clear_gpu_memory()

            # Get active locales
            active_locales = trained_locales.copy()
            target_languages = self.config.merge_config.target_languages

            if not target_languages:
                target_languages = active_locales

            # Execute merge with current epoch information
            success, merged_model_path = self.merge_coordinator.execute_merge(
                active_locales=active_locales,
                target_locales=target_languages,
                merge_metadata={
                    "epoch": epoch,
                    "trained_models": len(active_locales),
                    "merge_type": "iterative_epoch_merge",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            if success:
                logger.info(f"✅ Epoch {epoch} merge completed successfully: {merged_model_path}")

                # Update all trainers with merged weights for next epoch
                self._update_all_trainers_with_merge(merged_model_path, active_locales, epoch)

                # Clear GPU memory after merge
                self._clear_gpu_memory()

                return True
            else:
                logger.error(f"❌ Epoch {epoch} merge failed")
                return False

        except Exception as e:
            logger.error(f"Epoch {epoch} merge cycle failed: {e}")
            return False

    def _update_all_trainers_with_merge(self, merged_model_path: str, active_locales: List[str], epoch: int):
        """Update all trainers with weights from the merged model for continued training."""
        try:
            if merged_model_path and os.path.exists(merged_model_path):
                # Load merged model
                logger.info("Loading merged model to update trainers")
                merged_model = AutoModelForSequenceClassification.from_pretrained(merged_model_path)

                # Update all trainers (not just the active ones)
                for locale in self.trainers.keys():
                    if locale in self.trainers:
                        # Update the trainer's model with merged weights
                        self.trainers[locale].model.load_state_dict(merged_model.state_dict())
                        logger.info(f"Updated {locale} trainer with merged weights for epoch {epoch+1}")

                # Clean up merged model from memory
                del merged_model
                self._clear_gpu_memory()

                logger.info("Successfully updated all trainers with merged weights")
            else:
                logger.warning(f"Could not update trainers - merged model not found: {merged_model_path}")

        except Exception as e:
            logger.error(f"Failed to update trainers with merge weights: {e}")

    def _update_remaining_trainers_with_merge(self, merged_model_path: str, trained_locales: List[str], target_locales: List[str]):
        """Update remaining untrained models with weights from the merged model."""
        try:
            if merged_model_path and os.path.exists(merged_model_path):
                # Load merged model
                merged_model = AutoModelForSequenceClassification.from_pretrained(merged_model_path)

                # Update trainers for remaining untrained locales
                all_locales = list(self.trainers.keys())
                remaining_locales = [loc for loc in all_locales if loc not in trained_locales]

                for locale in remaining_locales:
                    if locale in self.trainers:
                        # Update the trainer's model with merged weights
                        self.trainers[locale].model.load_state_dict(merged_model.state_dict())
                        logger.info(f"Updated {locale} trainer with merged weights")

                # Clean up
                del merged_model
                self._clear_gpu_memory()

        except Exception as e:
            logger.warning(f"Failed to update remaining trainers with merge weights: {e}")

    def _monitor_training_and_merges(self):
        """Monitor training progress and trigger merges when appropriate."""
        last_merge_epoch = {locale: -1 for locale in self.trainers.keys()}

        while not self.stop_event.is_set() and not self.training_complete.is_set():
            try:
                # Check if any merge should be triggered
                for locale in self.trainers.keys():
                    current_epoch = self.epoch_progress.get(locale, 0)
                    current_step = self.step_progress.get(locale, 0)

                    # Check if we should merge (avoid multiple merges for same epoch)
                    if (current_epoch > last_merge_epoch[locale] and
                        self.merge_coordinator.should_merge(
                            current_epoch=current_epoch,
                            current_step=current_step,
                            metrics_history=[]  # TODO: Pass actual metrics history
                        )):

                        logger.info(f"Triggering merge at epoch {current_epoch}")
                        self._execute_merge_cycle()

                        # Update last merge epoch for all models
                        for l in last_merge_epoch:
                            last_merge_epoch[l] = current_epoch

                        break  # Only one merge per check cycle

                # Wait before next check
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in training monitoring: {e}")
                time.sleep(30)  # Wait longer on error

    def _execute_merge_cycle(self):
        """Execute a complete merge cycle."""
        logger.info("Starting merge cycle")

        try:
            # Pause training for all models
            logger.info("Pausing training for merge cycle")
            self._pause_all_training()

            # Get active locales (those currently training)
            active_locales = list(self.trainers.keys())
            target_languages = self.config.merge_config.target_languages

            if not target_languages:
                target_languages = active_locales

            # Execute merge
            success, merged_model_path = self.merge_coordinator.execute_merge(
                active_locales=active_locales,
                target_locales=target_languages,
                merge_metadata={
                    "epoch": max(self.epoch_progress.values()),
                    "step": max(self.step_progress.values()),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            if success:
                logger.info(f"Merge cycle completed successfully: {merged_model_path}")
            else:
                logger.error("Merge cycle failed")

            # Resume training
            logger.info("Resuming training after merge cycle")
            self._resume_all_training()

        except Exception as e:
            logger.error(f"Merge cycle failed: {e}")
            # Ensure training resumes even if merge fails
            self._resume_all_training()

    def _pause_all_training(self):
        """Pause training for all models (implementation depends on trainer)."""
        # Note: This is a simplified implementation
        # In practice, pausing might require more sophisticated synchronization
        logger.info("Training pause requested")

    def _resume_all_training(self):
        """Resume training for all models."""
        logger.info("Training resume requested")

    def update_progress(self, locale: str, epoch: int, step: int, metrics: TrainingMetrics):
        """Update training progress for a model."""
        self.epoch_progress[locale] = epoch
        self.step_progress[locale] = step

        # Update state manager
        self.state_manager.update_state(
            locale=locale,
            epoch=epoch,
            step=step,
            metrics=metrics
        )

        # Log progress
        if step % self.config.log_frequency == 0:
            logger.info(f"{locale} - Epoch {epoch}, Step {step}, Metrics: {metrics}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status for all models."""
        return {
            "start_time": self.start_time,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "epoch_progress": self.epoch_progress.copy(),
            "step_progress": self.step_progress.copy(),
            "merge_statistics": self.merge_coordinator.get_merge_statistics(),
            "training_complete": self.training_complete.is_set(),
            "stop_requested": self.stop_event.is_set()
        }

    def _cleanup(self):
        """Cleanup resources and save final state."""
        logger.info("Cleaning up resources")

        # Save final state summary
        self.state_manager.save_state_summary()

        # Save merge history
        merge_history_path = os.path.join(
            self.config.base_output_dir,
            "merge_history.json"
        )
        self.merge_coordinator.save_merge_history(merge_history_path)

        # Save final statistics
        final_stats = {
            "training_status": self.get_training_status(),
            "config": self.config.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }

        stats_path = os.path.join(
            self.config.base_output_dir,
            "final_statistics.json"
        )
        import json
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)

        logger.info("Cleanup completed")


class IterativeTrainingCallback(transformers.TrainerCallback):
    """Custom callback for iterative training to track progress and coordinate merges."""

    def __init__(self, locale: str, orchestrator: IterativeTrainingOrchestrator, state_manager: TrainingStateManager):
        self.locale = locale
        self.orchestrator = orchestrator
        self.state_manager = state_manager

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        logger.info(f"{self.locale} - Starting epoch {state.epoch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        logger.info(f"{self.locale} - Completed epoch {state.epoch}")

        # Update orchestrator progress
        metrics = TrainingMetrics(
            epoch=state.epoch,
            step=state.global_step,
            train_loss=state.log_history[-1].get('train_loss', 0) if state.log_history else 0,
            eval_loss=state.log_history[-1].get('eval_loss', 0) if state.log_history else 0,
            eval_accuracy=state.log_history[-1].get('eval_accuracy', 0) if state.log_history else 0,
            eval_f1=state.log_history[-1].get('eval_f1', 0) if state.log_history else 0,
            learning_rate=state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
        )

        self.orchestrator.update_progress(
            locale=self.locale,
            epoch=state.epoch,
            step=state.global_step,
            metrics=metrics
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called on each log."""
        if logs and state.global_step % 100 == 0:  # Log every 100 steps
            logger.debug(f"{self.locale} - Step {state.global_step}: {logs}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info(f"{self.locale} - Training completed")
        logger.info(f"{self.locale} - Final epoch: {state.epoch}, Final step: {state.global_step}")

        # Create final checkpoint - but let the main training loop handle state updates
        try:
            # Just save basic metadata, don't try to save model state here
            # The model state is already handled by the main training loop
            logger.info(f"{self.locale} - Training completed successfully")
        except Exception as e:
            logger.error(f"{self.locale} - Error in training completion: {e}")