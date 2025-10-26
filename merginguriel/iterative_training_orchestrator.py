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
        self.training_configs: Dict[str, IterativeTrainingConfig] = {}
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

        # Track pending merged checkpoints that should initialize next epoch
        self.pending_merged_checkpoints: Dict[str, str] = {}
        self.pending_resume_reset: Dict[str, bool] = {}

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Initialized IterativeTrainingOrchestrator with {len(config.training_configs)} models")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        # Removed log file creation - only console logging now
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
                # logging.FileHandler(
                #     os.path.join(self.config.base_output_dir, "iterative_training.log")
                # )
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()

    def setup_trainers(self):
        """Initialize trainer instances for the configured locales."""
        logger.info("Setting up trainers for all locales")

        for training_config in self.config.training_configs:
            locale = training_config.locale
            logger.info(f"Setting up trainer for {locale}")

            try:
                # Register model with state manager
                self.state_manager.register_model(locale, vars(training_config))

                # Create output directory
                os.makedirs(training_config.output_dir, exist_ok=True)

                # For sequential training, store configs and lazy-load trainers later
                if self.config.sequential_training:
                    # Store training config for lazy loading
                    self.training_configs[locale] = training_config
                    # Initialize progress tracking
                    self.epoch_progress[locale] = 0
                    self.step_progress[locale] = 0
                    logger.info(f"Prepared config for {locale} (lazy loading)")
                else:
                    # Load trainer immediately for parallel training
                    trainer = self._create_trainer(training_config)
                    self.trainers[locale] = trainer
                    # Initialize progress tracking
                    self.epoch_progress[locale] = 0
                    self.step_progress[locale] = 0
                    logger.info(f"Successfully set up trainer for {locale}")

            except Exception as e:
                logger.error(f"Failed to setup trainer for {locale}: {e}")
                raise

    def _lazy_load_trainer(self, locale: str) -> Trainer:
        """Lazy load a trainer only when needed."""
        if locale not in self.trainers:
            if locale not in self.training_configs:
                raise ValueError(f"No training config found for locale {locale}")

            logger.info(f"Lazy loading trainer for {locale}")
            training_config = self.training_configs[locale]
            trainer = self._create_trainer(training_config)

            # If a merged checkpoint is pending, initialize model weights from it
            pending_checkpoint = self.pending_merged_checkpoints.pop(locale, None)
            if pending_checkpoint and os.path.exists(pending_checkpoint):
                try:
                    logger.info(f"Initializing {locale} trainer with merged weights from {pending_checkpoint}")
                    merged_model = AutoModelForSequenceClassification.from_pretrained(
                        pending_checkpoint,
                        num_labels=trainer.model.config.num_labels
                    )
                    trainer.model.load_state_dict(merged_model.state_dict())
                    self.pending_resume_reset[locale] = True
                    del merged_model
                except Exception as load_err:
                    logger.warning(f"Failed to load merged weights for {locale}: {load_err}")
                    self.pending_resume_reset[locale] = False
            else:
                # Ensure flag is set even when no merged weights are pending
                self.pending_resume_reset.setdefault(locale, False)

            self.trainers[locale] = trainer
            logger.info(f"Successfully lazy loaded trainer for {locale}")

        return self.trainers[locale]

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

            # Initialize wandb only if enabled and not disabled
            if self.config.enable_wandb and self.config.wandb_mode != "disabled":
                wandb_mode = self.config.wandb_mode if self.config.wandb_mode in ["online", "offline"] else "offline"
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=f"{run_name}_{training_config.locale}",
                    mode=wandb_mode,
                    save_code=False,  # Don't save code files to wandb
                    dir=self.config.base_output_dir if wandb_mode == "offline" else None,  # Keep wandb files in output dir for offline mode
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

        # Use training_configs for sequential training (lazy loading)
        if self.config.sequential_training:
            locales = list(self.training_configs.keys())
        else:
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

            # For iterative training, we don't store massive state dicts
            # The merge coordinator reads model weights directly from checkpoints
            logger.info(f"Final training completed for {locale}: epoch={trainer.state.epoch}, step={trainer.state.global_step}")

            # Clear GPU memory after training
            self._clear_gpu_memory()

            logger.info(f"Training completed successfully for {locale}")

        except Exception as e:
            logger.error(f"Training failed for {locale}: {e}")
            # Clear GPU memory on error
            self._clear_gpu_memory()
            raise

    def _ensure_tokenizer_files(self, locale: str, model_dir: str):
        """Ensure tokenizer files exist by copying them from base model if needed."""
        try:
            # Find base model tokenizer files
            base_tokenizer_dirs = [
                "/home/coder/.cache/huggingface/hub/models--FacebookAI--xlm-roberta-base/snapshots/"
            ]

            base_tokenizer_path = None
            for base_dir in base_tokenizer_dirs:
                if os.path.exists(base_dir):
                    # Find the latest snapshot
                    snapshots = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                    if snapshots:
                        base_tokenizer_path = os.path.join(base_dir, sorted(snapshots)[-1])
                        break

            if not base_tokenizer_path:
                logger.error("Could not find base model tokenizer directory")
                return

            # Copy required tokenizer files
            required_files = [
                'tokenizer.json',
                'tokenizer_config.json',
                'sentencepiece.bpe.model',
                'special_tokens_map.json'
            ]

            for token_file in required_files:
                src_path = os.path.join(base_tokenizer_path, token_file)
                dst_path = os.path.join(model_dir, token_file)

                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {token_file} from base model to {locale}")
                elif os.path.exists(dst_path):
                    logger.info(f"Tokenizer file {token_file} already exists for {locale}")

        except Exception as e:
            logger.error(f"Failed to ensure tokenizer files for {locale}: {e}")

    def _find_latest_checkpoint(self, locale: str) -> Optional[str]:
        """Find the latest checkpoint directory for a given locale."""
        try:
            locale_output_dir = os.path.join(self.config.base_output_dir, locale)

            if not os.path.exists(locale_output_dir):
                return None

            # Look for checkpoint directories
            checkpoint_dirs = []
            for item in os.listdir(locale_output_dir):
                if item.startswith("checkpoint-") and os.path.isdir(os.path.join(locale_output_dir, item)):
                    try:
                        step_num = int(item.split("-")[1])
                        checkpoint_dirs.append((step_num, os.path.join(locale_output_dir, item)))
                    except (IndexError, ValueError):
                        continue

            if not checkpoint_dirs:
                return None

            # Return the checkpoint with the highest step number
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])
            return latest_checkpoint[1]

        except Exception as e:
            logger.warning(f"Error finding checkpoint for {locale}: {e}")
            return None

    def _run_single_epoch(self, locale: str, target_epoch: int):
        """Run training for a single model for one specific epoch."""
        logger.info(f"Starting epoch {target_epoch} for {locale}")
        try:
            # Clear GPU memory before training
            self._clear_gpu_memory()

            # Lazy load trainer only when needed
            trainer = self._lazy_load_trainer(locale)

            # Determine target directory for this locale's artifacts
            locale_output_path = getattr(trainer.args, "output_dir", None)
            if not locale_output_path:
                locale_output_path = os.path.join(self.config.base_output_dir, locale)
            os.makedirs(locale_output_path, exist_ok=True)

            # Check if we should reset resume checkpoint after merge
            pending_reset = self.pending_resume_reset.get(locale, False)

            # Set model to training device with memory optimization
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use gradient checkpointing to save memory
            if hasattr(trainer.model, 'gradient_checkpointing_enable'):
                trainer.model.gradient_checkpointing_enable()
                logger.info(f"Enabled gradient checkpointing for {locale}")

            # Use mixed precision if available to save memory
            if hasattr(trainer.args, 'fp16') and trainer.args.fp16:
                logger.info(f"Using FP16 mixed precision for {locale}")

            trainer.model.to(device)
            logger.info(f"Training {locale} on device: {device}")

            # Find the latest checkpoint to resume from
            latest_checkpoint = None if pending_reset else self._find_latest_checkpoint(locale)

            # Set training parameters for this epoch
            original_max_epochs = trainer.args.num_train_epochs
            original_resume_from_checkpoint = getattr(trainer.args, 'resume_from_checkpoint', None)

            # Configure trainer to train for exactly one more epoch
            # We use the current epoch + 1 to train exactly one epoch
            current_epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
            trainer.args.num_train_epochs = current_epoch + 1

            logger.info(f"Debug: {locale} current_epoch={current_epoch}, target_epoch={target_epoch}, will train to epoch {current_epoch + 1}")

            # Set resume checkpoint if available
            if latest_checkpoint:
                trainer.args.resume_from_checkpoint = latest_checkpoint
                logger.info(f"Resuming {locale} from checkpoint: {latest_checkpoint}")
            else:
                # Remove any resume checkpoint to start fresh
                trainer.args.resume_from_checkpoint = None
                if pending_reset:
                    logger.info(f"Starting {locale} training from merged weights (no checkpoint resume)")
                else:
                    logger.info(f"Starting {locale} training from scratch")

            # Train for exactly one epoch
            logger.info(f"Training {locale} for one epoch (current: {current_epoch} → target: {current_epoch + 1})")
            trainer.train()

            # Restore original training arguments
            trainer.args.num_train_epochs = original_max_epochs
            trainer.args.resume_from_checkpoint = original_resume_from_checkpoint

            # Save model state after this epoch
            trainer.save_model()
            trainer.save_state()

            # Verify training completed to expected epoch
            final_epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
            logger.info(f"Debug: {locale} training completed. Expected epoch: {current_epoch + 1}, Actual epoch: {final_epoch}")

            if final_epoch != current_epoch + 1:
                logger.warning(f"Warning: {locale} training ended at epoch {final_epoch}, expected {current_epoch + 1}")
            else:
                logger.info(f"✅ {locale} training completed successfully to epoch {final_epoch}")

            # CRITICAL: Always save the tokenizer explicitly for merging
            # trainer.save_model() doesn't save the tokenizer by default
            try:
                # Use the same path where the model was saved
                tokenizer_save_path = locale_output_path

                # Get tokenizer from trainer
                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    tokenizer = trainer.tokenizer
                else:
                    # Load tokenizer from the model config
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(trainer.args.model_name_or_path)

                tokenizer.save_pretrained(tokenizer_save_path)
                logger.info(f"✅ SAVED TOKENIZER to: {tokenizer_save_path}")

            except Exception as e:
                logger.error(f"Failed to save tokenizer: {e}")
                # Copy from base model as fallback
                self._ensure_tokenizer_files(locale, locale_output_path)

                # Verify tokenizer files were saved
                tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'sentencepiece.bpe.model']
                missing_files = []
                for token_file in tokenizer_files:
                    token_path = os.path.join(tokenizer_save_path, token_file)
                    if not os.path.exists(token_path):
                        missing_files.append(token_file)

                if missing_files:
                    logger.warning(f"Missing tokenizer files: {missing_files}")
                    # Fallback: copy from base model if missing
                    self._ensure_tokenizer_files(locale, tokenizer_save_path)
                else:
                    logger.info("✅ All tokenizer files verified successfully")

            # Update state manager with lightweight metadata (no GPU tensors)
            # Create basic metrics object
            current_metrics = TrainingMetrics(
                epoch=int(trainer.state.epoch) if trainer.state.epoch is not None else 0,
                step=trainer.state.global_step,
                train_loss=trainer.state.log_history[-1].get('train_loss', 0.0) if trainer.state.log_history else 0.0,
                eval_loss=trainer.state.log_history[-1].get('eval_loss', None) if trainer.state.log_history else None,
                eval_accuracy=trainer.state.log_history[-1].get('eval_accuracy', None) if trainer.state.log_history else None,
                eval_f1=trainer.state.log_history[-1].get('eval_f1', None) if trainer.state.log_history else None
            )

            # Update state manager with metadata only (no GPU tensors)
            self.state_manager.update_state(
                locale=locale,
                epoch=int(trainer.state.epoch) if trainer.state.epoch is not None else 0,
                step=trainer.state.global_step,
                model_state_dict=None,  # Don't store GPU tensors
                optimizer_state_dict=None,  # Don't store optimizer state
                scheduler_state_dict=None,  # Don't store scheduler state
                metrics=current_metrics,
                checkpoint_path=locale_output_path  # Store path to saved model
            )

            logger.info(f"Completed training for {locale}: epoch={trainer.state.epoch}, step={trainer.state.global_step}")

            # Update progress tracking
            self.epoch_progress[locale] = trainer.state.epoch
            self.step_progress[locale] = trainer.state.global_step

            # Clear GPU memory after training
            self._clear_gpu_memory()

            # For sequential training, unload the model to free memory
            if self.config.sequential_training and locale in self.trainers:
                logger.info(f"Unloading model for {locale} to free GPU memory")

                # Move model to CPU first to ensure GPU memory is freed
                if hasattr(trainer, 'model') and trainer.model is not None:
                    trainer.model.cpu()

                # Delete trainer and force garbage collection
                del self.trainers[locale]
                del trainer

                # Force garbage collection and clear GPU cache multiple times
                import gc
                gc.collect()
                self._clear_gpu_memory()

                # Clear any cached states in state manager
                if hasattr(self.state_manager, 'clear_model_cache'):
                    self.state_manager.clear_model_cache(locale)
                    logger.info(f"Cleared state manager cache for {locale}")

                gc.collect()
                self._clear_gpu_memory()
                gc.collect()
                self._clear_gpu_memory()

                logger.info(f"Aggressively cleared GPU memory for {locale}")

            logger.info(f"Completed epoch {target_epoch} for {locale}")

            # Clear pending reset flag after successful training
            self.pending_resume_reset[locale] = False

        except Exception as e:
            logger.error(f"Training failed for {locale} epoch {target_epoch}: {e}")
            # Clear GPU memory on error
            if self.config.sequential_training and locale in self.trainers:
                trainer = self.trainers[locale]
                if hasattr(trainer, 'model') and trainer.model is not None:
                    trainer.model.cpu()
                del self.trainers[locale]
                del trainer

                import gc
                gc.collect()
                self._clear_gpu_memory()
            self._clear_gpu_memory()
            raise

    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        if torch.cuda.is_available():
            # Get current memory usage before clearing
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated_before = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved_before = torch.cuda.memory_reserved() / 1024**3   # GB
                logger.info(f"GPU memory before clearing: {allocated_before:.2f}GB allocated, {reserved_before:.2f}GB reserved")

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Get memory usage after clearing
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated_after = torch.cuda.memory_allocated() / 1024**3   # GB
                reserved_after = torch.cuda.memory_reserved() / 1024**3    # GB
                logger.info(f"GPU memory after clearing: {allocated_after:.2f}GB allocated, {reserved_after:.2f}GB reserved")

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
                merged_state_dict = merged_model.state_dict()
                tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

                # Update persistent checkpoints for every locale
                for locale, training_config in self.training_configs.items():
                    locale_output_dir = training_config.output_dir or os.path.join(self.config.base_output_dir, locale)
                    os.makedirs(locale_output_dir, exist_ok=True)

                    # Save merged model and tokenizer so disk checkpoints reflect latest weights
                    merged_model.save_pretrained(locale_output_dir)
                    tokenizer.save_pretrained(locale_output_dir)

                    # Update state manager tracking
                    state = self.state_manager.get_state(locale)
                    if state:
                        state.checkpoint_path = locale_output_dir
                        state.checkpoint_timestamp = datetime.utcnow().isoformat()
                        state.model_state_dict = None  # Rely on disk-based checkpoint

                    # Ensure next epoch loads from merged checkpoint
                    self.pending_merged_checkpoints[locale] = locale_output_dir
                    self.pending_resume_reset[locale] = True

                    # Update trainer currently in memory, if any
                    if locale in self.trainers:
                        self.trainers[locale].model.load_state_dict(merged_state_dict)
                        logger.info(f"Updated {locale} trainer with merged weights for epoch {epoch+1}")

                # Clean up merged artifacts from memory
                del merged_model
                del tokenizer
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
                merged_state_dict = merged_model.state_dict()
                tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

                # Update trainers for remaining untrained locales
                all_locales = list(self.trainers.keys())
                remaining_locales = [loc for loc in all_locales if loc not in trained_locales]

                for locale in remaining_locales:
                    if locale in self.trainers:
                        # Update the trainer's model with merged weights
                        self.trainers[locale].model.load_state_dict(merged_state_dict)
                        logger.info(f"Updated {locale} trainer with merged weights")

                    # Persist merged weights for locales that will train later
                    training_config = self.training_configs.get(locale)
                    if training_config:
                        locale_output_dir = training_config.output_dir or os.path.join(self.config.base_output_dir, locale)
                        os.makedirs(locale_output_dir, exist_ok=True)
                        merged_model.save_pretrained(locale_output_dir)
                        tokenizer.save_pretrained(locale_output_dir)

                        state = self.state_manager.get_state(locale)
                        if state:
                            state.checkpoint_path = locale_output_dir
                            state.checkpoint_timestamp = datetime.utcnow().isoformat()
                            state.model_state_dict = None

                        self.pending_merged_checkpoints[locale] = locale_output_dir
                        self.pending_resume_reset[locale] = True

                # Clean up
                del merged_model
                del tokenizer
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

        # For iterative training, we don't need state manager updates
        # Progress is tracked locally in epoch_progress/step_progress

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
