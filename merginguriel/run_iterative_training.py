#!/usr/bin/env python
"""
Iterative Training and Merging CLI for MergingUriel.

This script provides a command-line interface for running iterative training
and merging experiments with the MergingUriel framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.iterative_config import (
    IterativeOrchestratorConfig,
    create_default_config,
    IterativeTrainingConfig,
    IterativeMergeConfig
)
from merginguriel.iterative_training_orchestrator import IterativeTrainingOrchestrator
from merginguriel.adaptive_merging import AdaptiveMergeScheduler, EnhancedMonitor
from merginguriel.similarity_utils import load_and_process_similarity


def setup_logging(log_level: str, log_file: Optional[str] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create log file only when explicitly requested
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def create_training_configs_from_args(args) -> List[IterativeTrainingConfig]:
    """Create training configurations from command line arguments."""
    locales = [locale.strip() for locale in args.locales.split(",")]

    training_configs = []
    for i, locale in enumerate(locales):
        config = IterativeTrainingConfig(
            locale=locale,
            dataset_config_name=locale,
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            model_name_or_path=args.model_name_or_path,
            output_dir=os.path.join(args.output_dir, locale),
            checkpoint_dir=os.path.join(args.output_dir, locale, "checkpoints"),
            save_strategy=args.save_strategy,
            eval_strategy=args.eval_strategy,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            early_stopping_patience=args.early_stopping_patience,
            bf16=args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay
        )
        training_configs.append(config)

    return training_configs


def create_merge_config_from_args(args) -> IterativeMergeConfig:
    """Create merge configuration from command line arguments."""
    # Map mode to merge_algorithm and weight_calculation for backward compatibility
    mode = args.mode

    # Determine merge algorithm based on mode
    if mode in ["linear", "fisher_simple", "fisher_dataset"]:
        merge_algorithm = mode
        weight_calculation = "similarity"  # Default for these modes
    elif mode == "similarity":
        merge_algorithm = "linear"
        weight_calculation = "similarity"
    elif mode == "average":
        merge_algorithm = "linear"
        weight_calculation = "average"
    elif mode == "uriel":
        merge_algorithm = "linear"
        weight_calculation = "similarity"
    elif mode == "manual":
        merge_algorithm = "linear"
        weight_calculation = "manual"
    else:  # iterative or other modes
        merge_algorithm = "linear"
        weight_calculation = "similarity"

    return IterativeMergeConfig(
        merge_frequency=args.merge_frequency,
        merge_algorithm=merge_algorithm,
        weight_calculation=weight_calculation,
        target_languages=[args.target_lang],  # Use target_lang for consistency with main pipeline
        top_k=args.top_k,
        sinkhorn_iters=args.sinkhorn_iters,
        num_languages=args.num_languages,
        fisher_data_mode=args.fisher_data_mode,
        preweight=args.preweight,
        num_fisher_examples=args.num_fisher_examples,
        fisher_batch_size=args.fisher_batch_size,
        fisher_max_seq_length=args.fisher_max_seq_length,
        checkpoint_before_merge=args.checkpoint_before_merge,
        retain_merge_checkpoints=args.retain_merge_checkpoints,
        merge_output_base_dir=os.path.join(args.output_dir, "merged_models")
    )


def create_orchestrator_config_from_args(args) -> IterativeOrchestratorConfig:
    """Create orchestrator configuration from command line arguments."""
    training_configs = create_training_configs_from_args(args)
    merge_config = create_merge_config_from_args(args)

    return IterativeOrchestratorConfig(
        training_configs=training_configs,
        merge_config=merge_config,
        orchestrator_name=args.experiment_name,
        base_output_dir=args.output_dir,
        log_level=args.log_level,
        sequential_training=args.sequential_training,
        max_sync_wait_time=args.max_sync_wait_time,
        merge_timeout=args.merge_timeout,
        enable_distributed=args.enable_distributed,
        max_gpu_memory=args.max_gpu_memory,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        log_frequency=args.log_frequency,
        enable_auto_recovery=args.enable_auto_recovery,
        max_merge_attempts=args.max_merge_attempts,
        validate_merge_integrity=args.validate_merge_integrity,
        adaptive_merge_frequency=args.adaptive_merge_frequency,
        performance_merge_trigger=args.performance_merge_trigger,
        convergence_threshold=args.convergence_threshold
    )


def auto_select_source_locales(target_languages: List[str], num_languages: int, top_k: int, sinkhorn_iters: int) -> List[str]:
    """Auto-select source locales based on similarity to target languages."""
    from collections import defaultdict

    # Get similarity matrix path
    similarity_matrix_path = os.path.join(project_root, "language_similarity_matrix_unified.csv")

    selected_locales = []

    for target_lang in target_languages:
        print(f"Finding source languages for target: {target_lang}")

        # Get similar languages using same logic as main pipeline
        similar_languages = load_and_process_similarity(
            similarity_matrix_path, target_lang, num_languages,
            top_k, sinkhorn_iters, verbose=False
        )

        # Extract just the locale codes
        source_locales = [locale for locale, weight in similar_languages]
        selected_locales.extend(source_locales)

        print(f"  Auto-selected source languages: {source_locales}")

    # Remove duplicates and limit to reasonable number
    unique_locales = list(dict.fromkeys(selected_locales))  # Preserve order, remove duplicates

    # Limit to 2x num_languages to avoid too many training jobs
    max_locales = max(num_languages * 2, len(target_languages))
    final_locales = unique_locales[:max_locales]

    print(f"Final source languages for training: {final_locales}")
    return final_locales


def validate_args(args):
    """Validate command line arguments."""
    # For iterative training, we need target_lang to find source languages
    if not args.target_lang:
        raise ValueError("Target language must be specified with --target-lang")

    # If locales is not specified, we'll auto-select source languages based on similarity
    if not args.locales:
        print("No source locales specified. Will auto-select based on similarity to target language.")
        # Auto-select based on the target language
        auto_selected = auto_select_source_locales(
            [args.target_lang], args.num_languages, args.top_k, args.sinkhorn_iters
        )
        args.locales = ",".join(auto_selected)
        print(f"Auto-selected source locales: {args.locales}")

    # Validate merge frequency
    if args.merge_frequency <= 0:
        raise ValueError("Merge frequency must be positive")

    # Validate batch size
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Validate learning rate
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    # Check if output directory exists or can be created
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory {args.output_dir}: {e}")

    # Validate GPU memory if specified
    if args.max-gpu-memory and args.max-gpu-memory <= 0:
        raise ValueError("Max GPU memory must be positive if specified")


def print_experiment_summary(config: IterativeOrchestratorConfig):
    """Print a summary of the experiment configuration."""
    print("="*80)
    print("ITERATIVE TRAINING & MERGING EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Experiment Name: {config.orchestrator_name}")
    print(f"Output Directory: {config.base_output_dir}")
    print(f"Number of Models: {len(config.training_configs)}")
    print()

    print("TRAINING CONFIGURATIONS:")
    print("-"*40)
    for i, training_config in enumerate(config.training_configs, 1):
        print(f"{i}. Locale: {training_config.locale}")
        print(f"   Dataset: {training_config.dataset_config_name}")
        print(f"   Max Epochs: {training_config.epochs}")
        print(f"   Batch Size: {training_config.batch_size}")
        print(f"   Learning Rate: {training_config.learning_rate}")
        print(f"   Output Dir: {training_config.output_dir}")
        print()

    print("MERGE CONFIGURATION:")
    print("-"*40)
    print(f"Algorithm: {config.merge_config.merge_algorithm}")
    print(f"Weight Calculation: {config.merge_config.weight_calculation}")
    print(f"Frequency: Every {config.merge_config.merge_frequency} epochs")
    print(f"Target Languages: {config.merge_config.target_languages}")
    print(f"Checkpoint Before Merge: {config.merge_config.checkpoint_before_merge}")
    print()

    print("ADVANCED FEATURES:")
    print("-"*40)
    print(f"Adaptive Merge Frequency: {config.adaptive_merge_frequency}")
    print(f"Performance Merge Trigger: {config.performance_merge_trigger}")
    print(f"Auto Recovery: {config.enable_auto_recovery}")
    print(f"Wandb Integration: {config.enable_wandb}")
    print(f"Distributed Training: {config.enable_distributed}")
    print()

    print("="*80)


def main():
    """Main function for the iterative training CLI."""
    parser = argparse.ArgumentParser(
        description="Run iterative training and merging experiments with MergingUriel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment configuration
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="iterative_training_experiment",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="iterative_training_results",
        help="Base output directory for all results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    # Model configuration (matching main pipeline)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["uriel", "manual", "similarity", "average", "fisher", "fisher_simple", "fisher_dataset", "iterative"],
        help="The merging mode to use."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="sq-AL",
        help="Target language/locale for similarity-based merging (e.g., sq-AL, th-TH, af-ZA)"
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
        help="Column name containing text data (MASSIVE uses \"utt\")"
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
        help="Sinkhorn normalization iterations for similarity computation"
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
        "--locales",
        type=str,
        help="Comma-separated list of source locale codes to train (e.g., en-US,fr-FR,de-DE). If not specified, will auto-select based on similarity to target-lang"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="xlm-roberta-base",
        help="Base model to fine-tune"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    # fp16 removed - bf16 is now the standard precision
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )

    # Checkpointing and evaluation
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="When to save checkpoints"
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="When to run evaluation"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (if save_strategy=steps)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run evaluation every N steps (if eval_strategy=steps)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Patience for early stopping"
    )

    # Merge configuration (matching main pipeline where possible)
    parser.add_argument(
        "--merge-frequency",
        type=int,
        default=1,
        help="Number of epochs between merges (always per epoch)"
    )

    # Fisher-based merging parameters
    parser.add_argument(
        "--fisher-batch-size",
        type=int,
        default=16,
        help="Batch size for Fisher computation"
    )
    parser.add_argument(
        "--fisher-max-seq-length",
        type=int,
        default=128,
        help="Max sequence length for Fisher computation"
    )

    # Merge execution parameters
    parser.add_argument(
        "--checkpoint-before-merge",
        action="store_true",
        help="Create checkpoints before each merge"
    )
    parser.add_argument(
        "--retain-merge-checkpoints",
        type=int,
        default=3,
        help="Number of merge checkpoints to retain"
    )

    # Orchestrator configuration
    parser.add_argument(
        "--max-sync-wait-time",
        type=int,
        default=300,
        help="Maximum time to wait for synchronization (seconds)"
    )
    parser.add_argument(
        "--merge-timeout",
        type=int,
        default=600,
        help="Timeout for merge operations (seconds)"
    )
    parser.add_argument(
        "--enable-distributed",
        action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        help="Maximum GPU memory per model (MB)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin memory for data loading"
    )

    # Monitoring and logging
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="MergingUriel-Iterative",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Wandb entity name"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Wandb mode: online (sync to cloud), offline (local only), disabled (no logging)"
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=100,
        help="Logging frequency (steps)"
    )

    # Training strategy
    parser.add_argument(
        "--sequential-training",
        action="store_true",
        default=True,
        help="Train models one by one to prevent OOM (default: enabled)"
    )

    # Recovery and robustness
    parser.add_argument(
        "--enable-auto-recovery",
        action="store_true",
        help="Enable automatic recovery from failures"
    )
    parser.add_argument(
        "--max-merge-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each merge"
    )
    parser.add_argument(
        "--validate-merge-integrity",
        action="store_true",
        help="Validate merge integrity"
    )

    # Advanced features
    parser.add_argument(
        "--adaptive-merge-frequency",
        action="store_true",
        help="Enable adaptive merge frequency adjustment"
    )
    parser.add_argument(
        "--performance-merge-trigger",
        action="store_true",
        help="Enable performance-based merge triggering"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Convergence threshold for performance-based merging"
    )

    # Configuration file options
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save configuration to file"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Load configuration from JSON file"
    )
    parser.add_argument(
        "--enable-log-file",
        action="store_true",
        help="Enable log file creation (default: disabled, only enabled for large-scale experiments)"
    )


    args = parser.parse_args()

    # Setup logging with conditional file creation
    # Enable log files for large-scale experiments or when explicitly requested
    enable_log_file = args.enable_log_file or os.environ.get("LARGE_SCALE_EXPERIMENT", "false").lower() == "true"

    if enable_log_file:
        log_file = os.path.join(args.output_dir, "iterative_training.log")
        setup_logging(args.log_level, log_file)
    else:
        setup_logging(args.log_level)  # Console-only logging

    logger = logging.getLogger(__name__)

    try:
        # Load configuration from file if specified
        if args.config_file:
            logger.info(f"Loading configuration from {args.config_file}")
            config = IterativeOrchestratorConfig.load_config(args.config_file)
        else:
            # Validate arguments
            validate_args(args)

            # Create configuration from arguments
            config = create_orchestrator_config_from_args(args)

        # Save configuration if requested
        if args.save_config:
            config_path = os.path.join(args.output_dir, "experiment_config.json")
            config.save_config(config_path)
            logger.info(f"Configuration saved to {config_path}")

        # Print experiment summary
        print_experiment_summary(config)

        # Create and run the orchestrator
        logger.info("Initializing Iterative Training Orchestrator")
        orchestrator = IterativeTrainingOrchestrator(config)

        # Setup trainers
        logger.info("Setting up trainers")
        orchestrator.setup_trainers()

        # Run the training loop
        logger.info("Starting iterative training loop")
        orchestrator.run_training_loop()

        logger.info("Iterative training completed successfully")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    finally:
        logger.info("Cleaning up resources")


if __name__ == "__main__":
    main()