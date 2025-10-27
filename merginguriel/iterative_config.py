"""
Configuration system for iterative training and merging.

This module provides the configuration classes needed for the iterative
training & merging pipeline, extending the existing MergingUriel configuration
system with new parameters specific to training-time merging.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path


@dataclass
class IterativeTrainingConfig:
    """Configuration for individual language model training."""

    # Basic training parameters
    locale: str
    dataset_config_name: str
    epochs: int = 15
    learning_rate: float = 5e-5
    batch_size: int = 32  # Reduced from 128 to prevent OOM in sequential training
    max_seq_length: int = 128

    # Training state tracking
    current_epoch: int = 0
    current_step: int = 0
    best_metric: float = 0.0
    patience_counter: int = 0

    # Checkpoint management
    checkpoint_dir: str = ""
    save_strategy: str = "epoch"  # "epoch" or "steps"
    save_steps: int = 500
    eval_strategy: str = "epoch"
    eval_steps: int = 500

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Model-specific
    model_name_or_path: str = "xlm-roberta-base"
    output_dir: str = ""

    # Resource management
    bf16: bool = True  # bf16 precision enabled by default for better performance
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    def __post_init__(self):
        """Initialize derived fields after object creation."""
        if not self.output_dir:
            self.output_dir = f"iterative_training_results/{self.locale}"
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")


@dataclass
class IterativeMergeConfig:
    """Configuration for merge operations during training."""

    # Merge timing
    merge_frequency: int = 3  # Merge every N epochs

    # Merge method configuration
    merge_algorithm: str = "linear"  # "linear", "fisher_simple", "fisher_dataset"
    weight_calculation: str = "similarity"  # "similarity", "average", "manual"
    target_languages: List[str] = field(default_factory=list)

    # Similarity-based merging (if weight_calculation == "similarity")
    similarity_source: str = "sparse"  # "sparse", "dense"
    top_k: int = 20
    sinkhorn_iters: int = 20
    num_languages: int = 5

    # Fisher-based merging (if merge_algorithm contains "fisher")
    fisher_data_mode: str = "target"  # "target", "sources", "both"
    preweight: str = "equal"  # "equal", "uriel"
    num_fisher_examples: int = 100
    fisher_batch_size: int = 16
    fisher_max_seq_length: int = 128

    # Manual weights (if weight_calculation == "manual")
    manual_weights: Optional[Dict[str, float]] = None

    # Note: Merging always occurs per epoch for consistent training dynamics

    # Merge execution
    merge_device: str = "auto"  # "auto", "cpu", "cuda"
    merge_precision: str = "bf16"  # "fp32", "bf16" - bf16 by default

    # Checkpoint handling for merges
    checkpoint_before_merge: bool = True
    retain_merge_checkpoints: int = 3
    merge_output_base_dir: str = "iterative_merge_results"


@dataclass
class IterativeOrchestratorConfig:
    """Main configuration for the iterative training orchestrator."""

    # Training configurations
    training_configs: List[IterativeTrainingConfig] = field(default_factory=list)

    # Merge configuration
    merge_config: IterativeMergeConfig = field(default_factory=IterativeMergeConfig)

    # Orchestrator settings
    orchestrator_name: str = "iterative_training"
    base_output_dir: str = "iterative_training_results"
    log_level: str = "INFO"
    sequential_training: bool = True  # Train models one by one to prevent OOM

    # Synchronization settings
    max_sync_wait_time: int = 300  # seconds
    merge_timeout: int = 600  # seconds
    enable_distributed: bool = False

    # Resource management
    max-gpu-memory: Optional[int] = None  # MB, None for unlimited
    num_workers: int = 1
    pin_memory: bool = True

    # Monitoring and logging
    enable_wandb: bool = False  # Disabled by default for cleaner runs
    wandb_project: str = "MergingUriel-Iterative"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "disabled"  # "online", "offline", "disabled"
    log_frequency: int = 100  # steps

    # Recovery and robustness
    enable_auto_recovery: bool = True
    max_merge_attempts: int = 3
    validate_merge_integrity: bool = True

    # Advanced features
    adaptive_merge_frequency: bool = False
    performance_merge_trigger: bool = False
    convergence_threshold: float = 1e-4

    def __post_init__(self):
        """Initialize derived fields and validate configuration."""
        # Ensure base output directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)

        # Validate training configs
        if not self.training_configs:
            raise ValueError("At least one training configuration must be provided")

        # Set default orchestrator name if not provided
        if not self.orchestrator_name:
            self.orchestrator_name = f"iterative_training_{len(self.training_configs)}_models"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IterativeOrchestratorConfig":
        """Create configuration from a dictionary."""
        # Extract training configs
        training_configs = []
        for i, training_dict in enumerate(config_dict.get("training_configs", [])):
            training_configs.append(IterativeTrainingConfig(**training_dict))

        # Extract merge config
        merge_dict = config_dict.get("merge_config", {})
        merge_config = IterativeMergeConfig(**merge_dict)

        # Extract nested dictionaries and flatten them
        sync_dict = config_dict.get("synchronization", {})
        resource_dict = config_dict.get("resource_management", {})
        monitor_dict = config_dict.get("monitoring", {})
        recovery_dict = config_dict.get("recovery", {})
        features_dict = config_dict.get("advanced_features", {})

        # Create orchestrator config
        orchestrator_dict = config_dict.copy()
        orchestrator_dict.pop("training_configs", None)
        orchestrator_dict.pop("merge_config", None)
        orchestrator_dict.pop("synchronization", None)
        orchestrator_dict.pop("resource_management", None)
        orchestrator_dict.pop("monitoring", None)
        orchestrator_dict.pop("recovery", None)
        orchestrator_dict.pop("advanced_features", None)

        # Merge nested dictionaries
        orchestrator_dict.update(sync_dict)
        orchestrator_dict.update(resource_dict)
        orchestrator_dict.update(monitor_dict)
        orchestrator_dict.update(recovery_dict)
        orchestrator_dict.update(features_dict)

        return cls(
            training_configs=training_configs,
            merge_config=merge_config,
            **orchestrator_dict
        )

    @classmethod
    def from_args(cls, args) -> "IterativeOrchestratorConfig":
        """Create configuration from command line arguments."""
        # This will be implemented in the CLI module
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "orchestrator_name": self.orchestrator_name,
            "base_output_dir": self.base_output_dir,
            "log_level": self.log_level,
            "training_configs": [vars(tc) for tc in self.training_configs],
            "merge_config": vars(self.merge_config),
            "synchronization": {
                "max_sync_wait_time": self.max_sync_wait_time,
                "merge_timeout": self.merge_timeout,
                "enable_distributed": self.enable_distributed
            },
            "resource_management": {
                "max-gpu-memory": self.max-gpu-memory,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory
            },
            "monitoring": {
                "enable_wandb": self.enable_wandb,
                "wandb_project": self.wandb_project,
                "wandb_entity": self.wandb_entity,
                "wandb_mode": self.wandb_mode,
                "log_frequency": self.log_frequency
            },
            "recovery": {
                "enable_auto_recovery": self.enable_auto_recovery,
                "max_merge_attempts": self.max_merge_attempts,
                "validate_merge_integrity": self.validate_merge_integrity
            },
            "advanced_features": {
                "adaptive_merge_frequency": self.adaptive_merge_frequency,
                "performance_merge_trigger": self.performance_merge_trigger,
                "convergence_threshold": self.convergence_threshold
            }
        }

    def save_config(self, output_path: str):
        """Save configuration to a JSON file."""
        import json
        config_dict = self.to_dict()
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, config_path: str) -> "IterativeOrchestratorConfig":
        """Load configuration from a JSON file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_default_config(
    locales: List[str],
    base_output_dir: str = "iterative_training_results",
    merge_frequency: int = 3,
    merge_algorithm: str = "linear",
    weight_calculation: str = "similarity"
) -> IterativeOrchestratorConfig:
    """
    Create a default configuration for the given locales.

    Args:
        locales: List of locale codes to train
        base_output_dir: Base directory for outputs
        merge_frequency: Number of epochs between merges
        merge_algorithm: Merging algorithm to use
        weight_calculation: Weight calculation strategy

    Returns:
        Configured IterativeOrchestratorConfig instance
    """
    # Create training configs
    training_configs = []
    for locale in locales:
        training_configs.append(IterativeTrainingConfig(
            locale=locale,
            dataset_config_name=locale,
            output_dir=f"{base_output_dir}/{locale}",
            checkpoint_dir=f"{base_output_dir}/{locale}/checkpoints"
        ))

    # Create merge config
    merge_config = IterativeMergeConfig(
        merge_frequency=merge_frequency,
        merge_algorithm=merge_algorithm,
        weight_calculation=weight_calculation,
        target_languages=locales,
        merge_output_base_dir=f"{base_output_dir}/merged_models"
    )

    # Create orchestrator config
    return IterativeOrchestratorConfig(
        training_configs=training_configs,
        merge_config=merge_config,
        base_output_dir=base_output_dir,
        orchestrator_name=f"iterative_training_{len(locales)}_models"
    )