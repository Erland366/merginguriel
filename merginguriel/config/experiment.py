"""
Experiment-specific configuration classes.

This module provides top-level configuration classes for each experiment type,
composing the base configuration classes into complete experiment specifications.

Usage:
    from merginguriel.config import MergingExperimentConfig

    config = MergingExperimentConfig.from_yaml("experiment.yaml")
    print(config.model.base_model)
    print(config.similarity.type)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import (
    SimilarityConfig,
    TargetConfig,
    ModelConfig,
    DatasetConfig,
    FisherConfig,
    OutputConfig,
    TrainingConfig,
    MergeConfig,
    VALID_MODES,
    VALID_VOTING_METHODS,
)
from .loader import ConfigLoader


@dataclass
class MergingExperimentConfig:
    """Configuration for merging experiments (run_large_scale_experiment.py).

    This is the main configuration for running model merging across
    multiple locales and merging methods.

    Attributes:
        locales: List of target locales to process (None = all available)
        modes: List of merging methods to apply
        similarity: Similarity computation config
        target: Target language config
        model: Model selection config
        dataset: Dataset config
        fisher: Fisher merging config
        output: Output paths config
        preset: Experiment preset ("none", "fairness", "target")
        resume: Resume from existing results
        start_from: Index of first locale to process
        max_locales: Maximum number of locales to process (None = all)
    """
    # Target specification
    locales: Optional[List[str]] = None
    target: TargetConfig = field(default_factory=TargetConfig)

    # Methods to run
    modes: List[str] = field(default_factory=lambda: [
        "baseline", "similarity", "average", "fisher",
        "ties", "task_arithmetic", "slerp", "regmean"
    ])

    # Core configurations
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Experiment control
    preset: str = "none"
    resume: bool = True
    start_from: int = 0
    max_locales: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate modes
        for mode in self.modes:
            if mode not in VALID_MODES:
                raise ValueError(
                    f"Invalid mode '{mode}'. Valid modes are: {VALID_MODES}"
                )

        # Validate preset
        if self.preset not in ("none", "fairness", "target"):
            raise ValueError(
                f"Invalid preset '{self.preset}'. Must be 'none', 'fairness', or 'target'"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "MergingExperimentConfig":
        """Load configuration from a YAML file."""
        return ConfigLoader.from_yaml(cls, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergingExperimentConfig":
        """Create configuration from a dictionary."""
        return ConfigLoader.from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return ConfigLoader.to_dict(self)


@dataclass
class EnsembleExperimentConfig:
    """Configuration for ensemble experiments (run_large_scale_ensemble_experiments.py).

    Attributes:
        target_languages: List of target languages (None = all available)
        voting_methods: List of ensemble voting methods to use
        similarity: Similarity computation config
        target: Target language config
        model: Model selection config
        output: Output paths config
        num_examples: Number of test examples (None = all)
        ensemble_output_dir: Output directory for ensemble results
        resume: Resume from existing results
        start_from: Index of first locale to process
        max_locales: Maximum number of locales to process
    """
    target_languages: Optional[List[str]] = None
    voting_methods: List[str] = field(default_factory=lambda: [
        "majority", "weighted_majority", "soft", "uriel_logits"
    ])

    # Core configurations
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Ensemble-specific
    num_examples: Optional[int] = None
    ensemble_output_dir: str = "urie_ensemble_results"

    # Experiment control
    resume: bool = True
    start_from: int = 0
    max_locales: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        for method in self.voting_methods:
            if method not in VALID_VOTING_METHODS:
                raise ValueError(
                    f"Invalid voting method '{method}'. Valid methods are: {VALID_VOTING_METHODS}"
                )

    @classmethod
    def from_yaml(cls, path: Path) -> "EnsembleExperimentConfig":
        """Load configuration from a YAML file."""
        return ConfigLoader.from_yaml(cls, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleExperimentConfig":
        """Create configuration from a dictionary."""
        return ConfigLoader.from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return ConfigLoader.to_dict(self)


@dataclass
class IterativeExperimentConfig:
    """Configuration for iterative training experiments (run_large_scale_iterative_training.py).

    Attributes:
        target_languages: List of target languages (None = all available)
        mode: Merging mode for iterative training
        similarity: Similarity computation config
        target: Target language config
        model: Model selection config
        dataset: Dataset config
        output: Output paths config
        training: Training hyperparameters config
        merge: Merge operation config
        sequential_training: Train models one by one (prevents OOM)
        enable_wandb: Enable Weights & Biases logging
        wandb_mode: WandB mode ("online", "offline", "disabled")
        resume: Resume from existing results
        start_from: Index of first locale to process
        max_locales: Maximum number of locales to process
        max_models: Maximum number of models to train
        timeout_hours: Maximum time for experiment
        cleanup_intermediate: Clean up intermediate checkpoints
    """
    target_languages: Optional[List[str]] = None
    mode: str = "similarity"

    # Core configurations
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)

    # Orchestration
    sequential_training: bool = True
    enable_wandb: bool = False
    wandb_mode: str = "disabled"

    # Experiment control
    resume: bool = True
    start_from: int = 0
    max_locales: Optional[int] = None
    max_models: int = 5
    timeout_hours: float = 12.0
    cleanup_intermediate: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Valid modes are: {VALID_MODES}"
            )
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(
                f"Invalid wandb_mode '{self.wandb_mode}'. Must be 'online', 'offline', or 'disabled'"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "IterativeExperimentConfig":
        """Load configuration from a YAML file."""
        return ConfigLoader.from_yaml(cls, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterativeExperimentConfig":
        """Create configuration from a dictionary."""
        return ConfigLoader.from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return ConfigLoader.to_dict(self)


@dataclass
class PipelineConfig:
    """Configuration for single merging pipeline (run_merging_pipeline_refactored.py).

    This is used for running a single merge operation on one target language.

    Attributes:
        mode: Merging method to use
        target: Target language config
        similarity: Similarity computation config
        model: Model selection config
        dataset: Dataset config
        fisher: Fisher merging config
        output: Output paths config
    """
    mode: str = "similarity"
    target: TargetConfig = field(default_factory=TargetConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Valid modes are: {VALID_MODES}"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        return ConfigLoader.from_yaml(cls, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from a dictionary."""
        return ConfigLoader.from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return ConfigLoader.to_dict(self)
