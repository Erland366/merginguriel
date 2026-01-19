"""
MergingUriel Configuration System.

This package provides a unified YAML-based configuration system for all
MergingUriel experiments. Configuration uses hierarchical attribute access
for clean, type-safe configuration management.

Usage:
    from merginguriel.config import MergingExperimentConfig

    # Load from YAML
    config = MergingExperimentConfig.from_yaml("experiment.yaml")

    # Access with hierarchical attributes
    print(config.model.base_model)        # "xlm-roberta-base"
    print(config.similarity.type)         # "URIEL"
    print(config.target.locale)           # "sq-AL"

Example YAML:
    target:
      locale: "sq-AL"
      inclusion: "ExcTar"

    modes:
      - baseline
      - similarity

    similarity:
      type: "URIEL"
      top_k: 20

    model:
      base_model: "xlm-roberta-base"
      num_languages: 5
"""

# Base configuration classes
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

# Experiment-specific configurations
from .experiment import (
    MergingExperimentConfig,
    EnsembleExperimentConfig,
    IterativeExperimentConfig,
    PipelineConfig,
)

# Loader utilities
from .loader import (
    ConfigLoader,
    ConfigDeprecationWarning,
    TrackProvidedArgsAction,
    emit_cli_deprecation,
    MERGING_EXPERIMENT_ARG_MAP,
    ENSEMBLE_EXPERIMENT_ARG_MAP,
    ITERATIVE_EXPERIMENT_ARG_MAP,
)

# Validation utilities
from .validation import (
    validate_config,
    validate_and_raise,
    validate_required_fields,
    validate_locale,
    ConfigValidationError,
)

__all__ = [
    # Base configs
    "SimilarityConfig",
    "TargetConfig",
    "ModelConfig",
    "DatasetConfig",
    "FisherConfig",
    "OutputConfig",
    "TrainingConfig",
    "MergeConfig",
    "VALID_MODES",
    "VALID_VOTING_METHODS",
    # Experiment configs
    "MergingExperimentConfig",
    "EnsembleExperimentConfig",
    "IterativeExperimentConfig",
    "PipelineConfig",
    # Loader
    "ConfigLoader",
    "ConfigDeprecationWarning",
    "TrackProvidedArgsAction",
    "emit_cli_deprecation",
    "MERGING_EXPERIMENT_ARG_MAP",
    "ENSEMBLE_EXPERIMENT_ARG_MAP",
    "ITERATIVE_EXPERIMENT_ARG_MAP",
    # Validation
    "validate_config",
    "validate_and_raise",
    "validate_required_fields",
    "validate_locale",
    "ConfigValidationError",
]
