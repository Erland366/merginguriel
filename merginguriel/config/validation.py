"""
Configuration validation utilities.

This module provides validation functions for configuration objects,
ensuring all values are valid before experiment execution.

Usage:
    errors = validate_config(config)
    if errors:
        print("Validation errors:", errors)

    # Or raise on errors:
    validate_and_raise(config)
"""

import re
from dataclasses import fields, is_dataclass
from typing import List, Any, Optional

from .base import VALID_MODES, VALID_VOTING_METHODS


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


# Locale format: xx-XX (e.g., "sq-AL", "af-ZA")
LOCALE_PATTERN = re.compile(r"^[a-z]{2}-[A-Z]{2}$")


def validate_locale(locale: str) -> Optional[str]:
    """Validate locale format.

    Args:
        locale: Locale string to validate

    Returns:
        Error message if invalid, None if valid
    """
    if not locale:
        return "Locale cannot be empty"
    if not LOCALE_PATTERN.match(locale):
        return f"Invalid locale format '{locale}'. Expected format: 'xx-XX' (e.g., 'sq-AL')"
    return None


def validate_config(config: Any) -> List[str]:
    """Validate a configuration object.

    Performs comprehensive validation of all configuration fields.

    Args:
        config: Configuration object to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Get the class name for error messages
    class_name = type(config).__name__

    # Validate similarity config
    if hasattr(config, "similarity"):
        sim = config.similarity
        if hasattr(sim, "type") and sim.type not in ("URIEL", "REAL"):
            errors.append(f"similarity.type must be 'URIEL' or 'REAL', got '{sim.type}'")
        if hasattr(sim, "source") and sim.source not in ("sparse", "dense"):
            errors.append(f"similarity.source must be 'sparse' or 'dense', got '{sim.source}'")
        if hasattr(sim, "top_k") and sim.top_k < 1:
            errors.append(f"similarity.top_k must be >= 1, got {sim.top_k}")
        if hasattr(sim, "sinkhorn_iters") and sim.sinkhorn_iters < 0:
            errors.append(f"similarity.sinkhorn_iters must be >= 0, got {sim.sinkhorn_iters}")

    # Validate target config
    if hasattr(config, "target"):
        target = config.target
        if hasattr(target, "locale") and target.locale:
            locale_error = validate_locale(target.locale)
            if locale_error:
                errors.append(f"target.locale: {locale_error}")
        if hasattr(target, "inclusion") and target.inclusion not in ("IncTar", "ExcTar"):
            errors.append(f"target.inclusion must be 'IncTar' or 'ExcTar', got '{target.inclusion}'")

    # Validate model config
    if hasattr(config, "model"):
        model = config.model
        if hasattr(model, "num_languages") and model.num_languages < 1:
            errors.append(f"model.num_languages must be >= 1, got {model.num_languages}")

    # Validate fisher config
    if hasattr(config, "fisher"):
        fisher = config.fisher
        if hasattr(fisher, "data_mode") and fisher.data_mode not in ("target", "sources", "both"):
            errors.append(f"fisher.data_mode must be 'target', 'sources', or 'both', got '{fisher.data_mode}'")
        if hasattr(fisher, "preweight") and fisher.preweight not in ("equal", "uriel"):
            errors.append(f"fisher.preweight must be 'equal' or 'uriel', got '{fisher.preweight}'")
        if hasattr(fisher, "num_examples") and fisher.num_examples < 1:
            errors.append(f"fisher.num_examples must be >= 1, got {fisher.num_examples}")

    # Validate modes (for MergingExperimentConfig)
    if hasattr(config, "modes"):
        for mode in config.modes:
            if mode not in VALID_MODES:
                errors.append(f"Invalid mode '{mode}'. Valid modes: {VALID_MODES}")

    # Validate mode (for IterativeExperimentConfig, PipelineConfig)
    if hasattr(config, "mode") and not hasattr(config, "modes"):
        if config.mode not in VALID_MODES:
            errors.append(f"Invalid mode '{config.mode}'. Valid modes: {VALID_MODES}")

    # Validate voting methods (for EnsembleExperimentConfig)
    if hasattr(config, "voting_methods"):
        for method in config.voting_methods:
            if method not in VALID_VOTING_METHODS:
                errors.append(f"Invalid voting method '{method}'. Valid methods: {VALID_VOTING_METHODS}")

    # Validate locales list
    if hasattr(config, "locales") and config.locales is not None:
        for locale in config.locales:
            locale_error = validate_locale(locale)
            if locale_error:
                errors.append(f"locales: {locale_error}")

    # Validate target_languages list
    if hasattr(config, "target_languages") and config.target_languages is not None:
        for locale in config.target_languages:
            locale_error = validate_locale(locale)
            if locale_error:
                errors.append(f"target_languages: {locale_error}")

    # Validate training config
    if hasattr(config, "training"):
        training = config.training
        if hasattr(training, "epochs") and training.epochs < 1:
            errors.append(f"training.epochs must be >= 1, got {training.epochs}")
        if hasattr(training, "learning_rate") and training.learning_rate <= 0:
            errors.append(f"training.learning_rate must be > 0, got {training.learning_rate}")
        if hasattr(training, "batch_size") and training.batch_size < 1:
            errors.append(f"training.batch_size must be >= 1, got {training.batch_size}")

    # Validate preset (for MergingExperimentConfig)
    if hasattr(config, "preset"):
        if config.preset not in ("none", "fairness", "target"):
            errors.append(f"preset must be 'none', 'fairness', or 'target', got '{config.preset}'")

    # Validate wandb_mode (for IterativeExperimentConfig)
    if hasattr(config, "wandb_mode"):
        if config.wandb_mode not in ("online", "offline", "disabled"):
            errors.append(f"wandb_mode must be 'online', 'offline', or 'disabled', got '{config.wandb_mode}'")

    # Validate start_from and max_locales
    if hasattr(config, "start_from") and config.start_from < 0:
        errors.append(f"start_from must be >= 0, got {config.start_from}")
    if hasattr(config, "max_locales") and config.max_locales is not None and config.max_locales < 1:
        errors.append(f"max_locales must be >= 1 or None, got {config.max_locales}")

    return errors


def validate_and_raise(config: Any):
    """Validate configuration and raise if errors found.

    Args:
        config: Configuration object to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = validate_config(config)
    if errors:
        raise ConfigValidationError(errors)


def validate_required_fields(config: Any, required: List[str]) -> List[str]:
    """Validate that required fields are present and non-empty.

    Args:
        config: Configuration object to validate
        required: List of dot-separated field paths (e.g., ["target.locale"])

    Returns:
        List of error messages for missing/empty fields
    """
    from .loader import ConfigLoader

    errors = []
    for field_path in required:
        value = ConfigLoader.get_nested_attr(config, field_path)
        if value is None or value == "":
            errors.append(f"Required field '{field_path}' is missing or empty")
    return errors
