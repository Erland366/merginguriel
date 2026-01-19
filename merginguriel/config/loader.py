"""
Configuration loading utilities.

This module provides utilities for loading configurations from YAML files
and merging CLI arguments with deprecation warnings.

Usage:
    config = ConfigLoader.from_yaml(MergingExperimentConfig, "experiment.yaml")
    config = ConfigLoader.merge_cli_args(config, args)
"""

import argparse
import warnings
from argparse import Namespace
from dataclasses import fields, is_dataclass, MISSING
from pathlib import Path
from typing import TypeVar, Type, Dict, Any, Optional, get_type_hints, get_origin, get_args

import yaml


class TrackProvidedArgsAction(argparse.Action):
    """Custom argparse action that tracks which arguments were explicitly provided.

    Usage:
        parser.add_argument("--foo", action=TrackProvidedArgsAction, default="bar")

    After parsing, args._provided_args will contain the set of argument names
    that were explicitly provided on the command line.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        # Track that this argument was provided
        if not hasattr(namespace, "_provided_args"):
            namespace._provided_args = set()
        namespace._provided_args.add(self.dest)


T = TypeVar("T")


class ConfigDeprecationWarning(DeprecationWarning):
    """Warning for deprecated CLI arguments."""
    pass


# Ensure deprecation warnings are always shown
warnings.filterwarnings("always", category=ConfigDeprecationWarning)


def emit_cli_deprecation(arg_name: str, config_path: str):
    """Emit deprecation warning for CLI argument usage.

    Args:
        arg_name: Name of the CLI argument
        config_path: Path in config file (e.g., "model.base_model")
    """
    warnings.warn(
        f"CLI argument '--{arg_name}' is deprecated. "
        f"Use config file with '{config_path}' instead. "
        f"CLI value will override config file for backward compatibility.",
        ConfigDeprecationWarning,
        stacklevel=4
    )


class ConfigLoader:
    """Utility class for loading and manipulating configurations."""

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """Load a YAML file and return as dictionary.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary containing YAML contents

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return data if data is not None else {}

    @staticmethod
    def save_yaml(data: Dict[str, Any], path: Path):
        """Save dictionary to YAML file.

        Args:
            data: Dictionary to save
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def from_yaml(config_class: Type[T], path: Path) -> T:
        """Create a configuration instance from a YAML file.

        Args:
            config_class: The dataclass type to instantiate
            path: Path to YAML file

        Returns:
            Instance of config_class populated from YAML
        """
        data = ConfigLoader.load_yaml(path)
        return ConfigLoader.from_dict(config_class, data)

    @staticmethod
    def from_dict(config_class: Type[T], data: Dict[str, Any]) -> T:
        """Create a configuration instance from a dictionary.

        Recursively instantiates nested dataclasses.

        Args:
            config_class: The dataclass type to instantiate
            data: Dictionary of configuration values

        Returns:
            Instance of config_class
        """
        if not is_dataclass(config_class):
            raise TypeError(f"{config_class} is not a dataclass")

        # Get type hints for the class
        type_hints = get_type_hints(config_class)
        init_kwargs = {}

        for fld in fields(config_class):
            field_name = fld.name
            field_type = type_hints.get(field_name, fld.type)

            if field_name in data:
                value = data[field_name]

                # Handle nested dataclasses
                if is_dataclass(field_type) and isinstance(value, dict):
                    value = ConfigLoader.from_dict(field_type, value)
                # Handle Optional[dataclass]
                elif get_origin(field_type) is type(None) or str(field_type).startswith("typing.Optional"):
                    args = get_args(field_type)
                    if args and is_dataclass(args[0]) and isinstance(value, dict):
                        value = ConfigLoader.from_dict(args[0], value)

                init_kwargs[field_name] = value
            elif fld.default is not MISSING:
                # Use default value
                pass
            elif fld.default_factory is not MISSING:
                # Use default factory
                pass
            # else: required field without default - will raise in constructor

        return config_class(**init_kwargs)

    @staticmethod
    def to_dict(config: Any) -> Dict[str, Any]:
        """Convert a configuration instance to a dictionary.

        Recursively converts nested dataclasses.

        Args:
            config: Configuration instance

        Returns:
            Dictionary representation
        """
        if not is_dataclass(config):
            return config

        result = {}
        for fld in fields(config):
            value = getattr(config, fld.name)
            if is_dataclass(value):
                value = ConfigLoader.to_dict(value)
            elif isinstance(value, list):
                value = [
                    ConfigLoader.to_dict(item) if is_dataclass(item) else item
                    for item in value
                ]
            result[fld.name] = value

        return result

    @staticmethod
    def merge_cli_args(
        config: T,
        args: Namespace,
        emit_warnings: bool = True,
        arg_to_config_map: Optional[Dict[str, str]] = None
    ) -> T:
        """Merge CLI arguments into a configuration.

        CLI arguments override config file values. Emits deprecation warnings
        for each CLI argument used when emit_warnings is True.

        Args:
            config: Configuration instance to update
            args: Parsed CLI arguments
            emit_warnings: Whether to emit deprecation warnings
            arg_to_config_map: Mapping from CLI arg names to config paths
                               (e.g., {"similarity_type": "similarity.type"})

        Returns:
            Updated configuration instance
        """
        if arg_to_config_map is None:
            arg_to_config_map = {}

        # Get the namespace as a dictionary
        args_dict = vars(args)

        # Track which args were explicitly provided (not default)
        # This requires the parser to track defaults, which we do via a custom action
        provided_args = getattr(args, "_provided_args", set())

        for arg_name, arg_value in args_dict.items():
            # Skip internal attributes
            if arg_name.startswith("_"):
                continue

            # Skip if not explicitly provided
            if arg_name not in provided_args:
                continue

            # Get config path
            config_path = arg_to_config_map.get(arg_name, arg_name)

            # Emit warning if enabled
            if emit_warnings:
                emit_cli_deprecation(arg_name, config_path)

            # Set the value in config
            config = ConfigLoader._set_nested_attr(config, config_path, arg_value)

        return config

    @staticmethod
    def _set_nested_attr(obj: Any, path: str, value: Any) -> Any:
        """Set a nested attribute using dot notation.

        Args:
            obj: Object to modify
            path: Dot-separated path (e.g., "model.base_model")
            value: Value to set

        Returns:
            Modified object
        """
        parts = path.split(".")

        if len(parts) == 1:
            if hasattr(obj, parts[0]):
                setattr(obj, parts[0], value)
        else:
            nested_obj = getattr(obj, parts[0], None)
            if nested_obj is not None:
                ConfigLoader._set_nested_attr(nested_obj, ".".join(parts[1:]), value)

        return obj

    @staticmethod
    def get_nested_attr(obj: Any, path: str, default: Any = None) -> Any:
        """Get a nested attribute using dot notation.

        Args:
            obj: Object to query
            path: Dot-separated path (e.g., "model.base_model")
            default: Default value if path doesn't exist

        Returns:
            Attribute value or default
        """
        parts = path.split(".")
        current = obj

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return default

        return current


# CLI argument to config path mappings for each experiment type
MERGING_EXPERIMENT_ARG_MAP = {
    # Similarity config
    "similarity_type": "similarity.type",
    "similarity_source": "similarity.source",
    "top_k": "similarity.top_k",
    "sinkhorn_iters": "similarity.sinkhorn_iters",
    # Target config
    "target_lang": "target.locale",
    "target_inclusion": "target.inclusion",
    "include_target": "target.inclusion",  # Legacy - maps True to "IncTar"
    # Model config
    "base_model": "model.base_model",
    "models_root": "model.models_root",
    "num_languages": "model.num_languages",
    # Dataset config
    "dataset_name": "dataset.name",
    "dataset_split": "dataset.split",
    "text_column": "dataset.text_column",
    # Fisher config
    "fisher_data_mode": "fisher.data_mode",
    "preweight": "fisher.preweight",
    "num_fisher_examples": "fisher.num_examples",
    "batch_size": "fisher.batch_size",
    "max_seq_length": "fisher.max_seq_length",
    # Output config
    "results_dir": "output.results_dir",
    "merged_models_dir": "output.merged_models_dir",
    "cleanup_after_eval": "output.cleanup_after_eval",
}

ENSEMBLE_EXPERIMENT_ARG_MAP = {
    **MERGING_EXPERIMENT_ARG_MAP,
    "num_examples": "num_examples",
    "output_dir": "ensemble_output_dir",
}

ITERATIVE_EXPERIMENT_ARG_MAP = {
    **MERGING_EXPERIMENT_ARG_MAP,
    # Training config
    "epochs": "training.epochs",
    "learning_rate": "training.learning_rate",
    "bf16": "training.bf16",
    # Merge config
    "merge_frequency": "merge.frequency",
    "merge_algorithm": "merge.algorithm",
    # Orchestration
    "sequential_training": "sequential_training",
    "enable_wandb": "enable_wandb",
    "wandb_mode": "wandb_mode",
    "timeout_hours": "timeout_hours",
}
