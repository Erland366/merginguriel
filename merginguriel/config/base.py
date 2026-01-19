"""
Base configuration classes for MergingUriel.

This module provides reusable, composable configuration dataclasses
that are shared across different experiment types.

Usage:
    config = MergingExperimentConfig.from_yaml("experiment.yaml")
    print(config.model.base_model)        # Hierarchical attribute access
    print(config.similarity.type)         # "URIEL"
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path


@dataclass
class SimilarityConfig:
    """Configuration for similarity computation.

    Attributes:
        type: Source of similarity scores - "URIEL" (linguistic features)
              or "REAL" (empirical evaluation results)
        source: Matrix type - "sparse" (precomputed) or "dense" (on-the-fly)
        top_k: Number of top similar languages to consider for normalization
        sinkhorn_iters: Number of Sinkhorn normalization iterations
    """
    type: Literal["URIEL", "REAL"] = "URIEL"
    source: Literal["sparse", "dense"] = "dense"
    top_k: int = 20
    sinkhorn_iters: int = 20

    def __post_init__(self):
        if self.type not in ("URIEL", "REAL"):
            raise ValueError(f"similarity.type must be 'URIEL' or 'REAL', got '{self.type}'")
        if self.source not in ("sparse", "dense"):
            raise ValueError(f"similarity.source must be 'sparse' or 'dense', got '{self.source}'")
        if self.top_k < 1:
            raise ValueError(f"similarity.top_k must be >= 1, got {self.top_k}")
        if self.sinkhorn_iters < 0:
            raise ValueError(f"similarity.sinkhorn_iters must be >= 0, got {self.sinkhorn_iters}")


@dataclass
class TargetConfig:
    """Configuration for target language.

    Attributes:
        locale: Target language locale code (e.g., "sq-AL", "af-ZA")
        inclusion: Whether to include target in merging - "IncTar" (include)
                   or "ExcTar" (exclude)
    """
    locale: str = ""
    inclusion: Literal["IncTar", "ExcTar"] = "ExcTar"

    def __post_init__(self):
        if self.inclusion not in ("IncTar", "ExcTar"):
            raise ValueError(f"target.inclusion must be 'IncTar' or 'ExcTar', got '{self.inclusion}'")

    @property
    def include_target(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.inclusion == "IncTar"


@dataclass
class ModelConfig:
    """Configuration for model selection.

    Attributes:
        base_model: Base model architecture ("xlm-roberta-base" or "xlm-roberta-large")
        models_root: Root directory containing fine-tuned models
        num_languages: Number of source languages to include in merge
    """
    base_model: str = "xlm-roberta-base"
    models_root: str = "haryos_model"
    num_languages: int = 5

    def __post_init__(self):
        if self.num_languages < 1:
            raise ValueError(f"model.num_languages must be >= 1, got {self.num_languages}")


@dataclass
class DatasetConfig:
    """Configuration for dataset.

    Attributes:
        name: HuggingFace dataset name
        split: Dataset split to use
        text_column: Column name for input text
        label_column: Column name for labels
    """
    name: str = "AmazonScience/massive"
    split: str = "train"
    text_column: str = "utt"
    label_column: str = "label"


@dataclass
class FisherConfig:
    """Configuration for Fisher-based merging.

    Attributes:
        data_mode: Source of Fisher examples - "target", "sources", or "both"
        preweight: Pre-weighting strategy - "equal" or "uriel"
        num_examples: Number of examples for Fisher computation
        batch_size: Batch size for Fisher computation
        max_seq_length: Maximum sequence length for tokenization
    """
    data_mode: Literal["target", "sources", "both"] = "target"
    preweight: Literal["equal", "uriel"] = "equal"
    num_examples: int = 1000
    batch_size: int = 16
    max_seq_length: int = 128

    def __post_init__(self):
        if self.data_mode not in ("target", "sources", "both"):
            raise ValueError(f"fisher.data_mode must be 'target', 'sources', or 'both', got '{self.data_mode}'")
        if self.preweight not in ("equal", "uriel"):
            raise ValueError(f"fisher.preweight must be 'equal' or 'uriel', got '{self.preweight}'")
        if self.num_examples < 1:
            raise ValueError(f"fisher.num_examples must be >= 1, got {self.num_examples}")


@dataclass
class OutputConfig:
    """Configuration for output paths and cleanup.

    Attributes:
        results_dir: Directory for saving results
        merged_models_dir: Directory for saving merged models
        cleanup_after_eval: Whether to delete merged models after evaluation
    """
    results_dir: str = "results"
    merged_models_dir: str = "merged_models"
    cleanup_after_eval: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
        max_seq_length: Maximum sequence length
        bf16: Use bfloat16 precision
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        save_strategy: Checkpoint save strategy ("epoch" or "steps")
        eval_strategy: Evaluation strategy ("epoch" or "steps")
        early_stopping_patience: Early stopping patience (epochs/steps)
    """
    epochs: int = 15
    learning_rate: float = 5e-5
    batch_size: int = 32
    max_seq_length: int = 128
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    early_stopping_patience: int = 3


@dataclass
class MergeConfig:
    """Configuration for merge operations during iterative training.

    Attributes:
        frequency: Merge every N epochs
        algorithm: Merge algorithm ("linear", "fisher_simple", "fisher_dataset")
        weight_calculation: Weight calculation strategy
        checkpoint_before_merge: Save checkpoint before merging
        retain_checkpoints: Number of merge checkpoints to retain
    """
    frequency: int = 3
    algorithm: str = "linear"
    weight_calculation: str = "similarity"
    checkpoint_before_merge: bool = True
    retain_checkpoints: int = 3


# Valid merging modes
VALID_MODES = [
    "baseline",
    "similarity",
    "average",
    "fisher",
    "fisher_dataset",
    "fisher_simple",
    "uriel",
    "manual",
    "ties",
    "task_arithmetic",
    "slerp",
    "regmean",
    "iterative",
]

# Valid voting methods for ensemble
VALID_VOTING_METHODS = [
    "majority",
    "weighted_majority",
    "soft",
    "uriel_logits",
]
