"""
SelectiveLayerMerger: Merge models while excluding specific layers.

Strategy:
1. Run merge with exclude_param_names_regex for excluded layers
2. Copy excluded layer parameters from the "best source" model

This enables leave-one-layer-out ablation to identify which layers
contribute positively vs cause interference in cross-lingual transfer.
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add submodule to path if not already there
_project_root = Path(__file__).parent.parent.parent
_submodule_path = _project_root / "submodules" / "auto_merge_llm"
if str(_submodule_path) not in sys.path:
    sys.path.insert(0, str(_submodule_path))

from merginguriel.selective_layer.layer_masking import (
    generate_layer_exclude_regex,
    get_layer_params_from_state_dict,
    NUM_LAYERS,
)


@dataclass
class SelectiveMergeResult:
    """Result of a selective layer merge operation."""

    model: Any  # The merged model
    tokenizer: Any  # The tokenizer
    exclude_layers: List[int]
    exclude_patterns: List[str]
    best_source_path: Optional[str]
    copied_params_count: int
    metadata: Dict[str, Any]


def find_best_source(
    holdout_locale: str,
    source_locales: List[str],
    nxn_matrix_path: str,
) -> Tuple[str, float]:
    """
    Find the best single source model for a holdout language.

    Uses the NxN evaluation matrix to find which source model performs
    best on the holdout locale (zero-shot cross-lingual transfer).

    Args:
        holdout_locale: The target/holdout locale (e.g., "sw-KE")
        source_locales: List of candidate source locales
        nxn_matrix_path: Path to NxN evaluation matrix CSV

    Returns:
        Tuple of (best_source_locale, accuracy)

    Example:
        >>> best, acc = find_best_source("sw-KE", ["en-US", "de-DE", "fr-FR"], "nxn.csv")
        >>> print(f"Best source for sw-KE: {best} with accuracy {acc:.4f}")
    """
    if not os.path.exists(nxn_matrix_path):
        raise FileNotFoundError(f"NxN matrix not found: {nxn_matrix_path}")

    df = pd.read_csv(nxn_matrix_path, index_col=0)

    # Validate holdout locale exists in matrix
    if holdout_locale not in df.columns:
        raise ValueError(
            f"Holdout locale '{holdout_locale}' not found in matrix columns. "
            f"Available: {list(df.columns)[:10]}..."
        )

    # Find best source
    best_source = None
    best_accuracy = -1.0

    for source in source_locales:
        if source not in df.index:
            print(f"  Warning: Source locale '{source}' not in matrix index, skipping")
            continue

        acc = df.loc[source, holdout_locale]
        if acc > best_accuracy:
            best_accuracy = acc
            best_source = source

    if best_source is None:
        raise ValueError(
            f"No valid source found for holdout '{holdout_locale}' "
            f"from candidates: {source_locales}"
        )

    return best_source, best_accuracy


def copy_layers_from_source(
    target_model: torch.nn.Module,
    source_model_path: str,
    layers_to_copy: List[int],
    device: str = "cpu",
) -> int:
    """
    Copy specific layer parameters from source model to target model.

    Args:
        target_model: The model to copy parameters INTO
        source_model_path: Path to source model to copy FROM
        layers_to_copy: List of layer indices to copy (0-11)
        device: Device for loading source model

    Returns:
        Number of parameters copied
    """
    if not layers_to_copy:
        return 0

    # Load source model
    source_model = AutoModelForSequenceClassification.from_pretrained(
        source_model_path,
        device_map=device,
    )

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    copied_count = 0
    for layer_id in layers_to_copy:
        if not 0 <= layer_id < NUM_LAYERS:
            print(f"  Warning: Layer {layer_id} out of range, skipping")
            continue

        layer_prefix = f"roberta.encoder.layer.{layer_id}."
        for param_name, param_value in source_state.items():
            if param_name.startswith(layer_prefix):
                if param_name in target_state:
                    target_state[param_name] = param_value.clone()
                    copied_count += 1
                else:
                    print(f"  Warning: Param '{param_name}' not in target model")

    target_model.load_state_dict(target_state)

    # Clean up source model
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return copied_count


class SelectiveLayerMerger:
    """
    Performs selective layer merging:
    - Merges specified layers across source models
    - Copies excluded layers from the best single source
    """

    def __init__(
        self,
        exclude_layers: List[int],
        best_source_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            exclude_layers: Layer indices to EXCLUDE from merging (0-11)
            best_source_path: Path to best single source model for excluded layers.
                If None, excluded layers will retain base model values.
            device: Device for model operations
        """
        self.exclude_layers = exclude_layers
        self.best_source_path = best_source_path
        self.device = device

        # Generate exclusion patterns
        self.exclude_patterns = generate_layer_exclude_regex(exclude_layers)

    def get_exclude_patterns(self) -> List[str]:
        """Get the regex patterns for excluding layers from merge."""
        return self.exclude_patterns

    def post_merge_copy(
        self,
        merged_model: torch.nn.Module,
    ) -> Tuple[torch.nn.Module, int]:
        """
        Copy excluded layers from best source after merge.

        Args:
            merged_model: The merged model (post standard merge)

        Returns:
            Tuple of (model with copied layers, count of copied params)
        """
        if not self.exclude_layers or not self.best_source_path:
            return merged_model, 0

        print(f"  Copying layers {self.exclude_layers} from best source: {self.best_source_path}")
        copied_count = copy_layers_from_source(
            target_model=merged_model,
            source_model_path=self.best_source_path,
            layers_to_copy=self.exclude_layers,
            device=self.device,
        )
        print(f"  Copied {copied_count} parameters")

        return merged_model, copied_count


def run_selective_merge_experiment(
    base_model_path: str,
    models_to_merge: List[str],
    weights: List[float],
    exclude_layers: List[int],
    best_source_path: Optional[str],
    merge_method: str = "linear",
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> SelectiveMergeResult:
    """
    Run a complete selective merge experiment.

    This is a standalone function that performs selective layer merging
    without depending on the full MergingPipeline infrastructure.

    Args:
        base_model_path: Path to base model (architecture reference)
        models_to_merge: Paths to models to merge
        weights: Weights for each model
        exclude_layers: Layers to exclude from merge
        best_source_path: Path to best source for excluded layers
        merge_method: Merging method ("linear", "ties", etc.)
        output_dir: Optional directory to save merged model
        device: Device for operations

    Returns:
        SelectiveMergeResult with merged model and metadata
    """
    from auto_merge_llm.methods import merging_methods_dict

    # Create selective merger
    selective_merger = SelectiveLayerMerger(
        exclude_layers=exclude_layers,
        best_source_path=best_source_path,
        device=device,
    )

    # Get the underlying merge method
    if merge_method not in merging_methods_dict:
        raise ValueError(f"Unknown merge method: {merge_method}. Available: {list(merging_methods_dict.keys())}")

    merger = merging_methods_dict[merge_method]()

    # Prepare method params
    method_params = {"weights": weights}

    # Perform merge with exclusion patterns
    print(f"Performing {merge_method} merge with {len(exclude_layers)} excluded layers...")
    result = merger.merge(
        base_model=base_model_path,
        models_to_merge=models_to_merge,
        method_params=method_params,
        exclude_param_names_regex=selective_merger.get_exclude_patterns(),
    )

    merged_model = result["merged_model"]
    tokenizer = result["base_tokenizer"]

    # Copy excluded layers from best source
    merged_model, copied_count = selective_merger.post_merge_copy(merged_model)

    # Optionally save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved merged model to: {output_dir}")

    return SelectiveMergeResult(
        model=merged_model,
        tokenizer=tokenizer,
        exclude_layers=exclude_layers,
        exclude_patterns=selective_merger.get_exclude_patterns(),
        best_source_path=best_source_path,
        copied_params_count=copied_count,
        metadata={
            "base_model": base_model_path,
            "models_merged": models_to_merge,
            "weights": weights,
            "merge_method": merge_method,
        },
    )


def evaluate_selective_merge(
    merge_result: SelectiveMergeResult,
    eval_locale: str,
    eval_split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate a selective merge result on a specific locale.

    Args:
        merge_result: Result from selective merge
        eval_locale: Locale to evaluate on (e.g., "sw-KE")
        eval_split: Dataset split ("test", "validation")

    Returns:
        Evaluation results dict with accuracy and metadata
    """
    from merginguriel.evaluate_specific_model import evaluate_specific_model

    # Save model to temp dir for evaluation
    with tempfile.TemporaryDirectory() as tmp_dir:
        merge_result.model.save_pretrained(tmp_dir)
        merge_result.tokenizer.save_pretrained(tmp_dir)

        # Run evaluation
        results = evaluate_specific_model(
            model_name=tmp_dir,
            locale=eval_locale,
        )

    if results:
        results["selective_merge_metadata"] = {
            "exclude_layers": merge_result.exclude_layers,
            "best_source": merge_result.best_source_path,
            "copied_params": merge_result.copied_params_count,
        }

    return results
