"""
Layer-swapping ablation utilities.

Generates ablation configurations and selects source model pairs
for systematic layer-swapping experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from merginguriel.layer_swapping import LayerSwappingConfig
from merginguriel.selective_layer.layer_masking import LAYER_GROUPS, NUM_LAYERS


# ---------------------------------------------------------------------------
# Ablation config generation
# ---------------------------------------------------------------------------

def generate_ablation_configs(
    ablation_type: str = "full",
    num_layers: int = NUM_LAYERS,
) -> list[tuple[str, LayerSwappingConfig]]:
    """Generate named ablation configurations.

    Args:
        ablation_type: One of "top_bottom_sweep", "single_layer",
            "layer_groups", "non_layer_source", "full".
        num_layers: Number of encoder layers in the model.

    Returns:
        List of (experiment_name, LayerSwappingConfig) tuples.
    """
    configs: list[tuple[str, LayerSwappingConfig]] = []

    if ablation_type in ("top_bottom_sweep", "full"):
        # Symmetric top+bottom sweeps
        for k in range(1, num_layers // 2 + 1):
            configs.append((
                f"top{k}_bottom{k}",
                LayerSwappingConfig(
                    swap_strategy="top_bottom", top_k=k, bottom_k=k,
                    non_layer_source="model_a",
                ),
            ))
        # Top-only
        for k in range(1, num_layers // 2 + 1):
            configs.append((
                f"top{k}_bottom0",
                LayerSwappingConfig(
                    swap_strategy="top_bottom", top_k=k, bottom_k=0,
                    non_layer_source="model_a",
                ),
            ))
        # Bottom-only
        for k in range(1, num_layers // 2 + 1):
            configs.append((
                f"top0_bottom{k}",
                LayerSwappingConfig(
                    swap_strategy="top_bottom", top_k=0, bottom_k=k,
                    non_layer_source="model_a",
                ),
            ))

    if ablation_type in ("single_layer", "full"):
        for layer_id in range(num_layers):
            configs.append((
                f"single_layer_{layer_id}",
                LayerSwappingConfig(
                    swap_strategy="custom",
                    custom_layers=[layer_id],
                    non_layer_source="model_a",
                ),
            ))

    if ablation_type in ("layer_groups", "full"):
        for group_name, layers in LAYER_GROUPS.items():
            configs.append((
                f"group_{group_name}",
                LayerSwappingConfig(
                    swap_strategy="custom",
                    custom_layers=list(layers),
                    non_layer_source="model_a",
                ),
            ))
        # Combined top+bottom (paper default pattern)
        top_bottom = list(LAYER_GROUPS["bottom"]) + list(LAYER_GROUPS["top"])
        configs.append((
            "group_top_bottom",
            LayerSwappingConfig(
                swap_strategy="custom",
                custom_layers=top_bottom,
                non_layer_source="model_a",
            ),
        ))

    if ablation_type in ("non_layer_source", "full"):
        configs.append((
            "nls_model_a__top2_bottom2",
            LayerSwappingConfig(
                swap_strategy="top_bottom", top_k=2, bottom_k=2,
                non_layer_source="model_a",
            ),
        ))
        configs.append((
            "nls_model_b__top2_bottom2",
            LayerSwappingConfig(
                swap_strategy="top_bottom", top_k=2, bottom_k=2,
                non_layer_source="model_b",
            ),
        ))

    return configs


# ---------------------------------------------------------------------------
# Source model pair selection
# ---------------------------------------------------------------------------

def load_nxn_matrix(nxn_matrix_path: str) -> pd.DataFrame:
    """Load NxN evaluation matrix (rows=source, columns=target)."""
    df = pd.read_csv(nxn_matrix_path, index_col=0)
    return df


def select_source_pairs(
    target_locale: str,
    strategy: str,
    nxn_matrix_path: str,
    *,
    model_a_locale: str | None = None,
    model_b_locale: str | None = None,
) -> list[tuple[str, str]]:
    """Select (model_a, model_b) source pairs for layer-swapping.

    Args:
        target_locale: Target locale to evaluate on (excluded from sources).
        strategy: One of "nxn_top2", "nxn_best_plus_diverse", "manual".
        nxn_matrix_path: Path to NxN evaluation matrix CSV.
        model_a_locale: Required for "manual" strategy.
        model_b_locale: Required for "manual" strategy.

    Returns:
        List of (model_a_locale, model_b_locale) pairs.
    """
    if strategy == "manual":
        if not model_a_locale or not model_b_locale:
            raise ValueError("manual strategy requires model_a_locale and model_b_locale")
        return [(model_a_locale, model_b_locale)]

    df = load_nxn_matrix(nxn_matrix_path)

    if target_locale not in df.columns:
        raise ValueError(f"Target locale {target_locale} not found in NxN matrix")

    # Get performance of all sources on target (exclude target itself)
    scores = df[target_locale].copy()
    if target_locale in scores.index:
        scores = scores.drop(target_locale)

    top_sources = scores.nlargest(5)
    pairs: list[tuple[str, str]] = []

    if strategy == "nxn_top2":
        if len(top_sources) < 2:
            raise ValueError(f"Not enough sources for target {target_locale}")
        best = top_sources.index[0]
        second = top_sources.index[1]
        pairs.append((best, second))
        # Role swap
        pairs.append((second, best))

    elif strategy == "nxn_best_plus_diverse":
        best = top_sources.index[0]
        # Pick the source from top-5 that is most different from best
        # Use NxN: find source whose performance profile differs most from best
        best_profile = df.loc[best]
        max_dist = -1.0
        diverse = top_sources.index[1]
        for candidate in top_sources.index[1:]:
            cand_profile = df.loc[candidate]
            dist = float(((best_profile - cand_profile) ** 2).mean())
            if dist > max_dist:
                max_dist = dist
                diverse = candidate
        pairs.append((best, diverse))
        pairs.append((diverse, best))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return pairs


# ---------------------------------------------------------------------------
# Model path utilities
# ---------------------------------------------------------------------------

def get_model_path(
    locale: str,
    models_root: str = "haryos_model",
    base_model: str = "xlm-roberta-base",
) -> str:
    """Construct model path from locale."""
    import os
    return os.path.join(models_root, f"{base_model}_massive_k_{locale}")


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    experiment_name: str
    model_a_locale: str
    model_b_locale: str
    target_locale: str
    swap_strategy: str
    swap_indices: list[int]
    non_layer_source: str
    accuracy: float | None = None
    baseline_a_accuracy: float | None = None
    baseline_b_accuracy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_vs_best_baseline(self) -> float | None:
        if self.accuracy is None:
            return None
        baselines = [b for b in [self.baseline_a_accuracy, self.baseline_b_accuracy] if b is not None]
        if not baselines:
            return None
        return self.accuracy - max(baselines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "model_a_locale": self.model_a_locale,
            "model_b_locale": self.model_b_locale,
            "target_locale": self.target_locale,
            "swap_strategy": self.swap_strategy,
            "swap_indices": self.swap_indices,
            "non_layer_source": self.non_layer_source,
            "accuracy": self.accuracy,
            "baseline_a_accuracy": self.baseline_a_accuracy,
            "baseline_b_accuracy": self.baseline_b_accuracy,
            "delta_vs_best_baseline": self.delta_vs_best_baseline,
        }


def aggregate_results(results: list[AblationResult]) -> pd.DataFrame:
    """Aggregate ablation results into a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


# ---------------------------------------------------------------------------
# Method comparison: layer-swapping vs conventional merging
# ---------------------------------------------------------------------------

@dataclass
class MethodComparisonConfig:
    """Configuration for a conventional merging baseline."""
    method: str  # 'average', 'ties', 'task_arithmetic'
    dare_enabled: bool = False
    dare_drop_rate: float = 0.9
    dare_rescale: bool = True
    dare_seed: int | None = 42


def generate_method_comparison_configs() -> list[tuple[str, MethodComparisonConfig]]:
    """Generate configs for conventional merging baselines.

    Returns:
        List of (experiment_name, MethodComparisonConfig) tuples.
    """
    return [
        ("baseline_average", MethodComparisonConfig(method="average")),
        ("baseline_ties", MethodComparisonConfig(method="ties")),
        ("baseline_task_arithmetic", MethodComparisonConfig(method="task_arithmetic")),
        (
            "baseline_dare_task_arithmetic",
            MethodComparisonConfig(
                method="task_arithmetic",
                dare_enabled=True,
                dare_drop_rate=0.9,
                dare_seed=42,
            ),
        ),
    ]


def run_conventional_merge(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    config: MethodComparisonConfig,
    base_model: str = "xlm-roberta-base",
) -> dict[str, Any]:
    """Run a conventional 2-model merge using auto_merge_llm.

    Args:
        model_a_path: Path to first fine-tuned model.
        model_b_path: Path to second fine-tuned model.
        output_path: Where to save the merged model.
        config: Method comparison configuration.
        base_model: Pretrained base model name (for task-vector methods).

    Returns:
        Dict with merge metadata.
    """
    import os
    import sys
    import shutil
    import tempfile

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    submodule_path = os.path.join(project_root, "submodules/auto_merge_llm")
    if submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)

    from auto_merge_llm.methods import merging_methods_dict

    TASK_VECTOR_METHODS = {"ties", "task_arithmetic"}
    models_to_merge = [model_a_path, model_b_path]

    if config.method in TASK_VECTOR_METHODS:
        # Task-vector methods need a pretrained base with matching classifier head
        from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

        ref_config = AutoConfig.from_pretrained(model_a_path)
        num_labels = ref_config.num_labels

        pretrained_dir = tempfile.mkdtemp(prefix="pretrained_base_")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            model.save_pretrained(pretrained_dir)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(pretrained_dir)

            merger = merging_methods_dict[config.method]()
            method_params: dict[str, Any] = {
                "scaling_coefficient": 1.0,
                "param_value_mask_rate": 0.8 if config.method == "ties" else 0.0,
                "dare_enabled": config.dare_enabled,
                "dare_drop_rate": config.dare_drop_rate,
                "dare_rescale": config.dare_rescale,
                "dare_seed": config.dare_seed,
            }

            result = merger.merge(
                base_model=pretrained_dir,
                models_to_merge=models_to_merge,
                method_params=method_params,
            )
        finally:
            shutil.rmtree(pretrained_dir, ignore_errors=True)
    else:
        # Average / linear: base_model is model_a, merge with model_b
        merger_cls = merging_methods_dict.get(config.method)
        if merger_cls is None:
            merger_cls = merging_methods_dict["linear"]
        merger = merger_cls()

        method_params = {"weights": [0.5, 0.5]}
        result = merger.merge(
            base_model=model_a_path,
            models_to_merge=models_to_merge,
            method_params=method_params,
        )

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model = result["merged_model"]
    tokenizer = result["base_tokenizer"]
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return {
        "method": config.method,
        "dare_enabled": config.dare_enabled,
        "dare_drop_rate": config.dare_drop_rate if config.dare_enabled else None,
        "model_a": model_a_path,
        "model_b": model_b_path,
    }
