"""
Layer masking utilities for XLM-RoBERTa selective layer merging.

XLM-RoBERTa layer parameter naming convention:
- roberta.encoder.layer.{0-11}.attention.self.{query,key,value}.{weight,bias}
- roberta.encoder.layer.{0-11}.attention.output.dense.{weight,bias}
- roberta.encoder.layer.{0-11}.attention.output.LayerNorm.{weight,bias}
- roberta.encoder.layer.{0-11}.intermediate.dense.{weight,bias}
- roberta.encoder.layer.{0-11}.output.dense.{weight,bias}
- roberta.encoder.layer.{0-11}.output.LayerNorm.{weight,bias}
"""

from typing import Dict, List, Set

# XLM-RoBERTa layer pattern - matches all parameters in a specific encoder layer
XLM_ROBERTA_LAYER_PATTERN = r"roberta\.encoder\.layer\.{layer_id}\."

# Layer groups based on linguistic analysis literature:
# - Bottom layers: lexical/phonological representations
# - Middle layers: syntactic processing
# - Top layers: semantic/task-specific features
LAYER_GROUPS = {
    "bottom": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "top": [8, 9, 10, 11],
}

# Total layers in XLM-RoBERTa-base
NUM_LAYERS = 12


def generate_layer_exclude_regex(exclude_layers: List[int]) -> List[str]:
    """
    Generate regex patterns to exclude specific layers from merging.

    Parameters that match these patterns will NOT be merged - they will
    retain their values from the base model (or be copied from best source).

    Args:
        exclude_layers: List of layer indices to exclude (0-11)

    Returns:
        List of regex patterns to pass to exclude_param_names_regex

    Example:
        >>> generate_layer_exclude_regex([0, 1])
        ['roberta\\.encoder\\.layer\\.0\\.', 'roberta\\.encoder\\.layer\\.1\\.']
    """
    patterns = []
    for layer_id in exclude_layers:
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError(f"Layer index {layer_id} out of range [0, {NUM_LAYERS})")
        pattern = XLM_ROBERTA_LAYER_PATTERN.format(layer_id=layer_id)
        patterns.append(pattern)
    return patterns


def generate_layer_group_exclude_regex(exclude_groups: List[str]) -> List[str]:
    """
    Generate regex patterns to exclude layer groups.

    Args:
        exclude_groups: List of group names ("bottom", "middle", "top")

    Returns:
        List of regex patterns

    Example:
        >>> generate_layer_group_exclude_regex(["top"])
        ['roberta\\.encoder\\.layer\\.8\\.', ..., 'roberta\\.encoder\\.layer\\.11\\.']
    """
    exclude_layers = []
    for group in exclude_groups:
        if group not in LAYER_GROUPS:
            raise ValueError(f"Unknown layer group: {group}. Valid: {list(LAYER_GROUPS.keys())}")
        exclude_layers.extend(LAYER_GROUPS[group])
    return generate_layer_exclude_regex(exclude_layers)


def get_merge_layers(exclude_layers: List[int], total_layers: int = NUM_LAYERS) -> List[int]:
    """
    Get the layers that WILL be merged (complement of exclude).

    Args:
        exclude_layers: Layers to exclude from merging
        total_layers: Total number of layers (default: 12)

    Returns:
        List of layer indices that will be merged
    """
    exclude_set = set(exclude_layers)
    return [i for i in range(total_layers) if i not in exclude_set]


def get_ablation_points() -> Dict[str, List[int]]:
    """
    Generate all ablation points for leave-one-layer-out analysis.

    Returns:
        Dict mapping ablation name to layers to EXCLUDE.
        When a layer is excluded, we measure whether performance improves
        (layer causes interference) or drops (layer has positive transfer).

    Structure:
        - Individual layers: 12 points (exclude_layer_0 through exclude_layer_11)
        - Layer groups: 3 points (exclude_group_bottom, middle, top)
        - Baseline: 1 point (baseline_all_layers with no exclusion)
    """
    points = {}

    # Baseline: no exclusion (all layers merged)
    points["baseline_all_layers"] = []

    # Individual layers: exclude each one
    for i in range(NUM_LAYERS):
        points[f"exclude_layer_{i}"] = [i]

    # Layer groups: exclude each group
    for group_name, layers in LAYER_GROUPS.items():
        points[f"exclude_group_{group_name}"] = layers.copy()

    return points


def get_layer_params_from_state_dict(
    state_dict: Dict[str, any],
    layer_id: int,
) -> Dict[str, any]:
    """
    Extract parameters for a specific layer from a state dict.

    Args:
        state_dict: Model state dict
        layer_id: Layer index (0-11)

    Returns:
        Dict of param_name -> param_value for the specified layer
    """
    prefix = f"roberta.encoder.layer.{layer_id}."
    return {
        name: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }


def identify_layer_from_param_name(param_name: str) -> int:
    """
    Extract layer index from a parameter name.

    Args:
        param_name: Full parameter name (e.g., "roberta.encoder.layer.5.attention.self.query.weight")

    Returns:
        Layer index (0-11) or -1 if not an encoder layer parameter
    """
    import re
    match = re.search(r"roberta\.encoder\.layer\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))
    return -1


def get_non_layer_param_prefixes() -> List[str]:
    """
    Get parameter prefixes that are NOT part of encoder layers.

    These include:
    - Embeddings (word, position, token type)
    - Classifier head
    - Pooler (if present)

    Returns:
        List of parameter name prefixes
    """
    return [
        "roberta.embeddings.",
        "classifier.",
        "roberta.pooler.",
    ]


def is_layer_param(param_name: str) -> bool:
    """Check if a parameter belongs to an encoder layer."""
    return identify_layer_from_param_name(param_name) >= 0
