"""
Selective Layer Merging module for cross-lingual transfer.

This module implements selective layer merging based on the hypothesis that
different transformer layers have varying cross-lingual transfer properties:
some contribute positively (should be merged), others cause interference
(should be excluded from merging).

Key components:
- layer_masking: Generate regex patterns for layer exclusion
- selective_merge: Orchestrate selective layer merging
- leave_one_out_cv: Cross-validation framework for layer ablation
"""

from merginguriel.selective_layer.layer_masking import (
    generate_layer_exclude_regex,
    generate_layer_group_exclude_regex,
    get_ablation_points,
    get_merge_layers,
    get_layer_params_from_state_dict,
    identify_layer_from_param_name,
    LAYER_GROUPS,
    NUM_LAYERS,
    XLM_ROBERTA_LAYER_PATTERN,
)

from merginguriel.selective_layer.selective_merge import (
    SelectiveLayerMerger,
    SelectiveMergeResult,
    find_best_source,
    copy_layers_from_source,
    run_selective_merge_experiment,
    evaluate_selective_merge,
)

from merginguriel.selective_layer.leave_one_out_cv import (
    LayerAblationResult,
    LayerAblationDB,
    LayerAblationConfig,
    LeaveOneSourceOutCV,
    analyze_ablation_results,
    print_transfer_summary,
    interpret_transfer,
)

__all__ = [
    # Layer masking
    "generate_layer_exclude_regex",
    "generate_layer_group_exclude_regex",
    "get_ablation_points",
    "get_merge_layers",
    "get_layer_params_from_state_dict",
    "identify_layer_from_param_name",
    "LAYER_GROUPS",
    "NUM_LAYERS",
    "XLM_ROBERTA_LAYER_PATTERN",
    # Selective merging
    "SelectiveLayerMerger",
    "SelectiveMergeResult",
    "find_best_source",
    "copy_layers_from_source",
    "run_selective_merge_experiment",
    "evaluate_selective_merge",
    # Leave-one-out CV
    "LayerAblationResult",
    "LayerAblationDB",
    "LayerAblationConfig",
    "LeaveOneSourceOutCV",
    "analyze_ablation_results",
    "print_transfer_summary",
    "interpret_transfer",
]
