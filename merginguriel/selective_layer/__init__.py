"""
Selective Layer Merging module for cross-lingual transfer.

This module implements selective layer merging based on the hypothesis that
different transformer layers have varying cross-lingual transfer properties:
some contribute positively (should be merged), others cause interference
(should be excluded from merging).

Key components:
- layer_masking: Generate regex patterns for layer exclusion/inclusion
- selective_merge: Orchestrate selective layer merging (exclude-based and include-only)
- leave_one_out_cv: Cross-validation framework for exclude-based layer ablation
- include_only_cv: Cross-validation framework for include-only layer ablation
"""

from merginguriel.selective_layer.layer_masking import (
    generate_layer_exclude_regex,
    generate_layer_group_exclude_regex,
    generate_layer_include_only_regex,
    get_ablation_points,
    get_include_only_ablation_points,
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
    IncludeOnlyMerger,
    BestSourceBaseMerger,
    create_pretrained_base_with_classifier,
    run_include_only_merge_experiment,
    run_best_source_base_merge_experiment,
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

from merginguriel.selective_layer.include_only_cv import (
    IncludeOnlyCVConfig,
    LeaveOneSourceOutIncludeOnlyCV,
    print_include_only_summary,
)

from merginguriel.selective_layer.best_source_base_cv import (
    BestSourceBaseCVConfig,
    LeaveOneSourceOutBestSourceBaseCV,
    print_best_source_base_summary,
)

from merginguriel.selective_layer.best_source_base_direct import (
    DirectBestSourceBaseConfig,
    DirectBestSourceBaseAblation,
    analyze_direct_results,
    print_direct_summary,
)

__all__ = [
    # Layer masking
    "generate_layer_exclude_regex",
    "generate_layer_group_exclude_regex",
    "generate_layer_include_only_regex",
    "get_ablation_points",
    "get_include_only_ablation_points",
    "get_merge_layers",
    "get_layer_params_from_state_dict",
    "identify_layer_from_param_name",
    "LAYER_GROUPS",
    "NUM_LAYERS",
    "XLM_ROBERTA_LAYER_PATTERN",
    # Selective merging (exclude-based)
    "SelectiveLayerMerger",
    "SelectiveMergeResult",
    "find_best_source",
    "copy_layers_from_source",
    "run_selective_merge_experiment",
    "evaluate_selective_merge",
    # Include-only merging
    "IncludeOnlyMerger",
    "create_pretrained_base_with_classifier",
    "run_include_only_merge_experiment",
    # Best-source-base merging
    "BestSourceBaseMerger",
    "run_best_source_base_merge_experiment",
    # Leave-one-out CV (exclude-based)
    "LayerAblationResult",
    "LayerAblationDB",
    "LayerAblationConfig",
    "LeaveOneSourceOutCV",
    "analyze_ablation_results",
    "print_transfer_summary",
    "interpret_transfer",
    # Include-only CV
    "IncludeOnlyCVConfig",
    "LeaveOneSourceOutIncludeOnlyCV",
    "print_include_only_summary",
    # Best-source-base CV
    "BestSourceBaseCVConfig",
    "LeaveOneSourceOutBestSourceBaseCV",
    "print_best_source_base_summary",
    # Best-source-base direct
    "DirectBestSourceBaseConfig",
    "DirectBestSourceBaseAblation",
    "analyze_direct_results",
    "print_direct_summary",
]
