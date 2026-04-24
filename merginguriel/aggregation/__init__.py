"""
Aggregation package for processing and analyzing experiment results.

This package provides modular components for:
- metadata: Data classes and parsing utilities for experiment metadata
- loaders: Functions for loading results and evaluation matrices
- reports: Analysis and reporting functions
"""

from merginguriel.aggregation.metadata import (
    ExperimentMetadata,
    BaselineData,
    is_merge_model_path,
    parse_num_languages_from_model_path,
    count_languages_in_merge_details,
    extract_locale_from_model_path,
    parse_experiment_metadata,
    determine_experiment_variant,
)

from merginguriel.aggregation.loaders import (
    load_results_from_folder,
    parse_merge_details,
    find_evaluation_matrices,
    load_evaluation_matrix,
    get_baseline_for_target,
    get_experiment_folders,
    aggregate_results,
    extract_accuracy,
)

from merginguriel.aggregation.reports import (
    create_comparison_table,
    generate_summary_stats,
    generate_win_rate_analysis,
    save_results,
    create_comprehensive_markdown_report,
)

__all__ = [
    # Metadata
    "ExperimentMetadata",
    "BaselineData",
    "is_merge_model_path",
    "parse_num_languages_from_model_path",
    "count_languages_in_merge_details",
    "extract_locale_from_model_path",
    "parse_experiment_metadata",
    "determine_experiment_variant",
    # Loaders
    "load_results_from_folder",
    "parse_merge_details",
    "find_evaluation_matrices",
    "load_evaluation_matrix",
    "get_baseline_for_target",
    "get_experiment_folders",
    "aggregate_results",
    "extract_accuracy",
    # Reports
    "create_comparison_table",
    "generate_summary_stats",
    "generate_win_rate_analysis",
    "save_results",
    "create_comprehensive_markdown_report",
]
