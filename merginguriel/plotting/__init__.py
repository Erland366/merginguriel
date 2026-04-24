"""
Plotting package for MergingUriel results analysis.

This package provides modular components for:
- Data loading and preprocessing
- Plotting utilities and helpers
- Plot generation for various analysis types
"""

from merginguriel.plotting.data_loader import ResultsDataLoader
from merginguriel.plotting.utils import (
    safe_float,
    maybe_float,
    format_method_key_for_filename,
    format_method_key_for_display,
    extract_baselines,
    get_method_num_language_set,
)
from merginguriel.plotting.generators import PlotGenerator

__all__ = [
    "ResultsDataLoader",
    "PlotGenerator",
    "safe_float",
    "maybe_float",
    "format_method_key_for_filename",
    "format_method_key_for_display",
    "extract_baselines",
    "get_method_num_language_set",
]
