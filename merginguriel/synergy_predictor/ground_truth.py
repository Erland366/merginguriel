"""
Ground truth manager for merging effect prediction.

Stores known merging results and provides source locale selection
using the same REAL similarity pipeline as the merging experiments.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class MergingResult:
    target_locale: str
    merged_accuracy: float
    best_source_accuracy: float
    merging_effect: float  # merged - best_source (percentage points)
    is_synergy: bool


# 21 validated targets (ExcTar, REAL similarity, 5 sources, method=similarity)
# Baseline: best zero-shot source accuracy (hardest bar)
KNOWN_RESULTS: dict[str, MergingResult] = {
    # Original 9
    "sw-KE": MergingResult("sw-KE", 0.4832, 0.4169, +6.63, True),
    "cy-GB": MergingResult("cy-GB", 0.4166, 0.4445, -2.79, False),
    "vi-VN": MergingResult("vi-VN", 0.6769, 0.7428, -6.59, False),
    "az-AZ": MergingResult("az-AZ", 0.6627, 0.6503, +1.24, True),
    "tr-TR": MergingResult("tr-TR", 0.7290, 0.7122, +1.68, True),
    "af-ZA": MergingResult("af-ZA", 0.5740, 0.6352, -6.12, False),
    "am-ET": MergingResult("am-ET", 0.4361, 0.4892, -5.31, False),
    "tl-PH": MergingResult("tl-PH", 0.5921, 0.5864, +0.57, True),
    "id-ID": MergingResult("id-ID", 0.7091, 0.8184, -10.93, False),
    # Expansion 12 (Jan 28, 2026)
    "ms-MY": MergingResult("ms-MY", 0.6658, 0.6745, -0.87, False),
    "hi-IN": MergingResult("hi-IN", 0.5928, 0.6886, -9.58, False),
    "km-KH": MergingResult("km-KH", 0.5010, 0.5403, -3.93, False),
    "th-TH": MergingResult("th-TH", 0.6469, 0.6994, -5.25, False),
    "ko-KR": MergingResult("ko-KR", 0.6537, 0.7081, -5.44, False),
    "fa-IR": MergingResult("fa-IR", 0.6974, 0.7555, -5.81, False),
    "lv-LV": MergingResult("lv-LV", 0.6026, 0.6678, -6.52, False),
    "el-GR": MergingResult("el-GR", 0.6564, 0.7031, -4.67, False),
    "my-MM": MergingResult("my-MM", 0.5995, 0.6190, -1.95, False),
    "ka-GE": MergingResult("ka-GE", 0.4919, 0.5501, -5.82, False),
    "bn-BD": MergingResult("bn-BD", 0.5407, 0.6200, -7.93, False),
    "ml-IN": MergingResult("ml-IN", 0.5935, 0.6819, -8.84, False),
}


def get_ground_truth_targets() -> list[str]:
    """Return list of targets with known merging effect."""
    return list(KNOWN_RESULTS.keys())


def get_source_locales(
    target: str,
    num_languages: int = 5,
    similarity_matrix_path: str = None,
) -> list[str]:
    """
    Get source locales as the merging pipeline would select them.

    Uses REAL similarity with Sinkhorn normalization, same as experiments.
    """
    if similarity_matrix_path is None:
        similarity_matrix_path = os.path.join(
            project_root, "language_similarity_matrix_unified.csv"
        )

    from merginguriel.similarity_utils import load_and_process_similarity

    sources = load_and_process_similarity(
        similarity_matrix_path,
        target,
        num_languages=num_languages,
        top_k=20,
        sinkhorn_iterations=20,
        include_target=False,
        verbose=False,
    )
    return [locale for locale, _weight in sources]
