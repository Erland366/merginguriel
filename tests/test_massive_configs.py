#!/usr/bin/env python3
"""
Static checks that our local similarity matrix covers the README locales.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_CSV = PROJECT_ROOT / "language_similarity_matrix_unified.csv"

README_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "hi-IN",
    "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE",
    "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM",
    "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SI", "sq-AL",
    "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "zh-TW",
]


@pytest.fixture(scope="module")
def similarity_locales():
    if not SIMILARITY_CSV.exists():
        pytest.skip("language_similarity_matrix_unified.csv not available in workspace")
    df = pd.read_csv(SIMILARITY_CSV, index_col=0)
    assert set(df.index) == set(df.columns), "Similarity matrix should be square with aligned axes"
    return set(df.index)


def test_similarity_matrix_covers_readme_locales(similarity_locales):
    missing = sorted(set(README_LOCALES) - similarity_locales)
    assert not missing, f"Missing locales from similarity matrix: {missing}"


def test_similarity_matrix_has_expected_cardinality(similarity_locales):
    assert len(similarity_locales) >= len(README_LOCALES)
