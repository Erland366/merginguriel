#!/usr/bin/env python3
"""
Pytest coverage for `merginguriel.similarity_utils`.

These tests exercise the lightweight processing helpers that the README highlights
for URIEL-based language selection, without touching heavyweight HF assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from merginguriel.similarity_utils import (
    load_and_process_similarity,
    load_similarity_matrix,
    process_similarity_matrix,
    get_similarity_weights,
)


@pytest.fixture
def toy_similarity_csv(tmp_path: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Create a small similarity matrix that roughly mirrors the README examples.

    Layout:
        en-US is closest to fr-FR, then de-DE
        fr-FR is closest to en-US, then de-DE
        de-DE prefers en-US over fr-FR
    """
    locales = ["en-US", "fr-FR", "de-DE"]
    data = [
        [1.0, 0.8, 0.4],
        [0.8, 1.0, 0.6],
        [0.7, 0.5, 1.0],
    ]
    df = pd.DataFrame(data, index=locales, columns=locales)
    csv_path = tmp_path / "toy_similarity.csv"
    df.to_csv(csv_path)
    return csv_path, df


def test_load_similarity_matrix_reads_expected_shape(toy_similarity_csv):
    csv_path, df = toy_similarity_csv

    loaded = load_similarity_matrix(str(csv_path), verbose=False)

    assert loaded.shape == df.shape
    assert list(loaded.index) == ["en-US", "fr-FR", "de-DE"]


def test_process_similarity_matrix_normalizes_rows(toy_similarity_csv):
    _, df = toy_similarity_csv

    processed = process_similarity_matrix(
        df,
        target_locale="en-US",
        top_k=1,
        sinkhorn_iterations=5,
        verbose=False,
    )

    # Sinkhorn should force each row to sum to ~1 even after sparsification
    for locale in processed.index:
        assert processed.loc[locale].sum() == pytest.approx(1.0, rel=1e-3)
        # With top_k=1 only the strongest neighbor remains non-zero
        assert (processed.loc[locale] > 0).sum() <= 1


def test_get_similarity_weights_respects_include_target_flag(toy_similarity_csv):
    csv_path, _ = toy_similarity_csv

    include_target = get_similarity_weights(
        load_similarity_matrix(str(csv_path), verbose=False),
        target_locale="en-US",
        num_languages=2,
        include_target=True,
        top_k=2,
        sinkhorn_iterations=5,
        verbose=False,
    )
    exclude_target = get_similarity_weights(
        load_similarity_matrix(str(csv_path), verbose=False),
        target_locale="en-US",
        num_languages=2,
        include_target=False,
        top_k=2,
        sinkhorn_iterations=5,
        verbose=False,
    )

    assert include_target[0][0] == "en-US"
    assert all(locale != "en-US" for locale, _ in exclude_target)
    assert len(include_target) == 2
    assert len(exclude_target) == 2


def test_load_and_process_similarity_returns_sorted_weights(toy_similarity_csv):
    csv_path, _ = toy_similarity_csv

    weights = load_and_process_similarity(
        str(csv_path),
        target_locale="en-US",
        num_languages=2,
        top_k=2,
        sinkhorn_iterations=5,
        include_target=False,
        verbose=False,
    )

    assert len(weights) == 2
    assert weights[0][1] >= weights[1][1]  # Sorted descending by weight
    assert all(locale in {"fr-FR", "de-DE"} for locale, _ in weights)
