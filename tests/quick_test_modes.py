#!/usr/bin/env python3
"""
Metadata validation for the CLI mode cheat-sheet.
"""

from __future__ import annotations

import pytest

MODE_DESCRIPTIONS = {
    "uriel": "URIEL similarity weighting + linear merge",
    "manual": "User-specified weights",
    "similarity": "Top-K selection from similarity matrix",
    "average": "Equal weights baseline",
    "fisher": "Fisher-based weighting",
    "iterative": "Iterative training/merging flow",
    "ties": "Sign-aware pruning merge",
    "task_arithmetic": "Task vector arithmetic merge",
    "slerp": "Incremental spherical interpolation",
    "regmean": "RegMean-inspired weighted average",
}


@pytest.mark.parametrize("mode,description", MODE_DESCRIPTIONS.items())
def test_mode_descriptions_are_non_empty(mode, description):
    assert description
    assert mode == mode.strip()


def test_mode_catalog_matches_cli_set():
    expected_modes = {
        "uriel", "manual", "similarity", "average", "fisher", "iterative",
        "ties", "task_arithmetic", "slerp", "regmean",
    }
    assert set(MODE_DESCRIPTIONS.keys()) == expected_modes
