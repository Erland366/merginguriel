#!/usr/bin/env python3
"""
Pytest coverage for the early-stopping recommendations described in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class EarlyStoppingScenario:
    name: str
    patience: int | None
    threshold: float | None
    description: str

    @property
    def uses_early_stopping(self) -> bool:
        return self.patience is not None


SCENARIOS = [
    EarlyStoppingScenario("default", patience=3, threshold=0.0, description="project default"),
    EarlyStoppingScenario("patient", patience=5, threshold=0.001, description="more patient"),
    EarlyStoppingScenario("aggressive", patience=1, threshold=0.0, description="fast fail"),
    EarlyStoppingScenario("bert", patience=4, threshold=0.0, description="bert baseline"),
    EarlyStoppingScenario("baseline_no_es", patience=None, threshold=None, description="no early stopping baseline"),
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_patience_and_threshold_ranges(scenario: EarlyStoppingScenario):
    if scenario.uses_early_stopping:
        assert scenario.patience >= 1
        assert scenario.threshold is not None and scenario.threshold >= 0.0
    else:
        assert scenario.patience is None
        assert scenario.threshold is None


def test_at_least_one_baseline_without_early_stopping():
    assert any(not scenario.uses_early_stopping for scenario in SCENARIOS)


def test_all_scenarios_have_descriptions():
    for scenario in SCENARIOS:
        assert scenario.description
