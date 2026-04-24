#!/usr/bin/env python3
"""
Pytest coverage for the batch-size guidance shared in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class BatchScenario:
    name: str
    train_batch: int
    eval_batch: int
    gradient_accumulation: int = 1

    @property
    def effective_batch(self) -> int:
        return self.train_batch * self.gradient_accumulation


BATCH_SCENARIOS = [
    BatchScenario("default", train_batch=8, eval_batch=8),
    BatchScenario("small", train_batch=4, eval_batch=8),
    BatchScenario("medium", train_batch=16, eval_batch=32),
    BatchScenario("large", train_batch=32, eval_batch=64),
    BatchScenario("grad_accumulation", train_batch=8, eval_batch=16, gradient_accumulation=4),
    BatchScenario("mismatched_eval", train_batch=12, eval_batch=48),
]


@pytest.mark.parametrize("scenario", BATCH_SCENARIOS)
def test_batch_size_recommendations_are_positive(scenario: BatchScenario):
    assert scenario.train_batch > 0
    assert scenario.eval_batch > 0
    assert scenario.gradient_accumulation >= 1
    assert scenario.effective_batch >= scenario.train_batch


@pytest.mark.parametrize("scenario", BATCH_SCENARIOS)
def test_effective_batch_matches_gradient_accumulation(scenario: BatchScenario):
    expected_effective = scenario.train_batch * scenario.gradient_accumulation
    assert scenario.effective_batch == expected_effective


def test_gradient_accumulation_profiles_include_memory_friendly_option():
    grad_profiles = [s for s in BATCH_SCENARIOS if s.gradient_accumulation > 1]
    assert grad_profiles, "Expected at least one gradient-accumulation recommendation"
    for profile in grad_profiles:
        assert profile.effective_batch >= 2 * profile.train_batch
