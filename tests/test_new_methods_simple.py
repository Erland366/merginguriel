#!/usr/bin/env python3
"""
Pytest verification that all advanced strategies are wired into the factory.
"""

from __future__ import annotations

import pytest

from merginguriel.run_merging_pipeline_refactored import (
    MergingStrategyFactory,
    TiesStrategy,
    TaskArithmeticStrategy,
    SlerpStrategy,
    RegMeanStrategy,
)


@pytest.mark.parametrize(
    ("mode", "expected_cls"),
    [
        ("ties", TiesStrategy),
        ("task_arithmetic", TaskArithmeticStrategy),
        ("slerp", SlerpStrategy),
        ("regmean", RegMeanStrategy),
    ],
)
def test_strategy_factory_returns_expected_type(mode, expected_cls):
    strategy = MergingStrategyFactory.create(mode)
    assert isinstance(strategy, expected_cls)
