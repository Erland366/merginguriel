#!/usr/bin/env python3
"""
Sanity checks covering every CLI mode supported by the merging pipeline.
"""

from __future__ import annotations

import pytest

from merginguriel.run_merging_pipeline_refactored import WeightCalculatorFactory, MergingStrategyFactory

CLI_MODES = [
    "uriel",
    "manual",
    "similarity",
    "average",
    "fisher",
    "iterative",
    "ties",
    "task_arithmetic",
    "slerp",
    "regmean",
]


@pytest.mark.parametrize("mode", CLI_MODES)
def test_weight_calculator_factory_accepts_cli_modes(mode):
    kwargs = {}
    if mode == "manual":
        kwargs["weights"] = {"en-US": 1.0}
    elif mode == "iterative":
        kwargs["active_model_states"] = {}
        kwargs["target_locales"] = ["en-US"]

    calculator = WeightCalculatorFactory.create_calculator(mode, **kwargs)
    assert calculator is not None


@pytest.mark.parametrize("mode", CLI_MODES)
def test_strategy_factory_handles_cli_modes(mode):
    strategy = MergingStrategyFactory.create(mode)
    assert strategy is not None


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        WeightCalculatorFactory.create_calculator("invalid-mode")
