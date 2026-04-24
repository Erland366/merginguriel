#!/usr/bin/env python3
"""
Structured checks for the advanced method cheat-sheet that used to live in the demo script.
"""

from __future__ import annotations

import pytest

from merginguriel.run_merging_pipeline_refactored import MergingStrategyFactory

ADVANCED_METHOD_SPECS = [
    {
        "name": "TIES",
        "mode": "ties",
        "description": "Resolves sign disagreements and prunes low-magnitude weights",
        "use_case": "Models with conflicting parameter directions",
        "command": "python merginguriel/run_merging_pipeline_refactored.py --mode ties --target-lang sq-AL --num-languages 5",
    },
    {
        "name": "TaskArithmetic",
        "mode": "task_arithmetic",
        "description": "Adds/subtracts task vectors representing fine-tuning changes",
        "use_case": "Task vector experimentation",
        "command": "python merginguriel/run_merging_pipeline_refactored.py --mode task_arithmetic --target-lang sq-AL --num-languages 5",
    },
    {
        "name": "SLERP",
        "mode": "slerp",
        "description": "Spherical Linear Interpolation with incremental merging",
        "use_case": "Smooth interpolation across many locales",
        "command": "python merginguriel/run_merging_pipeline_refactored.py --mode slerp --target-lang sq-AL --num-languages 5",
    },
    {
        "name": "RegMean",
        "mode": "regmean",
        "description": "Regression-based coefficient estimation (simplified)",
        "use_case": "Data-driven coefficient tuning",
        "command": "python merginguriel/run_merging_pipeline_refactored.py --mode regmean --target-lang sq-AL --num-languages 5",
    },
]


@pytest.mark.parametrize("spec", ADVANCED_METHOD_SPECS)
def test_demo_commands_reference_required_flags(spec):
    cmd = spec["command"]
    assert "--mode" in cmd and f"--mode {spec['mode']}" in cmd
    assert "--target-lang" in cmd
    assert "--num-languages" in cmd
    assert spec["description"]
    assert spec["use_case"]


@pytest.mark.parametrize("spec", ADVANCED_METHOD_SPECS)
def test_demo_modes_have_registered_strategies(spec):
    strategy = MergingStrategyFactory.create(spec["mode"])
    assert strategy.__class__.__name__.lower().startswith(spec["name"].lower().split()[0])
