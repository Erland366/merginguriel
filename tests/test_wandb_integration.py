#!/usr/bin/env python3
"""
Pytest-friendly checks for the wandb integration guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from merginguriel.training_bert import generate_wandb_run_name


@dataclass(frozen=True)
class WandbScenario:
    name: str
    flags: list[str]
    expect_offline: bool = False


SCENARIOS = [
    WandbScenario(
        "basic",
        [
            "--output_dir", "./test_results",
            "--do_train",
            "--do_eval",
            "--num-train-epochs", "1",
        ],
    ),
    WandbScenario(
        "custom_project",
        [
            "--wandb_project", "MergingUriel",
            "--wandb_tags", "roberta,massive,intent-classification,test",
        ],
    ),
    WandbScenario(
        "offline",
        [
            "--wandb_offline",
            "--output_dir", "./test_results_offline",
        ],
        expect_offline=True,
    ),
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_wandb_flag_sets_are_consistent(scenario: WandbScenario):
    joined = " ".join(scenario.flags)
    assert "--output_dir" in joined or "--wandb_project" in joined
    if scenario.expect_offline:
        assert "--wandb_offline" in scenario.flags
    else:
        assert "--wandb_offline" not in scenario.flags


def test_generate_wandb_run_name_matches_offline_scenario():
    class DummyArgs:
        def __init__(self, value):
            self.value = value

    model_args = type("ModelArgs", (), {"model_name_or_path": "roberta-base"})()
    data_args = type("DataArgs", (), {"dataset_name": "massive", "dataset_config_name": "en-US"})()
    training_args = type("TrainingArgs", (), {"learning_rate": 5e-5, "num_train_epochs": 1})()

    assert generate_wandb_run_name(model_args, data_args, training_args) == "roberta-base_massive_en-US_lr5e-5_ep1"
