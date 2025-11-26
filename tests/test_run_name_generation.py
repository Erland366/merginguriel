#!/usr/bin/env python3
"""
Pytest coverage for the wandb run-name helper documented in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from merginguriel.training_bert import generate_wandb_run_name


@dataclass
class MockModelArgs:
    model_name_or_path: str


@dataclass
class MockDataArgs:
    dataset_name: str
    dataset_config_name: str


@dataclass
class MockTrainingArgs:
    learning_rate: float
    num_train_epochs: int


@pytest.mark.parametrize(
    ("model_path", "dataset_name", "dataset_config", "lr", "epochs", "expected"),
    [
        (
            "FacebookAI/roberta-base",
            "AmazonScience/massive",
            "en-US",
            5e-5,
            3,
            "roberta-base_massive_en-US_lr5e-5_ep3",
        ),
        (
            "bert-base-uncased",
            "AmazonScience/massive",
            "en-US",
            3e-5,
            5,
            "bert-base-uncased_massive_en-US_lr3e-5_ep5",
        ),
        (
            "FacebookAI/roberta-base",
            "AmazonScience/massive",
            "fr-FR",
            1e-4,
            2,
            "roberta-base_massive_fr-FR_lr1e-4_ep2",
        ),
        (
            "bert-base-uncased",
            "AmazonScience/massive",
            "en-US",
            1e-2,
            1,
            "bert-base-uncased_massive_en-US_lr0.0100_ep1",
        ),
        (
            "roberta-base",
            "massive",
            "en-US",
            0.0,
            1,
            "roberta-base_massive_en-US_lr0_ep1",
        ),
    ],
)
def test_generate_wandb_run_name(model_path, dataset_name, dataset_config, lr, epochs, expected):
    model_args = MockModelArgs(model_path)
    data_args = MockDataArgs(dataset_name, dataset_config)
    training_args = MockTrainingArgs(lr, epochs)

    assert generate_wandb_run_name(model_args, data_args, training_args) == expected
