#!/usr/bin/env python3
"""
Unit tests for the strategy layer used by the refactored merging pipeline.

These tests focus on the configuration plumbing highlighted in README.md:
  * Fisher/URIEL integrations
  * Incremental SLERP orchestration
  * RegMean's simplified linear fallback
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from merginguriel.run_merging_pipeline_refactored import (
    MergeConfig,
    ModelInfo,
    MergingStrategyFactory,
    LinearStrategy,
    FisherDatasetStrategy,
    SlerpStrategy,
    RegMeanStrategy,
)


@pytest.fixture
def base_model_info() -> ModelInfo:
    return ModelInfo(
        model_name="haryos_model/xlm-roberta-base_massive_k_en-US",
        subfolder="",
        language="English",
        locale="en-US",
        weight=1.0,
    )


@pytest.fixture
def source_models() -> OrderedDict[str, ModelInfo]:
    models = OrderedDict()
    models["/models/de"] = ModelInfo(
        model_name="/models/de",
        subfolder="",
        language="German",
        locale="de-DE",
        weight=0.75,
    )
    models["/models/fr"] = ModelInfo(
        model_name="/models/fr",
        subfolder="",
        language="French",
        locale="fr-FR",
        weight=0.25,
    )
    return models


def test_strategy_factory_defaults_to_linear():
    strategy = MergingStrategyFactory.create("non-existent-mode")
    assert isinstance(strategy, LinearStrategy)


def test_fisher_dataset_requires_dataset_name(source_models, base_model_info):
    config = MergeConfig(mode="fisher", target_lang="en-US")
    strategy = FisherDatasetStrategy()

    with pytest.raises(ValueError):
        strategy.get_method_params(config, source_models, base_model_info)


def test_fisher_dataset_scales_weights_with_uriel(source_models, base_model_info):
    config = MergeConfig(
        mode="fisher",
        target_lang="en-US",
        dataset_name="AmazonScience/massive",
        dataset_split="train",
        text_column="utt",
        label_column="intent",
        preweight="uriel",
        num_fisher_examples=128,
        fisher_data_mode="both",
        batch_size=4,
        max_seq_length=64,
    )
    strategy = FisherDatasetStrategy()

    params = strategy.get_method_params(config, source_models, base_model_info)

    coeffs = params["fisher_scaling_coefficients"]
    assert coeffs == pytest.approx([0.75 / 1.0, 0.25 / 1.0])
    assert params["source_locales"] == ["de-DE", "fr-FR"]
    assert params["dataset_config"]["dataset_name"] == "AmazonScience/massive"


def test_slerp_strategy_builds_incremental_steps(source_models, base_model_info):
    config = MergeConfig(mode="slerp", target_lang="sq-AL")
    strategy = SlerpStrategy()

    params = strategy.get_method_params(config, source_models, base_model_info)

    assert params["incremental_slerp"] is True
    assert params["total_models"] == 3  # base + two sources
    assert len(params["merge_steps"]) == 2
    first_step = params["merge_steps"][0]
    assert first_step["base_model"] == base_model_info.model_name
    assert first_step["merge_model"] == "/models/de"
    assert 0.0 <= first_step["slerp_t"] <= 1.0


def test_regmean_strategy_returns_linear_params(source_models, base_model_info, capsys):
    config = MergeConfig(mode="regmean", target_lang="af-ZA")
    strategy = RegMeanStrategy()

    params = strategy.get_method_params(config, source_models, base_model_info)

    assert params["weights"] == [0.75, 0.25]
    captured = capsys.readouterr()
    assert "RegMean Strategy" in captured.out
