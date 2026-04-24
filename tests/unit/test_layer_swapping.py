#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

from merginguriel.layer_swapping import (
    apply_layer_swapping_state_dicts,
    get_swap_indices,
)
from merginguriel.layer_swap_ablation import (
    AblationResult,
    aggregate_results,
    generate_ablation_configs,
    generate_method_comparison_configs,
)


def _make_state(prefix: str) -> dict[str, torch.Tensor]:
    return {
        "roberta.encoder.layer.0.attention.self.query.weight": torch.tensor([1.0]) * (1 if prefix == "math" else 10),
        "roberta.encoder.layer.1.attention.self.query.weight": torch.tensor([2.0]) * (1 if prefix == "math" else 10),
        "classifier.weight": torch.tensor([3.0]) * (1 if prefix == "math" else 10),
    }


def test_get_swap_indices_top_bottom():
    indices = get_swap_indices(num_layers=12, top_k=2, bottom_k=2, swap_strategy="top_bottom")
    assert indices == [0, 1, 10, 11]


def test_get_swap_indices_custom():
    indices = get_swap_indices(num_layers=4, top_k=0, bottom_k=0, custom_layers=[2, 0, 2], swap_strategy="custom")
    assert indices == [0, 2]


def test_apply_layer_swapping_state_dicts_swaps_layers():
    model_b = _make_state("language")
    model_a = _make_state("math")
    merged, metadata = apply_layer_swapping_state_dicts(
        model_a_state=model_a,
        model_b_state=model_b,
        swap_layer_indices=[1],
        non_layer_source="model_a",
    )
    assert merged["roberta.encoder.layer.0.attention.self.query.weight"].item() == 1.0
    assert merged["roberta.encoder.layer.1.attention.self.query.weight"].item() == 20.0
    assert merged["classifier.weight"].item() == 3.0
    assert metadata["swapped_param_count"] == 1


def test_apply_layer_swapping_state_dicts_non_layer_from_model_b():
    model_b = _make_state("language")
    model_a = _make_state("math")
    merged, metadata = apply_layer_swapping_state_dicts(
        model_a_state=model_a,
        model_b_state=model_b,
        swap_layer_indices=[],
        non_layer_source="model_b",
    )
    assert merged["classifier.weight"].item() == 30.0
    assert metadata["non_layer_params_from_model_b"] == 1


def test_apply_layer_swapping_state_dicts_legacy_names():
    """Backward-compatible: 'math'/'language' still work."""
    model_b = _make_state("language")
    model_a = _make_state("math")
    merged, metadata = apply_layer_swapping_state_dicts(
        model_a_state=model_a,
        model_b_state=model_b,
        swap_layer_indices=[1],
        non_layer_source="math",  # legacy name for model_a
    )
    assert merged["roberta.encoder.layer.1.attention.self.query.weight"].item() == 20.0
    assert metadata["swapped_param_count"] == 1


def test_apply_layer_swapping_state_dicts_mismatch_keys():
    model_b = _make_state("language")
    model_a = _make_state("math")
    del model_b["classifier.weight"]
    with pytest.raises(ValueError):
        apply_layer_swapping_state_dicts(
            model_a_state=model_a,
            model_b_state=model_b,
            swap_layer_indices=[0],
            non_layer_source="model_a",
        )


# ---------------------------------------------------------------------------
# Ablation config generator tests
# ---------------------------------------------------------------------------


def test_generate_ablation_configs_top_bottom_sweep():
    configs = generate_ablation_configs("top_bottom_sweep", num_layers=12)
    # 6 symmetric + 6 top-only + 6 bottom-only = 18
    assert len(configs) == 18
    names = [name for name, _ in configs]
    assert "top1_bottom1" in names
    assert "top6_bottom0" in names
    assert "top0_bottom6" in names


def test_generate_ablation_configs_single_layer():
    configs = generate_ablation_configs("single_layer", num_layers=12)
    assert len(configs) == 12
    assert configs[0][0] == "single_layer_0"
    assert configs[11][0] == "single_layer_11"


def test_generate_ablation_configs_layer_groups():
    configs = generate_ablation_configs("layer_groups", num_layers=12)
    # bottom, middle, top + top_bottom = 4
    assert len(configs) == 4
    names = [name for name, _ in configs]
    assert "group_bottom" in names
    assert "group_top_bottom" in names


def test_generate_ablation_configs_non_layer_source():
    configs = generate_ablation_configs("non_layer_source", num_layers=12)
    assert len(configs) == 2
    sources = [cfg.non_layer_source for _, cfg in configs]
    assert "model_a" in sources
    assert "model_b" in sources


def test_generate_ablation_configs_full():
    configs = generate_ablation_configs("full", num_layers=12)
    # 18 (top_bottom) + 12 (single) + 4 (groups) + 2 (nls) = 36
    assert len(configs) == 36


def test_ablation_result_delta():
    r = AblationResult(
        experiment_name="test",
        model_a_locale="en-US",
        model_b_locale="fr-FR",
        target_locale="sw-KE",
        swap_strategy="top_bottom",
        swap_indices=[0, 1, 10, 11],
        non_layer_source="model_a",
        accuracy=0.85,
        baseline_a_accuracy=0.80,
        baseline_b_accuracy=0.82,
    )
    assert r.delta_vs_best_baseline == pytest.approx(0.03)


def test_aggregate_results():
    results = [
        AblationResult(
            experiment_name="exp1", model_a_locale="a", model_b_locale="b",
            target_locale="t", swap_strategy="custom", swap_indices=[0],
            non_layer_source="model_a", accuracy=0.9,
        ),
    ]
    df = aggregate_results(results)
    assert len(df) == 1
    assert df.iloc[0]["accuracy"] == 0.9


def test_generate_method_comparison_configs():
    configs = generate_method_comparison_configs()
    assert len(configs) == 4
    names = [name for name, _ in configs]
    assert "baseline_average" in names
    assert "baseline_ties" in names
    assert "baseline_task_arithmetic" in names
    assert "baseline_dare_task_arithmetic" in names
    # DARE config should have dare_enabled=True
    dare_cfg = next(cfg for name, cfg in configs if name == "baseline_dare_task_arithmetic")
    assert dare_cfg.dare_enabled is True
    assert dare_cfg.dare_drop_rate == 0.9

