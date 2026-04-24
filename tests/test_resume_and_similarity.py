import json
import os
from pathlib import Path

import pandas as pd

from merginguriel import similarity_utils
from merginguriel import run_large_scale_experiment as rls_experiment
from merginguriel import run_large_scale_ensemble_experiments as rls_ensemble
from merginguriel import run_large_scale_iterative_training as rls_iterative
from merginguriel.naming_config import naming_manager


def test_similarity_utils_deduplicates_and_aligns(tmp_path):
    """Duplicate locales in the similarity matrix should be deduped and kept square."""
    data = {
        "af-ZA": [1.0, 0.5, 0.1, 0.2],
        "bn-BD": [0.5, 1.0, 0.3, 0.4],
        "zh-TW": [0.1, 0.3, 1.0, 0.9],
        "zh-TW": [0.2, 0.4, 0.9, 1.0],  # duplicate column/header
    }
    df = pd.DataFrame(data, index=["af-ZA", "bn-BD", "zh-TW", "zh-TW"])  # duplicate index
    matrix_path = tmp_path / "dup_matrix.csv"
    df.to_csv(matrix_path)

    loaded = similarity_utils.load_similarity_matrix(matrix_path, verbose=False)

    # After deduplication, only unique locales remain and index/columns align
    assert loaded.index.tolist() == ["af-ZA", "bn-BD", "zh-TW"]
    assert loaded.columns.tolist() == ["af-ZA", "bn-BD", "zh-TW"]
    # Values for the kept first occurrence should match
    assert loaded.loc["bn-BD", "af-ZA"] == 0.5


def make_results_dir(base_dir: Path, experiment_type: str, method: str, similarity_type, locale: str,
                     model_family: str, num_languages: int | None, include_target: bool):
    """Create a minimal results directory with results.json to be discovered by resume helpers."""
    folder = naming_manager.get_results_dir_name(
        experiment_type=experiment_type,
        method=method,
        similarity_type=similarity_type,
        locale=locale,
        model_family=model_family,
        num_languages=num_languages,
        include_target=include_target,
    )
    dest = base_dir / "results" / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps({"performance": {"accuracy": 0.9}}))
    return dest


def test_resume_finds_existing_merging_results(tmp_path, monkeypatch):
    # Point runner to temp repo root
    monkeypatch.setattr(rls_experiment, "REPO_ROOT", tmp_path)
    model_family = "xlm-roberta-base_massive_k_zh-TW"
    existing = make_results_dir(
        tmp_path, "merging", "similarity", "URIEL", "zh-TW", model_family, num_languages=5, include_target=True
    )

    found = rls_experiment.results_already_exist(
        experiment_type="merging",
        method="similarity",
        similarity_type="URIEL",
        locale="zh-TW",
        model_family=model_family,
        results_dir="results",
        num_languages=5,
        include_target=True,
    )

    assert found == str(existing)


def test_resume_finds_existing_ensemble_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rls_ensemble, "REPO_ROOT", tmp_path)
    model_family = "xlm-roberta-base_massive_k_zh-TW"
    existing = make_results_dir(
        tmp_path, "ensemble", "majority", "URIEL", "zh-TW", model_family, num_languages=5, include_target=False
    )

    found = rls_ensemble.results_already_exist(
        experiment_type="ensemble",
        method="majority",
        similarity_type="URIEL",
        locale="zh-TW",
        model_family=model_family,
        results_dir="results",
        num_languages=5,
        include_target=False,
    )

    assert found == str(existing)


def test_resume_finds_existing_iterative_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rls_iterative, "REPO_ROOT", tmp_path)
    model_family = "xlm-roberta-base_massive_k_zh-TW"
    existing = make_results_dir(
        tmp_path, "iterative", "similarity", "URIEL", "zh-TW", model_family, num_languages=4, include_target=False
    )

    found = rls_iterative.results_already_exist(
        experiment_type="iterative",
        method="similarity",
        similarity_type="URIEL",
        locale="zh-TW",
        model_family=model_family,
        results_dir="results",
        num_languages=4,
        include_target=False,
    )

    assert found == str(existing)
