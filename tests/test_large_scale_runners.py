import json
from pathlib import Path

from merginguriel import run_large_scale_ensemble_experiments as rls_ensemble
from merginguriel import run_large_scale_iterative_training as rls_iterative
from merginguriel.naming_config import naming_manager


def _touch_model_dir(repo_root: Path, locale: str):
    """Create a dummy model directory so detection/validation passes."""
    model_dir = repo_root / "haryos_model" / f"xlm-roberta-base_massive_k_{locale}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _make_results_dir(repo_root: Path, experiment_type: str, method: str, similarity_type, locale: str,
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
    dest = repo_root / "results" / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps({"performance": {"accuracy": 0.9}}))
    return dest


def test_ensemble_runner_respects_resume(monkeypatch, tmp_path):
    """When results already exist, the ensemble runner should skip execution."""
    # Point to isolated repo root
    monkeypatch.setattr(rls_ensemble, "REPO_ROOT", tmp_path)

    # Create dummy model dir for locale
    _touch_model_dir(tmp_path, "zh-TW")
    model_family = "xlm-roberta-base_massive_k_zh-TW"

    # Seed existing results for ExcTar
    existing = _make_results_dir(
        tmp_path, "ensemble", "majority", "URIEL", "zh-TW", model_family, num_languages=5, include_target=False
    )

    calls = []

    def fake_run(method, locale, extra_args):
        calls.append((method, locale, extra_args))
        return False  # should not be called when resume hits

    monkeypatch.setattr(rls_ensemble, "run_ensemble_inference", fake_run)

    results = rls_ensemble.run_experiment_for_locale(
        locale="zh-TW",
        voting_methods=["majority"],
        ensemble_extra_args=[],
        models_root="haryos_model",
        similarity_type="URIEL",
        num_languages=5,
        include_target_modes=[False],
        results_dir="results",
        resume=True,
    )

    assert results["ExcTar"]["majority"] is True
    assert calls == []
    assert Path(existing, "results.json").exists()


def test_ensemble_runner_executes_and_saves(monkeypatch, tmp_path):
    """With no prior results, the ensemble runner should execute and call save."""
    monkeypatch.setattr(rls_ensemble, "REPO_ROOT", tmp_path)
    _touch_model_dir(tmp_path, "zh-TW")
    model_family = "xlm-roberta-base_massive_k_zh-TW"

    calls = {"run": 0, "save": 0}

    def fake_run(method, locale, extra_args):
        calls["run"] += 1
        return True

    def fake_load(locale, method, output_dir="urie_ensemble_results"):
        return {
            "experiment_info": {"num_models": 2, "num_examples": 1},
            "metadata": {"model_names": ["a", "b"]},
            "models": {"a": 0.6, "b": 0.4},
            "performance": {"accuracy": 0.9},
            "examples": [],
        }

    def fake_save(locale, voting_method, data, results_dir, model_family, similarity_type, num_models, include_target):
        calls["save"] += 1
        return True

    monkeypatch.setattr(rls_ensemble, "run_ensemble_inference", fake_run)
    monkeypatch.setattr(rls_ensemble, "load_ensemble_results", fake_load)
    monkeypatch.setattr(rls_ensemble, "save_ensemble_results", fake_save)

    results = rls_ensemble.run_experiment_for_locale(
        locale="zh-TW",
        voting_methods=["majority"],
        ensemble_extra_args=[],
        models_root="haryos_model",
        similarity_type="URIEL",
        num_languages=5,
        include_target_modes=[False],
        results_dir="results",
        resume=False,
    )

    assert results["ExcTar"]["majority"] is True
    assert calls["run"] == 1
    assert calls["save"] == 1


def test_iterative_runner_executes_and_saves(monkeypatch, tmp_path):
    """Iterative runner should call training path and persist results."""
    monkeypatch.setattr(rls_iterative, "REPO_ROOT", tmp_path)
    _touch_model_dir(tmp_path, "zh-TW")
    model_family = "xlm-roberta-base_massive_k_zh-TW"

    calls = {"select": 0, "train": 0, "extract": 0, "save": 0}

    def fake_select(target_lang, max_models, similarity_type, models_root, top_k, sinkhorn_iters, include_target):
        calls["select"] += 1
        return ["en-US"]

    def fake_train(target_lang, source_locales, mode, extra_args, output_base_dir, include_target):
        calls["train"] += 1
        return True

    def fake_extract(target_lang, mode, locale_output_dir):
        calls["extract"] += 1
        return {
            "evaluation_info": {"timestamp": "now", "model_name": "iter", "locale": target_lang},
            "model_info": {},
            "dataset_info": {},
            "performance": {"accuracy": 0.75},
            "metadata": {},
        }

    def fake_save(target_lang, mode, results, results_dir, model_family, similarity_type, num_languages, include_target):
        calls["save"] += 1
        return True

    monkeypatch.setattr(rls_iterative, "auto_select_iterative_sources", fake_select)
    monkeypatch.setattr(rls_iterative, "run_iterative_training", fake_train)
    monkeypatch.setattr(rls_iterative, "extract_iterative_results", fake_extract)
    monkeypatch.setattr(rls_iterative, "save_iterative_results", fake_save)

    results = rls_iterative.run_experiment_for_locale(
        locale="zh-TW",
        mode="similarity",
        training_extra_args=[],
        output_base_dir="iterative_output",
        max_models=3,
        include_target_modes=[False],
        similarity_type="URIEL",
        models_root="haryos_model",
        top_k=5,
        sinkhorn_iters=5,
        results_dir="results",
        resume=False,
    )

    assert results["variants"]["ExcTar"]["success"] is True
    assert results["variants"]["ExcTar"]["accuracy"] == 0.75
    assert calls == {"select": 1, "train": 1, "extract": 1, "save": 1}
