#!/usr/bin/env python3
"""
Evaluate all existing merged models by reusing evaluate_specific_model.

This generates distinct results folders for each (method, locale, num_languages)
combination so downstream aggregation can distinguish between variants.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from merginguriel.run_large_scale_experiment import detect_num_languages_from_model_path
from merginguriel.evaluate_specific_model import (
    create_results_folder,
    evaluate_specific_model,
    save_evaluation_results,
)


def ensure_cache_dirs(hf_home: Path) -> None:
    """Ensure HF caches live inside the workspace with the expected structure."""
    for sub in ("datasets", "transformers", "modules/datasets_modules/datasets"):
        target = hf_home / sub
        target.mkdir(parents=True, exist_ok=True)


def iter_merged_models(root: Path) -> Iterable[Path]:
    """Yield merged model directories in alphabetical order."""
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def parse_model_dir_name(name: str) -> Optional[re.Match]:
    """Parse merged model directory names."""
    pattern = re.compile(r"(?P<method>.+)_merge_(?P<locale>[a-z]{2}-[A-Z]{2})(?:_(?P<count>\d+)merged)?$")
    return pattern.match(name)


def should_skip(method: str, locale: str, methods_filter: Optional[set], locales_filter: Optional[set]) -> bool:
    """Return True if the entry should be skipped based on filters."""
    if methods_filter and method not in methods_filter:
        return True
    if locales_filter and locale not in locales_filter:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate existing merged models into distinct result folders.")
    parser.add_argument("--methods", nargs="+", help="Restrict evaluation to these merge methods (e.g., average fisher).")
    parser.add_argument("--locales", nargs="+", help="Restrict evaluation to these locale codes (e.g., af-ZA ja-JP).")
    parser.add_argument("--max-models", type=int, default=None, help="Stop after evaluating this many models.")
    parser.add_argument("--force", action="store_true", help="Re-run evaluations even if results.json already exists.")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (defaults to True).")

    args = parser.parse_args()

    repo_root = Path.cwd()
    merged_root = repo_root / "merged_models"
    results_root = repo_root / "results"
    results_root.mkdir(exist_ok=True)

    methods_filter = set(args.methods) if args.methods else None
    locales_filter = set(args.locales) if args.locales else None

    hf_home = Path(os.environ.get("HF_HOME", repo_root / ".hf_cache")).resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    if args.offline or "HF_DATASETS_OFFLINE" not in os.environ:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    ensure_cache_dirs(hf_home)

    evaluated = 0
    skipped = 0
    failures: list[str] = []

    for model_dir in iter_merged_models(merged_root):
        match = parse_model_dir_name(model_dir.name)
        if not match:
            continue

        method = match.group("method")
        locale = match.group("locale")
        raw_count = match.group("count")

        if should_skip(method, locale, methods_filter, locales_filter):
            continue

        if raw_count is not None:
            num_languages = int(raw_count)
        else:
            num_languages = detect_num_languages_from_model_path(str(model_dir))
            if not num_languages:
                num_languages = 4  # Legacy default when details are missing.

        prefix = f"{method}_{num_languages}lang"
        expected_results_dir = results_root / f"{prefix}_{locale}"

        results_file = expected_results_dir / "results.json"
        if results_file.exists() and not args.force:
            print(f"[SKIP] {model_dir.name} -> {expected_results_dir} (existing results)")
            skipped += 1
            continue

        # Ensure results directory matches our expectations.
        eval_folder = create_results_folder(str(model_dir), locale, prefix)
        eval_folder_path = Path(eval_folder)
        if eval_folder_path != expected_results_dir:
            # Harmonize by using the expected path explicitly.
            expected_results_dir.mkdir(parents=True, exist_ok=True)
            eval_folder_path = expected_results_dir

        print(f"[RUN] {model_dir.name} -> {eval_folder_path}")

        try:
            results = evaluate_specific_model(str(model_dir), locale, str(eval_folder_path))
        except Exception as exc:  # Ensure one failure doesn't halt all evaluations.
            print(f"[FAIL] {model_dir.name}: {exc}")
            failures.append(model_dir.name)
            continue

        if results:
            save_evaluation_results(results, str(eval_folder_path))
            evaluated += 1
        else:
            print(f"[FAIL] {model_dir.name}: evaluate_specific_model returned no data")
            failures.append(model_dir.name)

        if args.max_models and evaluated >= args.max_models:
            break

    print(f"\nEvaluation complete. Evaluated={evaluated}, skipped={skipped}, failures={len(failures)}")
    if failures:
        print("Failures:", ", ".join(failures))

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
