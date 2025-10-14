# Repository Guidelines

## Project Structure & Module Organization
- `merginguriel/` — entry points and utilities
  - `run_merging_pipeline_refactored.py` — main merging CLI
  - `run_large_scale_experiment.py` — multi‑locale sweeps
  - `training_bert.py` — train MASSIVE locale models
  - Other helpers: evaluation, copying, training utilities
- `submodules/auto_merge_llm/auto_merge_llm/methods/` — merge implementations (linear, fisher, regmean, etc.)
- `haryos_model/` — local fine‑tuned locale models (one folder per locale)
- `merged_models/` — outputs of merges
- `sparsed_language_similarity_matrix_unified.csv` — locale similarity weights
- `README.md` — usage; `EXPERIMENT_README.md` — experiment tips

## Build, Test, and Development Commands
- Run a single merge
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 200 --fisher-data-mode target --preweight equal`
- Run a sweep
  - `python merginguriel/run_large_scale_experiment.py --locales af-ZA sq-AL --modes baseline similarity average fisher_dataset`
- Train missing locale models
  - `python merginguriel/train_missing_models.py --dry-run`

## Coding Style & Naming Conventions
- Python 3.9+, PEP 8, 4‑space indentation, type hints where practical.
- Descriptive names (`target_lang`, `num_fisher_examples`); avoid single‑letter vars.
- Keep changes minimal and composable. For new merge logic:
  - Add a class in `submodules/auto_merge_llm/.../methods/` and register it in `__init__.py`.
  - If exposing via CLI, add a Strategy in `run_merging_pipeline_refactored.py` and wire flags.

## Testing Guidelines
- Smoke test locally with small Fisher settings (e.g., `--num-fisher-examples 64 --batch-size 16`).
- Validate outputs exist under `merged_models/{mode}_merge_{locale}` and run evaluation.
- If `pytest` is available: `pytest -q` (tests are lightweight sanity checks).

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., "Add FisherDatasetStrategy param trim").
- PRs: include summary, commands to reproduce, expected vs. actual behavior, and any metrics (accuracy/STS).
- Link related issues. Include before/after example commands when changing defaults.

## Security & Configuration Tips
- Offline/airgapped: ensure locale folders exist under `haryos_model/` and set `TRANSFORMERS_CACHE` if needed.
- Dataset access: MASSIVE requires network on first load; use local caches afterwards.
