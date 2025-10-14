**Project Overview**
- Merge multiple fine-tuned models to introduce or improve a target language using language similarity (URIEL) and Fisher-based merging.
- Core idea: compute per-parameter Fisher importance on a shared dataset slice (e.g., MASSIVE) and combine source models via a Fisher-weighted average, with optional URIEL pre-weights.

**Features**
- Merging methods:
  - `linear`: simple weighted average of parameters
  - `similarity`: auto-select top-K source languages via similarity CSV
  - `average`: equal weights across selected languages
  - `manual`: user-specified weights
  - `fisher_dataset`: gradient-based Fisher merging using HF datasets (target-only, sources-only, or both)
- URIEL cosine weighting:
  - Pre-weight source models by similarity to the target language
  - Alternatively use equal pre-weights for fairness baselines
- Evaluation: saves merged model, details, and runs evaluation after merging

**Prerequisites**
- Python 3.9+
- Install packages (GPU recommended):
  - `pip install torch torchvision torchaudio` (choose CUDA build that fits your system)
  - `pip install transformers datasets accelerate tqdm numpy pandas`
- Hugging Face datasets access (for MASSIVE): `AmazonScience/massive`

**Repository Layout**
- `merginguriel/run_merging_pipeline_refactored.py`: main, composable CLI pipeline
- `submodules/auto_merge_llm/auto_merge_llm/methods/fisher_dataset.py`: dataset-enabled Fisher wrapper (builds shared dataloader from locales)
- `submodules/auto_merge_llm/auto_merge_llm/methods/fisher.py`: full Fisher core (gradient-based merging)
- `submodules/auto_merge_llm/auto_merge_llm/methods/__init__.py`: merging method registry
- `sparsed_language_similarity_matrix_unified.csv`: cosine similarities between locales
- `merginguriel/`: utilities and evaluation

**Data Source: MASSIVE**
- Dataset: `AmazonScience/massive`
- Subsets by locale (e.g., `af-ZA`, `th-TH`, `sq-AL`)
- Text column: `utt` (defaulted in the CLI)

**Quick Start**
- Choose a target locale and number of source languages (top-K).
- Use `fisher_dataset` to compute Fisher on target-only, sources-only, or both.

**Examples**
- Equal-weight Fisher on target-only data (recommended when target text available):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode target --preweight equal`
- URIEL-weighted Fisher on target + sources:
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode both --preweight uriel`
- Equal-weight Fisher on the 5 sources (no target text):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode sources --preweight equal`
- Average (equal) merge without Fisher (baseline):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode average --target-lang af-ZA --num-languages 5`

Additional locale examples (swap `--target-lang`):
- `--target-lang th-TH` (Thai), `--target-lang sq-AL` (Albanian), `--target-lang id-ID` (Indonesian)

**Key Flags**
- `--mode`: `average` | `similarity` | `manual` | `fisher_dataset`
- `--target-lang`: target locale (e.g., `af-ZA`)
- `--num-languages`: number of source languages to include (top-K by similarity)
- Similarity options: `--similarity-source {sparse|dense}`, `--top-k`, `--sinkhorn-iters`
- Dataset config: `--dataset-name`, `--dataset-split`, `--text-column` (MASSIVE uses `utt`)
- Fisher config: `--num-fisher-examples`, `--fisher-data-mode {target|sources|both}`
- Pre-weighting: `--preweight {equal|uriel}`
- Loader config: `--batch-size`, `--max-seq-length`

**Large-Scale Experiments**
- Run sweeps across locales with an explicit list of modes using `merginguriel/run_large_scale_experiment.py`.
- Discover available merge methods: `python merginguriel/run_large_scale_experiment.py --list-modes`
- Run over selected locales and modes (baseline evaluates the base model):
  - `python merginguriel/run_large_scale_experiment.py --locales af-ZA sq-AL --modes baseline similarity average fisher_dataset --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode target --preweight equal`

Presets (for Fisher configs):
- `--preset fairness`: sources-only Fisher + equal preweights (fair baseline)
  - `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset fairness --num-languages 5`
- `--preset target`: target-only Fisher + URIEL preweights (biased toward target)
  - `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset target --num-languages 5`
- Explicit flags (e.g., `--preweight`, `--fisher-data-mode`) override presets.

**How `fisher_dataset` Works**
- `merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset`
  - Selects source locales and preweights (equal or URIEL cosine).
  - Builds a shared HF dataloader from `dataset_name/split/text_column`:
    - Data scope via `--fisher-data-mode {target|sources|both}`.
    - Caps total to `--num-fisher-examples` (trims last batch if needed).
  - Delegates to core Fisher merger (`submodules/auto_merge_llm/auto_merge_llm/methods/fisher.py`):
    - Loads base + source models, aligns tokenizers.
    - Computes Fisher per model over the shared dataloader.
    - Normalizes/aggregates Fisher; applies `fisher_scaling_coefficients` (from preweights).
    - Merges parameters and returns merged model + tokenizer.
  - Saves to `merged_models/fisher_dataset_merge_{target_lang}` and runs evaluation.

Flow overview:
`CLI` → `select locales + preweights` → `build shared dataloader` → `FisherMerging.merge()` → `save + eval`

**NxN Evaluation**
- Evaluate each local model against all target locales (cross‑lingual matrix) using `merginguriel/run_nxn_evaluation.py`.
- Discovers models by scanning `haryos_model/` for folders named `xlm-roberta-base_massive_k_{locale}` (e.g., `..._fr-FR`). No CSV required.
- Expects MASSIVE‑style locales (e.g., `fr-FR`, `th-TH`).
- Examples:
  - List discovered locales: `python merginguriel/run_nxn_evaluation.py --list-locales`
  - Run NxN over all discovered: `python merginguriel/run_nxn_evaluation.py`
  - Run subset: `python merginguriel/run_nxn_evaluation.py --locales fr-FR th-TH --max-workers 1`

**Build/Refresh Similarity Matrices**
- Generate a dense and sparsified+Sinkhorn matrix with MASSIVE locales as indices/columns:
  - `python merginguriel/generate_sparse_similarity_unified.py --k 20 --iterations 20 --feature-type syntax_knn --locales-csv model_mapping_unified.csv --out-dense language_similarity_matrix_unified.csv --out-sparse sparsed_language_similarity_matrix_unified.csv`
- Notes:
  - Uses lang2vec features; ensure `lang2vec` is available in `submodules/lang2vec`.
  - `--k 20` preserves top-20 neighbors per language; you can select any `--num-languages <= 20` at runtime.
  - Sinkhorn normalization balances rows and columns to reduce hub biases and improve fairness.

**Recommended Settings**
- Data for Fisher:
  - Best: target-only data
  - If no target text: `sources` or `both`, with `--preweight uriel` to bias toward target-like sources
- Data size: 300–1000 examples (good), 1000–2000 (robust). Diminishing returns beyond a few thousand short texts.
- Normalization: internal Fisher-norm normalization stabilizes contributions; can be exposed as a flag if you need exact URIEL proportions.

**Merging Method Examples**
- `similarity` (auto top‑K by language similarity; linear merge):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang th-TH --num-languages 5 --similarity-source dense --top-k 20 --sinkhorn-iters 20`
- `average` (equal weights across selected languages; linear merge):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode average --target-lang th-TH --num-languages 5`
- `uriel` (URIEL‑weighted linear merge; uses a fixed source→target mapping inside the calculator):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode uriel --target-lang th-TH`
- `manual` (custom weights for linear merge):
  - Note: The refactored pipeline’s manual calculator holds example weights in code. For custom weights, edit `ManualWeightCalculator` or use `run_merging_pipeline.py`.
- `fisher_simple` (magnitude‑proxy Fisher; can pass URIEL preweights):
  - Equal: `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_simple --target-lang th-TH --num-languages 5`
  - URIEL: `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_simple --target-lang th-TH --num-languages 5 --preweight uriel`
- `fisher_dataset` (full, dataset‑enabled Fisher; recommended):
  - Target‑only: `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang th-TH --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode target --preweight equal`
  - Sources‑only: `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang th-TH --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode sources --preweight uriel`
  - Both: `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang th-TH --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode both --preweight uriel`

**Train Missing Locale Models**
- If some locales are missing under `haryos_model/`, train them from MASSIVE using the helper:
  - Dry-run (print commands):
    - `python merginguriel/train_missing_models.py --dry-run`
  - Train a subset with defaults (xlm-roberta-base, 3 epochs):
    - `python merginguriel/train_missing_models.py --locales af-ZA sq-AL --fp16`
  - Options:
    - `--mapping-csv model_mapping_unified.csv` (source of locales)
    - `--base-model xlm-roberta-base` (HF base)
    - `--train-bs 32 --eval-bs 64 --epochs 3 --lr 5e-5`
    - `--max 5` to limit how many to train this run
    - Outputs to `haryos_model/xlm-roberta-base_massive_k_{locale}` per locale

- If your training saved into `checkpoint_results/` (e.g., due to wandb auto-naming), copy runs into `haryos_model/`:
  - Dry-run to preview copies:
    - `python merginguriel/copy_models_to_haryos.py --dry-run`
  - Copy all runs detected under `checkpoint_results/` into the expected layout:
    - `python merginguriel/copy_models_to_haryos.py --overwrite`
  - Copy specific runs:
    - `python merginguriel/copy_models_to_haryos.py --runs checkpoint_results/xlm-roberta-base_massive_af-ZA_lr5e-5_ep15`

Advanced methods (available in the underlying library):
- Methods like `regmean`, `slerp`, `breadcrumbs`, `task_arithmetic`, `ties`, `widen`, `stock` are registered in `submodules/auto_merge_llm`. Some require extra `method_params` (e.g., trainers or dataset batches). If you want to expose these in the refactored CLI, we can wire them similarly to `fisher_dataset` and add documented flags.

**Outputs**
- Merged model + tokenizer: `merged_models/{mode}_merge_{target_lang}`
- Merge details file with model list and weights
- Evaluation runs post-merge

**Single-Model Evaluation**
- Evaluate a single local model on a MASSIVE locale using `evaluate_specific_model.py`.
- Expects MASSIVE‑style locale codes (e.g., `fr-FR`, `th-TH`).
- Requires the model config to include `id2label/label2id` mappings.
- Examples:
  - Evaluate a local model folder:
    - `python merginguriel/evaluate_specific_model.py --base-model haryos_model/xlm-roberta-base_massive_k_fr-FR --locale th-TH`
  - Choose a results folder prefix (for organization):
    - `python merginguriel/evaluate_specific_model.py --base-model haryos_model/xlm-roberta-base_massive_k_fr-FR --locale th-TH --prefix single_eval`

**Troubleshooting**
- MASSIVE subset not found: verify locale codes (e.g., `af-ZA`).
- First run may download datasets: ensure network is available.
- Torch/CUDA: install a build compatible with your hardware.
- `fisher_dataset` delegation: this mode builds a shared dataloader from the dataset, then delegates the Fisher computation to the core merger in `submodules/auto_merge_llm/auto_merge_llm/methods/fisher.py`. Seeing `fisher.py` in stack traces is expected for `--mode fisher_dataset`.
- Partial last batch warning: if `--num-fisher-examples` is not divisible by `--batch-size`, the last batch may be partial. The implementation trims the last batch and uses exactly the requested number of examples. To avoid the warning entirely, choose values where `num_fisher_examples % batch_size == 0`.
- `--top-k` note: `--top-k` only applies when `--similarity-source dense` is used to compute similarities on-the-fly. With the default `sparse` similarities, `--top-k` is ignored.

**Extending**
- Add strategies under `submodules/auto_merge_llm/auto_merge_llm/methods` and register in `submodules/auto_merge_llm/auto_merge_llm/methods/__init__.py`.
- For new corpora, adapt `fisher_dataset.py` to construct a shared dataloader.

**How to Kill Iterative Training:**

Sometimes iterative training can become unresponsive or you may need to stop it. Here are the methods to kill the process:

1. **Find the Process ID (PID):**
   ```bash
   ps aux | grep python | grep iterative
   ```
   Look for the process running `run_iterative_training.py`

2. **Force Kill the Process:**
   ```bash
   kill -9 <PID>
   ```
   Replace `<PID>` with the actual process ID (e.g., `kill -9 916746`)

3. **Check if Process is Still Running:**
   ```bash
   ps aux | grep python | grep iterative
   ```

4. **Monitor GPU Usage (if needed):**
   ```bash
   nvidia-smi
   ```

**Why Ctrl+C May Not Work:**
- Sequential training can become unresponsive during model loading/saving
- GPU operations may prevent interrupt signals from being processed
- Training loops with heavy computation may not check for interrupts frequently
- Large batch sizes can make the system unresponsive

**Prevention Tips:**
- Use appropriate batch sizes for your GPU memory
- Monitor GPU memory usage with `nvidia-smi` during training
- Consider using `--max-epochs 1` for testing before full training runs
- Use `--sequential-training` (enabled by default) to prevent OOM issues