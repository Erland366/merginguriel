**Project Overview**
- Merge multiple fine-tuned models to introduce or improve a target language using language similarity (URIEL) and Fisher-based merging.
- Core idea: compute per-parameter Fisher importance on a shared dataset slice (e.g., MASSIVE) and combine source models via a Fisher-weighted average, with optional URIEL pre-weights.
- **Multi-Model Support**: Works with different base models including `xlm-roberta-base` and `xlm-roberta-large`, allowing you to choose the model size that best fits your computational resources and performance requirements.

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
- Merge count tracking: Output folders include the number of models merged (e.g., `similarity_merge_sq-AL_5merged`)
- Evaluation: saves merged model, details, and runs evaluation after merging

**Model Support**

The system supports multiple pre-trained models as base architectures:

**Supported Models:**
- **`xlm-roberta-base`** (default): 270M parameters, good balance of performance and efficiency
- **`xlm-roberta-large`**: 550M parameters, higher performance but requires more GPU memory and compute time
- **Custom models**: Any HuggingFace model compatible with the XLM-RoBERTa architecture

**Model Selection Trade-offs:**
- **Base model**: Faster training/inference, lower memory usage (~8GB GPU), good for experimentation
- **Large model**: Better performance on low-resource languages, higher memory requirements (~16GB GPU), better for final production models

**Prerequisites**
- Python 3.9+
- Install packages (GPU recommended):
  - `pip install torch torchvision torchaudio` (choose CUDA build that fits your system)
  - `pip install transformers datasets accelerate tqdm numpy pandas`
- Hugging Face datasets access (for MASSIVE): `AmazonScience/massive`

**Repository Layout**
- `merginguriel/run_merging_pipeline_refactored.py`: main, composable CLI pipeline with multi-model support
- `submodules/auto_merge_llm/auto_merge_llm/methods/fisher_dataset.py`: dataset-enabled Fisher wrapper (builds shared dataloader from locales)
- `submodules/auto_merge_llm/auto_merge_llm/methods/fisher.py`: full Fisher core (gradient-based merging)
- `submodules/auto_merge_llm/auto_merge_llm/methods/__init__.py`: merging method registry
- `sparsed_language_similarity_matrix_unified.csv`: cosine similarities between locales
- `merginguriel/`: utilities and evaluation
- `haryos_model/`: trained models with subdirectories by base model:
  - `haryos_model/xlm-roberta-base_massive_k_{locale}/` (default)
  - `haryos_model/xlm-roberta-large_massive_k_{locale}/` (large models)
  - `haryos_model/{custom-model}_massive_k_{locale}/` (other models)

**Data Source: MASSIVE**
- Dataset: `AmazonScience/massive`
- Subsets by locale (e.g., `af-ZA`, `th-TH`, `sq-AL`)
- Text column: `utt` (defaulted in the CLI)

**Quick Start**
- Choose a target locale and number of source languages (top-K).
- Use `fisher_dataset` to compute Fisher on target-only, sources-only, or both.

**Examples**

**XLMR-Base Examples (Default):**
- Equal-weight Fisher on target-only data (recommended when target text available):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode target --preweight equal`
- URIEL-weighted Fisher on target + sources:
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode both --preweight uriel`
- Equal-weight Fisher on the 5 sources (no target text):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode sources --preweight equal`
- Average (equal) merge without Fisher (baseline):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode average --target-lang af-ZA --num-languages 5`

**XLMR-Large Examples (Higher Performance):**
- XLMR-Large similarity merge:
  - `python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang af-ZA --base-model xlm-roberta-large --num-languages 5`
- XLMR-Large Fisher merge (target-only):
  - `python merginguriel/run_merging_pipeline_refactored.py --mode fisher_dataset --target-lang af-ZA --base-model xlm-roberta-large --num-languages 5 --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 500 --fisher-data-mode target --preweight equal`
- XLMR-Large ensemble inference:
  - `python merginguriel/uriel_ensemble_inference.py --target-lang af-ZA --base-model xlm-roberta-large --voting-method uriel_logits --num-languages 5`

Additional locale examples (swap `--target-lang`):
- `--target-lang th-TH` (Thai), `--target-lang sq-AL` (Albanian), `--target-lang id-ID` (Indonesian)

**Key Flags**
- `--mode`: `average` | `similarity` | `manual` | `fisher_dataset`
- `--base-model`: base model architecture (default: `xlm-roberta-base`, options: `xlm-roberta-large`, `FacebookAI/xlm-roberta-large`)
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
- List all available locales: `python merginguriel/run_large_scale_experiment.py --list-locales`

**Basic Usage:**
- Run for all locales with default settings (5 languages per merge):
  - `python merginguriel/run_large_scale_experiment.py`

**3-Language Merging (Recommended for efficiency):**
- To merge 3 languages per target language (more computationally efficient):
  - `python merginguriel/run_large_scale_experiment.py --num-languages 3`
- Specific modes with 3 languages:
  - `python merginguriel/run_large_scale_experiment.py --num-languages 3 --modes baseline similarity average fisher`

**Advanced Options:**
- Run over selected locales and modes (baseline evaluates the base model):
  - `python merginguriel/run_large_scale_experiment.py --locales af-ZA sq-AL --modes baseline similarity average fisher_dataset --dataset-name AmazonScience/massive --dataset-split train --text-column utt --num-fisher-examples 1000 --fisher-data-mode target --preweight equal`

Presets (for Fisher configs):
- `--preset fairness`: sources-only Fisher + equal preweights (fair baseline)
  - `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset fairness --num-languages 5`
  - **3-language version**: `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset fairness --num-languages 3`
- `--preset target`: target-only Fisher + URIEL preweights (biased toward target)
  - `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset target --num-languages 5`
  - **3-language version**: `python merginguriel/run_large_scale_experiment.py --locales th-TH --modes fisher_dataset --preset target --num-languages 3`
- Explicit flags (e.g., `--preweight`, `--fisher-data-mode`) override presets.

**Additional 3-Language Examples:**
- Test with limited locales first:
  - `python merginguriel/run_large_scale_experiment.py --num-languages 3 --max-locales 5`
- Resume from specific locale if interrupted:
  - `python merginguriel/run_large_scale_experiment.py --num-languages 3 --start-from 25`
- Using sparse similarities:
  - `python merginguriel/run_large_scale_experiment.py --num-languages 3 --similarity-source sparse`

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
  - Saves to `merged_models/fisher_dataset_merge_{target_lang}_{N}merged` (where N is the number of models merged) and runs evaluation.

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

**Multi-Language Training with Leave-One-Language-Out Support**

**NEW FEATURE**: The training script now supports multi-language datasets for training models on multiple languages simultaneously, with automatic dataset concatenation and shuffling.

**Key Features:**
- ✅ **Backward Compatible**: Single language training works exactly as before
- ✅ **Multi-Language Support**: Train on multiple languages by providing comma-separated language codes
- ✅ **Leave-One-Language-Out**: Train on multiple languages, evaluate on a held-out language
- ✅ **Automatic Processing**: Datasets are concatenated, shuffled, and processed automatically
- ✅ **Enhanced Logging**: Detailed information about language composition and dataset sizes

**Usage Examples:**

1. **Single Language Training (Original functionality):**
   ```bash
   python merginguriel/training_bert.py --dataset_config_name fr-FR --output_dir results/fr_french
   ```

2. **Multi-Language Training (NEW):**
   ```bash
   # Train on multiple European languages
   CUDA_VISIBLE_DEVICES=1 python merginguriel/training_bert.py \
       --dataset_config_name fr-FR,de-DE,es-ES,it-IT \
       --output_dir results/multilang_european \
       --model_name_or_path xlm-roberta-base \
       --num_train_epochs 10 \
       --do_train --do_eval --do_predict
   ```

3. **Large Multi-Language Training:**
   ```bash
   # Train on 7 languages from different families
   python merginguriel/training_bert.py \
       --dataset_config_name fr-FR,de-DE,es-ES,it-IT,nl-NL,pt-PT,ru-RU \
       --output_dir results/multilang_7langs \
       --max_train_samples 10000 \
       --do_train --do_eval
   ```

4. **Leave-One-Language-Out Training:**
   ```bash
   # Train on multiple languages, then evaluate on a held-out language
   # Step 1: Train on source languages
   python merginguriel/training_bert.py \
       --dataset_config_name fr-FR,de-DE,es-ES,it-IT,nl-NL \
       --output_dir results/source_languages_only \
       --do_train --do_eval --do_predict

   # Step 2: Load the trained model and evaluate on held-out language (en-US)
   python merginguriel/evaluate_specific_model.py \
       --base-model results/source_languages_only \
       --locale en-US
   ```

**Working Examples (✅ Tested):**

   **Quick Test (Small Scale):**
   ```bash
   # Quick test with small dataset for validation
   CUDA_VISIBLE_DEVICES=1 python merginguriel/training_bert.py \
       --dataset_config_name fr-FR,de-DE,es-ES,it-IT \
       --output_dir results/multilang_european_test \
       --model_name_or_path xlm-roberta-base \
       --num_train_epochs 1 \
       --max_train_samples 50 \
       --max_eval_samples 20 \
       --max_predict_samples 20 \
       --do_train --do_eval --do_predict \
       --overwrite_output_dir
   ```

   **Full Training (Production Ready):**
   ```bash
   # Full 10-epoch training with 4 European languages - FULLY TESTED & WORKING
   CUDA_VISIBLE_DEVICES=1 python merginguriel/training_bert.py \
       --dataset_config_name fr-FR,de-DE,es-ES,it-IT \
       --output_dir results/multilang_european_test \
       --model_name_or_path xlm-roberta-base \
       --num_train_epochs 10 \
       --do_train --do_eval --do_predict
   ```

   **Results:** Successfully loads 46,056 training samples across 4 languages, combines and shuffles datasets, and trains XLM-RoBERTa without issues.

**Supported Language Formats:**
- **Single language**: `--dataset_config_name fr-FR`
- **Comma-separated**: `--dataset_config_name fr-FR,de-DE,es-ES`
- **List format**: The script automatically parses comma-separated strings into lists

**Implementation Details:**

1. **Automatic Dataset Processing:**
   - Each language dataset is loaded separately
   - Datasets are concatenated using `datasets.concatenate_datasets()`
   - Combined dataset is shuffled to mix languages randomly
   - All MASSIVE languages have consistent 60 intent classes

2. **Enhanced Logging:**
   ```
   Loading and combining 4 language configurations: ['fr-FR', 'de-DE', 'es-ES', 'it-IT']
   Loading fr-FR...
   ✓ fr-FR: 11514 training samples
   Loading de-DE...
   ✓ de-DE: 11514 training samples
   Combining datasets...
   Combined training set: 46056 samples
   Shuffling datasets with seed 42
   Class distribution in train set:
     Languages: 4 languages: ['fr-FR', 'de-DE', 'es-ES', 'it-IT']
     Total samples: 46056
   ```

3. **Wandb Integration:**
   - Multi-language experiments are properly tracked
   - Dataset configuration includes number of languages and language list
   - Run names automatically reflect multi-language nature

**Dataset Configuration Argument:**
The `--dataset_config_name` argument now accepts:
- **String**: Single language code (e.g., `fr-FR`)
- **Comma-separated string**: Multiple languages (e.g., `fr-FR,de-DE,es-ES`)

**Common Use Cases:**

1. **Cross-Lingual Transfer**: Train on related languages to improve performance on a target language
2. **Multi-Lingual Models**: Create models that work well across multiple languages
3. **Low-Resource Languages**: Boost performance by training on similar high-resource languages
4. **Language Family Training**: Train on languages from the same family (e.g., Romance languages)

**Training Workflow:**
1. **Load Languages**: Each specified language dataset is loaded from MASSIVE
2. **Combine Datasets**: All training sets are concatenated into one large dataset
3. **Shuffle**: Combined dataset is shuffled to mix language examples
4. **Process**: Tokenization and preprocessing applied to combined dataset
5. **Train**: Model trained on multi-language data
6. **Evaluate**: Evaluation on validation/test sets from all languages

**Benefits:**
- **Improved Zero-Shot Performance**: Better cross-lingual transfer capabilities
- **Data Efficiency**: Leverage multiple languages for more robust training
- **Language Diversification**: Models learn patterns across different languages
- **Flexible Experiments**: Easy to experiment with different language combinations

**Train Missing Locale Models**
- If some locales are missing under `haryos_model/`, train them from MASSIVE using the helper:
  - Dry-run (print commands):
    - `python merginguriel/train_missing_models.py --dry-run`
  - Train XLMR-Base models (default, 3 epochs):
    - `python merginguriel/train_missing_models.py --locales af-ZA sq-AL --fp16`
  - Train XLMR-Large models (higher performance):
    - `python merginguriel/train_missing_models.py --base-model FacebookAI/xlm-roberta-large --locales af-ZA sq-AL --train-bs 16 --eval-bs 16 --fp16`
  - Train individual locale with XLMR-Large:
    - `python merginguriel/training_bert.py --model_name_or_path FacebookAI/xlm-roberta-large --dataset_config_name af-ZA --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --fp16`
  - Options:
    - `--mapping-csv model_mapping_unified.csv` or `available_locales.txt` (source of locales)
    - `--base-model` (HF base model: `xlm-roberta-base`, `FacebookAI/xlm-roberta-large`)
    - `--train-bs 32 --eval-bs 64 --epochs 3 --lr 5e-5` (training hyperparameters)
    - `--max 5` to limit how many to train this run
    - Outputs to `haryos_model/{base-model}_massive_k_{locale}` per locale

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
- Merged model + tokenizer: `merged_models/{mode}_merge_{target_lang}_{N}merged` (where N is the number of models merged)
  - Examples: `merged_models/similarity_merge_sq-AL_5merged`, `merged_models/average_merge_am-ET_3merged`
- Merge details file with model list, weights, and configuration
- Evaluation runs post-merge

**Merge Count Feature**: The output folder naming now includes the number of models merged, providing immediate insight into merge complexity and making it easy to compare different merge configurations at a glance.

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

# Iterative Training and Merging

## Overview

Iterative training is a powerful approach that trains multiple language models **sequentially** and merges them periodically during training, rather than only after complete training. This allows for deeper knowledge integration between languages throughout the training process.

## How It Works

### Sequential Training (VRAM-Friendly)

Unlike simultaneous training that would require multiple models in GPU memory at once, iterative training uses **sequential training** to work with limited VRAM:

```
🚀 ITERATIVE TRAINING CYCLE
│
├── 🤖 Model 1: Train en-US for N epochs
│   ├── 📈 Training epochs 1, 2, 3...
│   ├── 💾 Save checkpoint
│   └── 🧹 Clear GPU memory
│
├── 🤖 Model 2: Train fr-FR for N epochs
│   ├── 📈 Training epochs 1, 2, 3...
│   ├── 💾 Save checkpoint
│   └── 🧹 Clear GPU memory
│
├── 🤖 Model 3: Train de-DE for N epochs
│   ├── 📈 Training epochs 1, 2, 3...
│   ├── 💾 Save checkpoint
│   └── 🧹 Clear GPU memory
│
├── 🔄 MERGE POINT: Merge all trained models
│   ├── 📂 Load checkpoints for all models
│   ├── 🤝 Apply merge algorithm (linear, fisher, etc.)
│   ├── 📊 Use URIEL similarity weights
│   └── 💾 Save merged model
│
├── 🔄 Continue cycles (if more epochs needed)
│
└── ✅ FINAL RESULT: Single merged model
```

### The "Iterative" Aspect

The iterative nature happens through **training cycles**:

1. **Cycle 1**: Train all models individually → Merge
2. **Cycle 2**: Continue training from merged model → Merge again
3. **Cycle 3**: Continue training → Final merge

Each merge creates a new starting point for the next training cycle, allowing languages to influence each other progressively.

### Key Benefits

✅ **VRAM-Safe**: Only one model loaded in GPU memory at a time
✅ **Deep Integration**: Languages share knowledge throughout training, not just at the end
✅ **Progressive Learning**: Each merge creates better starting points for continued training
✅ **Flexible Configuration**: Control merge frequency, algorithms, and timing

## Quick Start Commands

### Basic Test (Recommended First)

```bash
# Quick test with 1 epoch to verify everything works
CUDA_VISIBLE_DEVICES=0 python merginguriel/run_iterative_training.py \
  --mode similarity \
  --target-lang sw-KE \
  --max-epochs 1 \
  --merge-frequency 1 \
  --output-dir test_iterative \
  --locales en-US
```

### Multi-Language Example

```bash
# Train 3 languages with periodic merging
CUDA_VISIBLE_DEVICES=0 python merginguriel/run_iterative_training.py \
  --mode similarity \
  --target-lang sw-KE \
  --locales en-US,fr-FR,de-DE \
  --max-epochs 9 \
  --merge-frequency 3 \
  --output-dir iterative_results \
  --batch-size 16 \
  --fp16
```

### Advanced Configuration

```bash
# Full-featured iterative training with adaptive merging
CUDA_VISIBLE_DEVICES=0 python merginguriel/run_iterative_training.py \
  --mode similarity \
  --target-lang sw-KE \
  --locales en-US,fr-FR,de-DE,es-ES,it-IT \
  --max-epochs 15 \
  --merge-frequency 3 \
  --top-k 20 \
  --sinkhorn-iters 20 \
  --batch-size 16 \
  --fp16 \
  --adaptive-merge-frequency \
  --performance-merge-trigger \
  --convergence-threshold 1e-4 \
  --checkpoint-before-merge \
  --enable-wandb \
  --output-dir advanced_iterative
```

### Auto-Select Source Languages

```bash
# Let the system automatically select similar languages
CUDA_VISIBLE_DEVICES=0 python merginguriel/run_iterative_training.py \
  --mode similarity \
  --target-lang sw-KE \
  --max-epochs 12 \
  --merge-frequency 4 \
  --num-languages 5 \
  --top-k 15 \
  --output-dir auto_select_results
```

## Key Parameters

### Training Configuration
- `--max-epochs`: Total training epochs per cycle
- `--merge-frequency N`: Merge every N epochs (controls cycle length)
- `--batch-size`: Training batch size (adjust for VRAM)
- `--fp16`: Use mixed precision (saves memory)

### Merging Configuration
- `--mode`: Merging algorithm (similarity, average, fisher, etc.)
- `--target-lang`: Target language for similarity weighting
- `--num-languages`: Number of source languages to use
- `--top-k`: Consider top-K similar languages
- `--sinkhorn-iters`: Sinkhorn normalization iterations

### Advanced Features
- `--adaptive-merge-frequency`: Intelligently adjust merge timing
- `--performance-merge-trigger`: Merge when performance plateaus
- `--checkpoint-before-merge`: Save checkpoints before each merge
- `--sequential-training`: Train one model at a time (default, VRAM-safe)

## Memory Management

### VRAM Optimization
- **Sequential Training**: Default, loads only one model at a time
- **Memory Cleanup**: Automatically clears GPU cache between models
- **Mixed Precision**: Use `--fp16` to reduce memory usage
- **Batch Size**: Start with `--batch-size 16` for limited VRAM

### Monitoring Memory
```bash
# Monitor GPU usage during training
watch -n 2 nvidia-smi
```

## Expected Outputs

### Directory Structure
```
iterative_results/
├── en-US/
│   └── checkpoints/
│       └── checkpoint_epoch_1.0_step_90
├── fr-FR/
│   └── checkpoints/
├── merged_models/
│   ├── merged_cycle_1/
│   ├── merged_cycle_2/
│   └── final_merged_model/
├── state_summary.json
└── merge_history.json
```

### What to Expect
1. **Training Phase**: Each language trains individually
2. **Merge Phase**: Models are merged using specified algorithm
3. **Cycles**: Process repeats based on `--max-epochs` and `--merge-frequency`
4. **Final Model**: Single merged model incorporating all languages

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
```bash
# Reduce batch size and enable mixed precision
--batch-size 8 --fp16 --max-gpu-memory 4000
```

**Slow Training**:
```bash
# Reduce merge frequency for fewer merge operations
--merge-frequency 5  # Merge every 5 epochs instead of 1
```

**Dataset Loading Issues**:
- Ensure internet connection for first download
- Check locale codes (use `en-US`, not `en_US`)

### Performance Tips

1. **Start Small**: Test with `--max-epochs 1` first
2. **Monitor Resources**: Use `nvidia-smi` to track GPU usage
3. **Adjust Batch Size**: Find the sweet spot for your GPU
4. **Use Mixed Precision**: `--fp16` saves memory and speeds up training

## How Iterative Merging Works: The Magic Behind the System

### **What Happens After Each Merge?**

Iterative merging is a powerful technique that allows models to learn from each other during training. Here's exactly what happens:

#### **Before First Merge (Epoch 1)**
```
en-US model: [English knowledge only] → trains 1 epoch → 44% accuracy
fr-FR model: [French knowledge only] → trains 1 epoch → 40% accuracy
```

#### **During First Merge**
1. **Collection**: System takes both trained models (en-US and fr-FR)
2. **Weighting**: Applies URIEL similarity weights (50% each in this case)
3. **Merging**: Creates a new combined model: 50% English + 50% French knowledge
4. **Distribution**: Updates BOTH original models with the merged weights

#### **After First Merge - The Magic!**
```
en-US model: [50% English + 50% French knowledge]
fr-FR model: [50% English + 50% French knowledge]
```
**Key Point**: Both models now contain cross-lingual knowledge they didn't have before!

#### **Training Continuation (Epoch 2)**
```
en-US model: [Combined knowledge] → trains 1 more epoch → 88% accuracy ✅
fr-FR model: [Combined knowledge] → trains 1 more epoch → 87% accuracy ✅
```
**Result**: Massive improvement because each model learns from the other language's patterns!

#### **Second Merge**
- Takes the enhanced models (88% and 87% accuracy)
- Merges them again with 50% weights
- Creates an even better combined model
- Updates both models with the new merged weights

### **Why This Works So Well**

#### **1. Cross-Language Pattern Transfer**
- English model learns universal patterns from French
- French model learns English patterns that help with classification
- Both benefit from each other's linguistic insights

#### **2. Knowledge Synchronization**
- Prevents models from drifting too far apart
- Regularly shares improvements between models
- Maintains consistency across languages

#### **3. Cumulative Improvement**
```
Traditional: Model A → final A, Model B → final B
Iterative: A+B → better A+B → even better A+B
```

### **Real Training Timeline Example**

With `--max-epochs 3 --merge-frequency 1`:

```
ROUND 1 (Epoch 1):
  en-US: Train epoch 1 (32s) → 44% accuracy
  fr-FR: Train epoch 1 (31s) → 40% accuracy
  ⚡️ MERGE #1 (4.8s) → Both models get combined knowledge

ROUND 2 (Epoch 2):
  en-US: Train epoch 2 (62s) → 88% accuracy ✅ (+44% improvement!)
  fr-FR: Train epoch 2 (62s) → 87% accuracy ✅ (+47% improvement!)
  ⚡️ MERGE #2 (4.8s) → Both models get enhanced combined knowledge

ROUND 3 (Epoch 3):
  en-US: Train epoch 3 → Would start from 88% baseline
  fr-FR: Train epoch 3 → Would start from 87% baseline
  ⚡️ MERGE #3 → Final combined model
```

### **The Key Innovation**

**Traditional Multi-Lingual Training:**
- Each model trains in isolation
- No knowledge sharing during training
- Final models may have very different capabilities

**Iterative Merging:**
- Models share knowledge continuously during training
- Each model benefits from others' discoveries
- Final models have consistent, enhanced capabilities

### **Performance Impact**

In our test case:
- **English model improvement**: 44% → 88% (100% relative improvement)
- **French model improvement**: 40% → 87% (117% relative improvement)
- **Training overhead**: Only ~10 seconds extra per merge cycle
- **Memory usage**: Still sequential (VRAM-friendly)

This is why iterative merging is so powerful - it creates a virtuous cycle where models continuously improve each other during training!

## How to Kill Iterative Training

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

**Results Analysis and Visualization**

The project includes a comprehensive analysis and plotting system that processes experiment results and generates publication-ready visualizations with fair comparisons.

**Features:**
- **Advanced Merging Methods Analysis**: TIES, Task Arithmetic, SLERP, RegMean, DARE, Fisher
- **Ensemble Inference Analysis**: majority, weighted_majority, soft, uriel_logits voting methods
- **Fair Comparison Framework**: Compares methods against best performance from actual source languages used
- **Multiple Baseline Comparisons**: vs average zero-shot, vs best zero-shot, vs best source (fair)
- **Individual Method Plots**: Separate plots for each method to avoid crowding
- **Comprehensive CSV Output**: All results with comparison metrics and source language details

**Running the Analysis System:**

1. **Quick Analysis (Recommended):**
   ```bash
   # Automatically finds latest results and generates all plots
   python plot_results.py
   ```

2. **Use Specific CSV File:**
   ```bash
   # Plot from a specific results file
   python plot_results.py --csv-file results_comprehensive_20251015_064211.csv
   ```

3. **Interactive Interface:**
   ```bash
   # User-friendly interactive plot generator
   python quick_plot.py
   ```

4. **List Available Data:**
   ```bash
   # See all available CSV files
   python plot_results.py --list-csv
   ```

**What the System Generates:**

**CSV Files:**
- `advanced_analysis_summary_YYYYMMDD_HHMMSS.csv`: Complete analysis with:
  - Pure performance scores for all methods
  - Source languages used (extracted from merge_details.txt)
  - Comparison metrics: vs avg zero-shot, vs best zero-shot, vs best source
  - Fair comparison baselines

**Plot Files (in `plots/` directory):**
- **Pure Scores**: `pure_scores_[method]_YYYYMMDD_HHMMSS.png`
  - Shows method performance vs all 4 baselines (baseline, avg zero-shot, best zero-shot, best source)
- **vs Avg Zero-shot**: `vs_avg_zero_[method]_YYYYMMDD_HHMMSS.png`
  - Improvement/degradation over average zero-shot baseline
- **vs Best Zero-shot**: `vs_best_zero_[method]_YYYYMMDD_HHMMSS.png`
  - Improvement/degradation over best individual source model (all languages)
- **vs Best Source**: `vs_best_source_[method]_YYYYMMDD_HHMMSS.png`
  - **FAIR COMPARISON**: Improvement/degradation over best performance from actual source languages used

**Understanding the Fair Comparison Framework:**

The system implements a fair comparison methodology that resolves the paradox where ensemble methods appeared to perform poorly:

- **Traditional vs Best Zero-shot**: Compares against best individual model from ALL possible source languages (unfair)
- **Fair vs Best Source**: Compares against best individual model from ONLY the source languages actually used in the merge

**Example:**
- Target: `af-ZA`
- Source languages used: `['id-ID', 'hy-AM', 'hu-HU', 'ka-GE', 'nl-NL']`
- Best source performance: `0.705` (best among these 5 languages)
- Ensemble performance: `0.494`
- Fair comparison: `-0.211` (realistic performance gap)

**Requirements:**
- Main results CSV (from `aggregate_results.py` or large-scale experiments)
- N-x-N evaluation matrix (from `run_nxn_evaluation.py`)
- Merge details files (in `merged_models/*/merge_details.txt`)

**Output Structure:**
```
plots/
├── pure_scores_similarity_20251015_072835.png
├── pure_scores_average_20251015_072835.png
├── vs_avg_zero_similarity_20251015_072835.png
├── vs_best_zero_similarity_20251015_072835.png
├── vs_best_source_similarity_20251015_072835.png
└── ... (one plot per method)

advanced_analysis_summary_20251015_072835.csv
```

**Usage Workflow:**
1. Run experiments (individual or large-scale)
2. Generate results with `aggregate_results.py` if needed
3. Run analysis: `python plot_results.py`
4. Find plots in `plots/` directory for presentations
5. Use CSV data for detailed analysis and reporting

This system provides comprehensive, fair, and publication-ready analysis of all your MergingUriel experiments!

**Results Aggregation System**

The project includes a powerful aggregation system (`merginguriel/aggregate_results.py`) that processes experiment results from multiple folders and creates comprehensive reports with baseline comparisons.

**What It Does:**
- Scans all experiment result folders in the `results/` directory
- Extracts metadata from `merge_details.txt` files and folder names
- Integrates N-x-N evaluation matrices for baseline comparisons
- Creates pivot tables comparing all methods side-by-side
- Generates comprehensive reports in multiple formats

**Key Features:**

1. **Dynamic Experiment Parsing:**
   - Automatically detects experiment types (similarity, average, task_arithmetic, ties, slerp, regmean, fisher, ensemble methods)
   - Extracts source languages and weights from merge details
   - Handles complex folder naming patterns (e.g., `task_arithmetic_merge_af-ZA`, `ensemble_uriel_logits_sq-AL`)

2. **Baseline Integration:**
   - **Best Source Language**: Best performing individual source model among those actually used
   - **Best Overall Zero-shot**: Best performing model from all available languages (excluding target)
   - **Native Baseline**: Target language's native model performance

3. **Comprehensive Analysis:**
   - Win rate analysis for each method vs baselines
   - Improvement calculations (method - baseline)
   - Statistical summaries for all experiment types
   - Missing experiment detection

**Running the Aggregation:**

1. **Basic Aggregation:**
   ```bash
   # Process all experiments and generate comprehensive report
   python merginguriel/aggregate_results.py
   ```

2. **With Filtering Options:**
   ```bash
   # Filter to specific locales
   python merginguriel/aggregate_results.py --locales af-ZA sq-AL th-TH

   # Filter to specific experiment types
   python merginguriel/aggregate_results.py --experiment-types similarity average fisher

   # Combine filters
   python merginguriel aggregate_results.py --locales af-ZA sq-AL --experiment-types similarity average baseline
   ```

3. **Advanced Options:**
   ```bash
   # Use specific evaluation matrix
   python merginguriel/aggregate_results.py --evaluation-matrix nxn_results/nxn_eval_20251014_231051/evaluation_matrix.csv

   # Skip baseline integration (faster processing)
   python merginguriel/aggregate_results.py --no-baselines

   # Show missing/failed experiments
   python merginguriel/aggregate_results.py --show-missing

   # Enable verbose logging
   python merginguriel/aggregate_results.py --verbose
   ```

**What It Generates:**

**CSV Files:**
- `results_aggregated_YYYYMMDD_HHMMSS.csv`: Raw experiment data with metadata
- `results_comparison_YYYYMMDD_HHMMSS.csv`: Pivot table with methods as columns
- `results_comprehensive_YYYYMMDD_HHMMSS.csv**: Main comparison table (used by plotting system)

**JSON Files:**
- `results_summary_YYYYMMDD_HHMMSS.json`: Statistical summaries for each method
- `results_win_analysis_YYYYMMDD_HHMMSS.json`: Win rate analysis and comparisons

**Markdown Report:**
- `results_report_YYYYMMDD_HHMMSS.md`: Comprehensive human-readable report with:
  - Executive summary
  - Method statistics
  - Win rate analysis
  - Detailed comparison tables
  - Methodology explanation

**Example Output:**
```
Loading evaluation matrix: nxn_results/nxn_eval_20251014_231051/evaluation_matrix.csv
Loaded evaluation matrix with shape (49, 49)
Aggregating experiment results...
Processing 156 experiment results
Creating comparison table...
Generating summary statistics...
Analyzing win rates...

Summary Statistics:
  similarity: 49 experiments, mean accuracy = 0.5422
  average: 49 experiments, mean accuracy = 0.5440
  task_arithmetic: 49 experiments, mean accuracy = 0.5415
  ties: 49 experiments, mean accuracy = 0.5568

Win Rate Analysis Summary:
  similarity: 73.5% win rate vs baseline (36/49)
  average: 75.5% win rate vs baseline (37/49)
  task_arithmetic: 69.4% win rate vs baseline (34/49)

Files generated:
  raw_filename: results_aggregated_20251015_072718.csv
  comparison_filename: results_comparison_20251015_072718.csv
  comprehensive_filename: results_comprehensive_20251015_072718.csv
  summary_filename: results_summary_20251015_072718.json
  win_analysis_filename: results_win_analysis_20251015_072718.json
  markdown_filename: results_report_20251015_072718.md
```

**Integration with Plotting System:**

The aggregation system outputs `results_comprehensive_*.csv` files that are:
- **Automatically detected** by the plotting system
- **Used as input** for generating all visualization plots
- **Essential** for the fair comparison framework

**Required Data Structure:**

The aggregation system expects experiment results in this structure:
```
results/
├── similarity_merge_af-ZA/
│   ├── results.json (experiment performance data)
│   └── merge_details.txt (source languages, weights, metadata)
├── average_merge_af-ZA/
│   ├── results.json
│   └── merge_details.txt
├── task_arithmetic_merge_af-ZA/
│   ├── results.json
│   └── merge_details.txt
└── ensemble_uriel_logits_af-ZA/
    ├── results.json
    └── merge_details.txt
```

**Complete Workflow:**
1. **Run Experiments**: Large-scale or individual experiments create result folders
2. **Aggregate Results**: `python merginguriel/aggregate_results.py` processes all folders
3. **Generate Plots**: `python plot_results.py` uses aggregated CSV for visualization
4. **Present Results**: Use plots and reports for analysis and presentations

---

## 🌍 Multi-Language Training

For simpler multi-language model training without complex merging, use the `run_multilang_training.py` script. This provides three main approaches:

### 🚀 Quick Start

```bash
# Train on all languages except English (LOLO mode)
uv run merginguriel/run_multilang_training.py --exclude en-US

# Train on all 49 languages at once (Super Model)
uv run merginguriel/run_multilang_training.py --super-model

# Run LOLO training for all 49 languages (Batch Mode)
uv run merginguriel/run_multilang_training.py --all-lolo
```

### 📋 Training Modes

#### 1. LOLO (Leave One Language Out)
Train on all languages except one. Useful for:
- Understanding language contribution
- Creating language-specific models
- Testing cross-lingual transfer

```bash
uv run merginguriel/run_multilang_training.py --exclude fr-FR
uv run merginguriel/run_multilang_training.py --exclude sq-AL
```

#### 2. Super Model
Train on all 49 languages simultaneously for maximum multilingual capability:

```bash
uv run merginguriel/run_multilang_training.py --super-model
```

#### 3. All LOLO (Batch Mode)
Run LOLO training for all 49 languages in sequence. This creates 49 separate models, each trained on all languages except one:

```bash
uv run merginguriel/run_multilang_training.py --all-lolo
```

**Features:**
- **Batch Processing**: Automatically runs LOLO training for all 49 languages
- **Progress Tracking**: Shows progress bar and detailed logging for each run
- **Error Resilience**: Continues with remaining languages even if some runs fail
- **Summary Report**: Displays success/failure statistics upon completion

**Output Models Created:**
- `xlm-roberta-base_massive_LOLO_without_af_ZA`
- `xlm-roberta-base_massive_LOLO_without_am_ET`
- `xlm-roberta-base_massive_LOLO_without_ar_SA`
- ... (49 total models, one for each language)

### ⚙️ Configuration Options

```bash
# Custom output directory
uv run merginguriel/run_multilang_training.py --exclude de-DE --output_dir my_model

# Custom training parameters
uv run merginguriel/run_multilang_training.py --super-model \
  --additional-args --learning_rate 1e-4 --batch_size 32 --num_train_epochs 10
```

### 📊 Supported Languages

All 49 MASSIVE languages:
```
af-ZA, am-ET, ar-SA, az-AZ, bn-BD, ca-ES, cy-GB, da-DK, de-DE, el-GR,
en-US, es-ES, fa-IR, fi-FI, fr-FR, hi-IN, hu-HU, hy-AM, id-ID, is-IS,
it-IT, ja-JP, jv-ID, ka-GE, km-KH, kn-IN, ko-KR, lv-LV, ml-IN, mn-MN,
ms-MY, my-MM, nb-NO, nl-NL, pl-PL, pt-PT, ro-RO, ru-RU, sl-SL, sq-AL,
sw-KE, ta-IN, te-IN, th-TH, tl-PH, tr-TR, ur-PK, vi-VN, zh-TW
```

### 🎯 Output Naming

Models are saved in the `haryos_model/` directory following this naming convention:

- **LOLO Mode**: `haryos_model/xlm-roberta-base_massive_LOLO_without_{language}` (e.g., `xlm-roberta-base_massive_LOLO_without_en_US`)
- **Super Model**: `haryos_model/xlm-roberta-base_massive_LOLO_all_languages`
- **All LOLO Mode**: Creates 49 models with naming `haryos_model/xlm-roberta-base_massive_LOLO_without_{language}` for each language
- **Custom**: Whatever you specify with `--output-dir` (still placed in `haryos_model/`)

### 🔧 Default Settings

- **Model**: `xlm-roberta-base`
- **Epochs**: 15
- **Training**: `--do_train --do_eval --do_predict`
- **Other parameters**: Use defaults from `training_bert.py`

### 💡 When to Use Multi-Language Training

**Multi-Language Training** is ideal when you want:
- **Simplicity**: Single training run vs complex iterative merging
- **Speed**: No sequential training and merging cycles
- **Cross-lingual Transfer**: Languages learn from each other during training
- **Memory Efficiency**: Only one model in memory
- **Batch Experiments**: `--all-lolo` for comprehensive LOLO experiments across all languages

**Iterative Training** is better when you need:
- **Sophisticated Merging**: Advanced merging strategies and weight calculations
- **Research Experiments**: Comparing different merging approaches
- **Incremental Training**: Building models step-by-step with intermediate checkpoints

### 🏆 Example Workflow

```bash
# 1. Train a super model on all languages
uv run merginguriel/run_multilang_training.py --super-model

# 2. Run comprehensive LOLO experiments (creates 49 models)
uv run merginguriel/run_multilang_training.py --all-lolo

# 3. Compare with specific LOLO models
uv run merginguriel/run_multilang_training.py --exclude en-US
uv run merginguriel/run_multilang_training.py --exclude fr-FR
uv run merginguriel/run_multilang_training.py --exclude sq-AL

# 4. Choose the best approach for your use case
```

The aggregation system transforms hundreds of individual experiment results into a unified, analyzable dataset perfect for research presentations! 📊