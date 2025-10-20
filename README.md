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

The aggregation system transforms hundreds of individual experiment results into a unified, analyzable dataset perfect for research presentations! 📊