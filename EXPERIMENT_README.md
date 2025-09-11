# Large-Scale Model Merging Experiments

This directory contains scripts for running large-scale model merging experiments across all locales from the MASSIVE dataset.

## Scripts Overview

### 1. `run_large_scale_experiment.py`
Main script to run complete experiments for multiple locales.

**Usage:**
```bash
# Run experiments for all locales
python run_large_scale_experiment.py

# Run for specific locales only
python run_large_scale_experiment.py --locales sq-AL cy-GB ja-JP

# Skip certain experiment types
python run_large_scale_experiment.py --skip-baseline --skip-similarity

# Resume from a specific locale (useful for interrupted runs)
python run_large_scale_experiment.py --start-from 10

# Limit number of locales to process
python run_large_scale_experiment.py --max-locales 5

# List all available locales
python run_large_scale_experiment.py --list-locales
```

**What it does:**
1. For each locale:
   - **Baseline**: Evaluates the original fine-tuned model (`haryoaw/xlm-roberta-base_massive_k_{locale}` with `alpha_0.5_{locale}_epoch-9`)
   - **Similarity**: Runs similarity-based merging and evaluates the result
   - **Average**: Runs average-based merging and evaluates the result
2. Saves progress to `experiment_progress.json`
3. Saves final results to `experiment_final_results.json`

### 2. `aggregate_results.py`
Script to aggregate and compare results from all experiments.

**Usage:**
```bash
# Aggregate all results
python aggregate_results.py

# Show missing/failed experiments
python aggregate_results.py --show-missing

# Generate specific output formats
python aggregate_results.py --format csv
python aggregate_results.py --format markdown
```

**What it does:**
1. Scans all result folders in `results/`
2. Extracts accuracy scores and metadata
3. Creates comparison tables showing baseline vs. similarity vs. average results
4. Calculates improvements over baseline
5. Generates summary statistics
6. Saves results in multiple formats (CSV, JSON, Markdown)

### 3. `run_merging_pipeline.py` (Modified)
Original merging script with standardized output paths.

**Key Changes:**
- Output directories now use predictable names: `merged_models/{mode}_merge_{target_lang}`
- No timestamp in folder names for easier automation

### 4. `evaluate_specific_model.py` (Modified)
Original evaluation script with structured output paths.

**Key Changes:**
- Added `--prefix` argument for experiment type identification
- Structured folder names: `results/{prefix}_{locale}` or `results/{base_name}_{model_dir}_{locale}`
- Better automation support for result collection

## Experiment Workflow

### For a single locale (e.g., sq-AL):
```bash
# 1. Baseline evaluation
uv run evaluate_specific_model.py --base-model haryoaw/xlm-roberta-base_massive_k_sq-AL --model-dir alpha_0.5_sq-AL_epoch-9 --locale sq-AL --prefix baseline

# 2. Similarity merge and evaluation
python run_merging_pipeline.py --mode similarity --target-lang sq-AL
uv run evaluate_specific_model.py --base-model merged_models/similarity_merge_sq-AL --locale sq-AL --prefix similarity

# 3. Average merge and evaluation
python run_merging_pipeline.py --mode average --target-lang sq-AL
uv run evaluate_specific_model.py --base-model merged_models/average_merge_sq-AL --locale sq-AL --prefix average
```

### For all locales:
```bash
# Run everything
python run_large_scale_experiment.py

# Then aggregate results
python aggregate_results.py
```

## Output Structure

### Results Directory
```
results/
├── baseline_sq-AL/
│   └── results.json
├── similarity_sq-AL/
│   └── results.json
├── average_sq-AL/
│   └── results.json
└── ... (other locales)
```

### Merged Models Directory
```
merged_models/
├── similarity_merge_sq-AL/
├── average_merge_sq-AL/
└── ... (other locales)
```

### Aggregated Results
```
results_aggregated_YYYYMMDD_HHMMSS.csv          # Raw data
results_comparison_YYYYMMDD_HHMMSS.csv          # Comparison table
results_summary_YYYYMMDD_HHMMSS.json            # Statistics
results_report_YYYYMMDD_HHMMSS.md               # Markdown report
```

## Expected Experiment Count

- **Total locales**: 52 (excluding 'unknown')
- **Experiments per locale**: 3 (baseline, similarity, average)
- **Total experiments**: 156

## Resuming Interrupted Runs

If the experiment is interrupted:
1. Check `experiment_progress.json` to see how far it got
2. Resume with `--start-from N` where N is the locale index to continue from
3. Use `--max-locales` to limit the remaining number

## Example Commands

### Quick test with 2 locales:
```bash
python run_large_scale_experiment.py --locales sq-AL cy-GB --max-locales 2
```

### Only run similarity merging for all locales:
```bash
python run_large_scale_experiment.py --skip-baseline --skip-average
```

### Resume from locale 10 (ja-JP):
```bash
python run_large_scale_experiment.py --start-from 10
```

## Monitoring Progress

During execution:
- Progress is saved to `experiment_progress.json` after each locale
- Console output shows which locale is being processed
- Summary statistics are printed at the end

After completion:
- Check `experiment_final_results.json` for detailed results
- Run `python aggregate_results.py` to generate comparison reports
- Check the generated Markdown report for easy viewing