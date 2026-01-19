# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-01-19 – Retrospective: Code Simplification

**Type:** Retrospective
**General description:** Migrated MergingUriel from CLI arguments to YAML config system and refactored large modules into packages.

### What we tried

1. **YAML Configuration System**
   - Created `merginguriel/config/` package with hierarchical dataclass configs
   - Implemented `TrackProvidedArgsAction` for CLI argument tracking
   - Added `--config` support to all 4 entry points
   - Built deprecation warning system for CLI-to-config migration

2. **Large Module Refactoring**
   - `plot_results.py`: 2,398 → 853 lines (extracted to `plotting/` package)
   - `aggregate_results.py`: 1,160 → 226 lines (extracted to `aggregation/` package)
   - `naming_config.py`: 526 → 381 lines (dropped legacy patterns)

### Key findings

- Dataclass-based configs provide IDE autocomplete and type hints
- `TrackProvidedArgsAction` cleanly detects explicitly-provided CLI args
- Dropping legacy naming patterns (15 → 3) significantly simplified maintenance
- Verified identical results between original and refactored code (Spearman 0.5779)

### What failed

- Monkeypatching lazy imports: patch at definition site, not usage site
- Hyphen in dataclass field name caused syntax error
- `--resume` flag comparison gave different results due to cached models

### Outcome

- Net reduction of ~3,300 lines through modularization
- 110 tests passing, 38 new config tests
- Created skill: `python-config-refactoring`

---

## 2026-01-19 – Observation: Research Goals and Evaluation Framework

**Type:** Observation
**General description:** Clarified MergingUriel's research objectives and the correct evaluation metrics.

### Research Goals

MergingUriel's primary objectives:

1. **Beat zero-shot cross-lingual performance**: A merged model should outperform individual source models evaluated directly on the target language (zero-shot transfer).

2. **Significantly improve over baseline**: The merged model should substantially beat the pretrained XLM-RoBERTa without any fine-tuning.

### The Actual Task: MASSIVE Intent Classification

- **Dataset**: AmazonScience/massive (49 locales, 60 intent classes)
- **Task**: Sequence classification (intent prediction)
- **Metric**: Accuracy = correct_predictions / total_predictions
- **Evaluation script**: `evaluate_specific_model.py`

### Baselines to Compare Against

1. **Zero-shot baseline**: Source language model evaluated on target language test set
   - Example: `af-ZA` model tested on `th-TH` → 44% accuracy
   - Stored in `nxn_results/*/evaluation_matrix.csv` (49×49 cross-lingual matrix)

2. **Best source baseline**: Best-performing source model for target (from NxN matrix)
   - For target `th-TH`: find max across all source models

3. **Best overall baseline**: Highest zero-shot accuracy from any model (excluding self)

### Current Pipeline Issue

The `run_merging_pipeline_refactored.py` currently evaluates using `evaluate_base_encoder.py` which measures **STS-B Spearman correlation** (semantic similarity) — this is a proxy metric, NOT the actual task.

**Wrong metric**: STS-B Spearman correlation (~0.55-0.58)
**Correct metric**: MASSIVE intent classification accuracy (~0.40-0.90 depending on language pair)

### How to Properly Evaluate

```python
# After merging, evaluate on MASSIVE intent classification:
from merginguriel.evaluate_specific_model import evaluate_specific_model

results = evaluate_specific_model(
    model_name="path/to/merged_model",
    locale="th-TH",  # Target locale
    eval_folder="results/merging_similarity_URIEL_th-TH_xlm-roberta-base_3lang_ExcTar"
)

# Compare results["performance"]["accuracy"] against:
# 1. Zero-shot baseline from evaluation_matrix.csv
# 2. Best source model accuracy
# 3. Pretrained XLM-RoBERTa baseline
```

### Success Criteria

For merged model targeting `th-TH`:
- `merged_accuracy > avg(source_accuracies_on_th)` → beats average zero-shot
- `merged_accuracy > max(source_accuracies_on_th)` → beats best zero-shot
- `merged_accuracy > pretrained_baseline` → beats raw pretrained model

### Key Files

| File | Purpose |
|------|---------|
| `evaluate_specific_model.py` | Correct evaluation (MASSIVE intent classification) |
| `evaluate_base_encoder.py` | Proxy metric (STS-B, not recommended for final eval) |
| `nxn_results/*/evaluation_matrix.csv` | Pre-computed baselines (49×49 accuracy matrix) |
| `run_nxn_evaluation.py` | Generate cross-lingual evaluation matrix |

---

<!-- New entries go above this line -->
