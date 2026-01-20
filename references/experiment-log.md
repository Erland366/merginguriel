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

## 2026-01-20 – Retrospective: DARE & Source Selection Ablation

**Type:** Retrospective
**General description:** Tested DARE preprocessing and REAL vs URIEL similarity on cross-lingual merging; DARE harmful, REAL essential.

### What we tried

1. **URIEL vs REAL similarity for source selection**
   - sq-AL with URIEL: selected ar-SA, cy-GB, fi-FI
   - sq-AL with REAL: selected de-DE (43%), it-IT (29%), tr-TR (28%)

2. **DARE preprocessing (drop_rate=0.9)**
   - Tested with TIES merging on sq-AL, sw-KE, vi-VN
   - Both with and without DARE

3. **Locale diversity**
   - sq-AL: neutral case
   - sw-KE: "constructive" (56% diagonal ratio)
   - vi-VN: "destructive" (86% diagonal ratio)

### Key findings

| Finding | Evidence |
|---------|----------|
| URIEL = catastrophic | sq-AL: 11.63% with URIEL vs 66.54% with REAL |
| REAL achieves Goal 1 | sq-AL: 66.54% > 66.44% baseline; vi-VN: 74.45% > 74.31% baseline |
| DARE destroys performance | sq-AL: 66.54% → 24.21%; vi-VN: 74.45% → 32.08% |
| Sinkhorn suboptimal | sw-KE picked az-AZ (41.5%) instead of sq-AL (46.7%) |

### What failed

1. **DARE for cross-lingual**: -63% relative performance drop on sq-AL
2. **sw-KE "constructive" case**: Failed despite favorable diagonal ratio
3. **STS-B as proxy**: Showed minimal DARE impact vs catastrophic on actual task

### Outcome

- Goal 1 (ExcTar) achieved for 2/3 locales (sq-AL, vi-VN)
- Created skills: `source-selection-cross-lingual`, `dare-cross-lingual-negative`
- Next: Fix Sinkhorn → top-k selection; test IncTar experiments

---

## 2026-01-20 – Retrospective: AdaMerging for Cross-Lingual Merging

**Type:** Retrospective
**General description:** Tested AdaMerging (entropy-based coefficient learning) for cross-lingual merging; helps hard cases, hurts easy cases.

### What we tried

1. **AdaMerging integration**
   - Implemented task-wise AdaMerging in pipeline
   - Used REAL similarity weights as initial coefficients
   - Optimized via entropy minimization on target language data

2. **Three locales tested**
   - sq-AL (easy): TIES baseline 66.54%
   - sw-KE (hard): TIES baseline 42.74%
   - vi-VN (medium): TIES baseline 74.45%

### Key findings

| Finding | Evidence |
|---------|----------|
| AdaMerging helps hard cases | sw-KE: 48.08% vs 42.74% TIES (+5.34%) |
| AdaMerging hurts easy cases | sq-AL: 65.90% vs 66.54%, vi-VN: 70.07% vs 74.45% |
| Coefficients learn selectively | sw-KE shifted [0.33→0.73], sq-AL unchanged |
| Goal 1 achieved for sw-KE only | 48.08% > 46.70% baseline |

### What failed

1. **sq-AL and vi-VN degraded**: Entropy minimization overcorrected
2. **Only 5 iterations**: May be insufficient for proper convergence
3. **CPU execution**: RTX 4090 available but not used (fixed)

### Outcome

- AdaMerging is situational: use only when TIES fails
- Fixed GPU support in adamerging.py
- Created report: `training_reports/adamerging-ablation-2026-01-20.md`
- Next: Test with more iterations on GPU; try AdaMerging++

---

<!-- New entries go above this line -->
