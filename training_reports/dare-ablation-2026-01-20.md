# DARE & Source Selection Ablation Study

**Date:** 2026-01-20
**Domain:** Cross-lingual model merging
**Status:** Complete

---

## Objective

Test whether DARE preprocessing and REAL similarity can help cross-lingual model merging achieve:
- **Goal 1 (ExcTar)**: merged_accuracy > max(source_accuracies_on_target)
- **Goal 2 (IncTar)**: merged_accuracy > target_model_on_target

---

## Experimental Setup

### Models
- **Base**: xlm-roberta-base
- **Source models**: Fine-tuned on MASSIVE intent classification (various locales)
- **Merge method**: TIES (k=0.2)

### Configurations Tested

| Config | Similarity | DARE | Locales |
|--------|------------|------|---------|
| A | URIEL | No | sq-AL |
| B | REAL | No | sq-AL, sw-KE, vi-VN |
| C | REAL | 0.9 | sq-AL, sw-KE, vi-VN |

### Evaluation
- **Task**: MASSIVE intent classification (60 classes)
- **Metric**: Accuracy
- **Script**: `python -m merginguriel.evaluate_specific_model`

---

## Results

### Primary Comparison: URIEL vs REAL (sq-AL)

| Similarity | Sources Selected | Accuracy | vs Baseline |
|------------|------------------|----------|-------------|
| URIEL | ar-SA, cy-GB, fi-FI | 11.63% | -54.81% |
| REAL | de-DE, it-IT, tr-TR | 66.54% | +0.10% |
| Baseline | fr-FR (best zero-shot) | 66.44% | - |

**Finding**: URIEL similarity selects linguistically similar but empirically poor sources. REAL similarity is essential.

### DARE Ablation (REAL similarity)

| Locale | Baseline | No DARE | With DARE | Goal 1? |
|--------|----------|---------|-----------|---------|
| sq-AL | 66.44% | **66.54%** | 24.21% | No DARE: Yes |
| sw-KE | 46.70% | 42.74% | 35.71% | No |
| vi-VN | 74.31% | **74.45%** | 32.08% | No DARE: Yes |

**Finding**: DARE consistently destroys performance (-16% to -63% relative drop).

### Source Selection Analysis (sw-KE failure)

| Source | Empirical Transfer | Selected by REAL? |
|--------|-------------------|-------------------|
| sq-AL | 46.70% | No |
| en-US | 46.70% | No |
| tl-PH | 46.43% | No |
| az-AZ | 41.52% | Yes (32.7%) |
| th-TH | 39.13% | Yes (35.3%) |
| tr-TR | 37.56% | Yes (32.0%) |

**Finding**: Sinkhorn normalization optimizes for "balanced distribution" rather than selecting best sources.

---

## Key Insights

### 1. Source Selection > Merge Algorithm
Wrong sources = catastrophic failure. URIEL gave 11.63%, REAL gave 66.54% on same merge method.

### 2. DARE Doesn't Generalize to Cross-Lingual
DARE works for same-language multi-task merging (original paper). Cross-lingual transfer relies on shared representations that get destroyed by random dropout.

### 3. Sinkhorn Normalization is Suboptimal
For sw-KE, best empirical sources were sq-AL/en-US (46.7%), but Sinkhorn picked az-AZ/th-TH (~40%).

### 4. Goal 1 Achieved for 2/3 Locales
- sq-AL: 66.54% > 66.44% (+0.15%)
- vi-VN: 74.45% > 74.31% (+0.19%)
- sw-KE: 42.74% < 46.70% (-8.5%)

---

## Recommendations

1. **Always use REAL similarity** for cross-lingual merging
2. **Never use DARE** for cross-lingual (drop_rate=0.0)
3. **Replace Sinkhorn** with top-k source selection
4. **Test IncTar** to see if including target model helps

---

## Files Modified/Created

| File | Type |
|------|------|
| `.codex/skills/source-selection-cross-lingual/SKILL.md` | New skill |
| `.codex/skills/dare-cross-lingual-negative/SKILL.md` | New skill |
| `references/experiment-log.md` | Updated |

---

## Next Steps

1. Implement top-k source selection (bypass Sinkhorn)
2. Run IncTar experiments (include target model in merge)
3. Test AdaMerging (entropy-based coefficient learning)
4. Broader locale validation
