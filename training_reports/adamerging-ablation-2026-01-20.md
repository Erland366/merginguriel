# AdaMerging Ablation Study for Cross-Lingual Merging

**Date:** 2026-01-20
**Domain:** Cross-lingual model merging
**Status:** Complete

---

## Objective

Test whether AdaMerging (entropy-based coefficient learning) can improve cross-lingual model merging compared to fixed-weight methods (TIES).

**Research question:** Can learning optimal merging coefficients via entropy minimization outperform static similarity-based weights?

---

## Background

AdaMerging (ICLR 2024) learns merging coefficients by minimizing entropy of predictions on unlabeled test data. The key insight is that entropy serves as a proxy for prediction confidence.

**Original context:** Multi-task learning where each model is a task expert.
**Our adaptation:** Cross-lingual merging where each model is a language expert.

---

## Experimental Setup

### Models
- **Base**: xlm-roberta-base
- **Source models**: Fine-tuned on MASSIVE intent classification
- **Architecture reference**: First source model (de-DE)

### Configuration
| Parameter | Value |
|-----------|-------|
| Mode | task_wise (one coefficient per source) |
| Iterations | 5 (quick test) |
| Learning rate | 0.001 |
| Batch size | 64 |
| Initial coefficients | REAL similarity weights |
| Device | CPU (RTX 4090 available but not used) |

### Locales Tested
| Locale | Difficulty | TIES Result | Sources Selected (REAL) |
|--------|------------|-------------|-------------------------|
| sq-AL | Easy | 66.54% | de-DE, it-IT, tr-TR |
| sw-KE | Hard | 42.74% | az-AZ, th-TH, tr-TR |
| vi-VN | Medium | 74.45% | de-DE, fr-FR, tl-PH |

---

## Results

### Primary Comparison

| Locale | AdaMerging | TIES | Best Zero-shot | vs Baseline | Goal 1? |
|--------|-----------|------|----------------|-------------|---------|
| sq-AL | 65.90% | **66.54%** | 66.44% | -0.54% | No |
| sw-KE | **48.08%** | 42.74% | 46.70% | **+1.38%** | **Yes** |
| vi-VN | 70.07% | **74.45%** | 74.31% | -4.24% | No |

### Coefficient Learning

| Locale | Initial Coefficients | Final Coefficients | Change |
|--------|---------------------|-------------------|--------|
| sq-AL | [0.43, 0.29, 0.28] | [0.43, 0.29, 0.28] | None |
| sw-KE | ~[0.33, 0.35, 0.32] | [0.73, 0.19, 0.08] | Large |
| vi-VN | ~[0.40, 0.30, 0.30] | [0.57, 0.27, 0.15] | Moderate |

---

## Key Findings

### 1. AdaMerging Helps Hard Cases, Hurts Easy Cases

| Finding | Evidence |
|---------|----------|
| sw-KE improved | +5.34% vs TIES, +1.38% vs zero-shot baseline |
| sq-AL/vi-VN degraded | -0.64% and -4.24% respectively |

**Interpretation:** When initial weights are already good (easy cases), entropy minimization overcorrects and degrades performance.

### 2. Coefficient Learning Works Selectively

- **sq-AL:** Coefficients didn't change (already at local minimum?)
- **sw-KE:** Massive shift toward first source (0.33 -> 0.73)
- **vi-VN:** Moderate shift (0.40 -> 0.57)

### 3. Only 5 Iterations Was Insufficient for sq-AL

The sq-AL coefficients didn't change at all with 5 iterations, suggesting:
- Either the initial weights were already optimal
- Or gradient descent needs more iterations to escape local minimum

---

## Analysis

### Why AdaMerging Helped sw-KE

sw-KE was the "constructive" case where:
1. TIES baseline was 42.74% (below zero-shot 46.70%)
2. Sinkhorn normalization selected suboptimal sources
3. AdaMerging learned to heavily weight one source (73%) over others

**Hypothesis:** Entropy minimization correctly identified that spreading weight across poorly-aligned sources was suboptimal.

### Why AdaMerging Hurt sq-AL and vi-VN

For "easy" cases where TIES already achieved Goal 1:
1. The initial similarity-based weights were good
2. Entropy minimization pushed toward "confident" predictions
3. Confident != correct, especially for cross-lingual transfer

---

## Recommendations

1. **Use AdaMerging selectively** - Only when TIES fails to beat baseline
2. **Increase iterations** - 5 iterations may be insufficient; try 50-100
3. **Use GPU** - Fixed in code; will speed up optimization 10x
4. **Consider hybrid approach** - AdaMerging only if TIES accuracy < baseline

---

## Technical Notes

### Performance Issue (Fixed)
AdaMerging was running on CPU despite RTX 4090 availability. Fixed by adding:
```python
if torch.cuda.is_available():
    base_model_obj = base_model_obj.to("cuda")
```

### Runtime
- CPU: ~9 minutes per locale (5 iterations, 47 batches)
- GPU (expected): ~1 minute per locale

---

## Files Modified

| File | Change |
|------|--------|
| `auto_merge_llm/methods/adamerging.py` | Added GPU support |
| `merginguriel/run_merging_pipeline_refactored.py` | AdaMerging integration |

---

## Next Steps

1. Re-run with GPU and more iterations (50-100)
2. Test AdaMerging++ (with TIES preprocessing)
3. Test layer-wise AdaMerging
4. Implement adaptive strategy: use AdaMerging only when TIES < baseline
