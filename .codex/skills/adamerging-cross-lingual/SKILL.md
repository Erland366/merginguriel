---
name: adamerging-cross-lingual
description: >
  Guidance for when and how to use AdaMerging for cross-lingual model merging.
  Use when: TIES underperforms (ExcTar) or to preserve/enhance target (IncTar).
metadata:
  short-description: "Use AdaMerging selectively; GPU + correct base model are critical"
  tags:
    - cross-lingual
    - model-merging
    - adamerging
    - task-vector
    - entropy-minimization
  domain: research
  created: 2026-01-20
  author: MergingUriel
---

# AdaMerging for Cross-Lingual Model Merging

## Description

AdaMerging (ICLR 2024) learns optimal merging coefficients by minimizing entropy of predictions on unlabeled test data. This skill documents when and how to use AdaMerging for cross-lingual merging.

---

## When to Apply

### ExcTar Mode (Exclude Target)

| Scenario | Use AdaMerging? | Rationale |
|----------|-----------------|-----------|
| TIES fails to beat baseline | **Yes** | AdaMerging can learn better weights |
| TIES already beats baseline | **No** | AdaMerging may overcorrect and degrade |

**Rule of thumb:** Only use AdaMerging for ExcTar when `TIES_accuracy < best_zero_shot_accuracy`.

### IncTar Mode (Include Target)

| Scenario | Use AdaMerging? | Rationale |
|----------|-----------------|-----------|
| Want to enhance target model | **Yes** | AdaMerging learns to preserve target (~98% weight) |

---

## Key Implementation Details

### 1. GPU Support (Critical)

AdaMerging optimization is compute-intensive. Always use GPU:

```python
# In _optimize_coefficients method
if torch.cuda.is_available():
    base_model_obj = base_model_obj.to("cuda")
```

**Impact:** 9 min → 1 min per locale

### 2. Proper Base Model for Task Vectors

See skill: `task-vector-base-model`

For IncTar mode, must use pretrained base (not source model) for task vector computation.

### 3. Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `adamerging_mode` | `task_wise` | One coefficient per source model |
| `adamerging_iterations` | 50-100 | 5 iterations insufficient for some locales |
| `adamerging_lr` | 0.001 | Default works well |
| `initial_coefficients` | REAL similarity weights | Better starting point than uniform |

---

## Experimental Evidence

### ExcTar Results

| Locale | AdaMerging | TIES | Verdict |
|--------|-----------|------|---------|
| sq-AL (easy) | 65.90% | **66.54%** | TIES wins |
| sw-KE (hard) | **48.08%** | 42.74% | AdaMerging wins |
| vi-VN (medium) | 70.07% | **74.45%** | TIES wins |

### IncTar Results (with proper base model)

| Locale | AdaMerging IncTar | Target-only | Goal 2? |
|--------|------------------|-------------|---------|
| sq-AL | **86.38%** | 86.31% | Yes (+0.07%) |
| sw-KE | 82.75% | 82.85% | Almost (-0.10%) |

### Coefficient Learning Behavior

| Locale | Initial | Final | Interpretation |
|--------|---------|-------|----------------|
| sq-AL ExcTar | [0.43, 0.29, 0.28] | [0.43, 0.29, 0.28] | No change (already good) |
| sw-KE ExcTar | [0.33, 0.35, 0.32] | [0.73, 0.19, 0.08] | Concentrated on best source |
| sq-AL IncTar | - | [0.997, 0.002, 0.001] | Heavily favors target |

---

## Failure Modes

1. **Overcorrection on easy cases:** Entropy minimization finds "confident" predictions, not necessarily correct ones
2. **Insufficient iterations:** Coefficients may not converge with <10 iterations
3. **Wrong base model:** IncTar fails catastrophically without proper pretrained base (see `task-vector-base-model` skill)

---

## Code Location

- AdaMerging method: `auto_merge_llm/methods/adamerging.py`
- Pipeline integration: `merginguriel/run_merging_pipeline_refactored.py`

---

## References

- Paper: "AdaMerging: Adaptive Model Merging for Multi-Task Learning" (ICLR 2024)
- arXiv: https://arxiv.org/abs/2310.02575
