---
name: task-vector-base-model
description: >
  Select the correct pretrained base model for task-vector merging methods.
  Use when: running TIES/task_arithmetic/AdaMerging with IncTar mode.
metadata:
  short-description: "Task vectors must be computed vs pretrained base (esp. IncTar)"
  tags:
    - cross-lingual
    - model-merging
    - task-vector
    - base-model
    - inctar
  domain: research
  created: 2026-01-20
  author: MergingUriel
---

# Task Vector Base Model Selection

## Description

Task-vector methods (TIES, task_arithmetic, AdaMerging) compute deltas between fine-tuned models and a base model. The choice of base model is critical, especially for IncTar (Include Target) mode.

---

## The Problem

### Incorrect Approach (Causes IncTar Failure)

Using the first source model as the base:

```python
# WRONG for IncTar
task_vector = finetuned_model - first_source_model
```

**Why it fails:**
- For ExcTar: Works because all models are sources with same structure
- For IncTar: Target model's "delta" relative to a source is meaningless

**Evidence:**
| Locale | IncTar (wrong base) | IncTar (correct base) |
|--------|--------------------|-----------------------|
| sw-KE | 41.90% | 82.75% |
| sq-AL | 64.09% | 86.38% |

### Correct Approach

Use the actual pretrained base model:

```python
# CORRECT
task_vector = finetuned_model - pretrained_base_model
```

---

## When to Apply

**Always** when using task-vector methods with IncTar mode:
- TIES + IncTar
- task_arithmetic + IncTar
- AdaMerging + IncTar

For ExcTar-only experiments, using first source as base is acceptable (though not ideal).

---

## Implementation

### The Challenge

Pretrained `xlm-roberta-base` has a 2-class classifier, but fine-tuned models have 60 classes (MASSIVE intents). Dimensions don't match.

### The Solution

Create a pretrained base with correctly-sized classifier head:

```python
def _create_pretrained_base_for_task_vectors(self, reference_model_path: str) -> str:
    """Create pretrained base with classifier matching fine-tuned models."""
    from transformers import AutoModelForSequenceClassification, AutoConfig
    import tempfile

    # Get num_labels from a fine-tuned model
    ref_config = AutoConfig.from_pretrained(reference_model_path)
    num_labels = ref_config.num_labels  # e.g., 60 for MASSIVE

    # Load pretrained with correct classifier size (randomly initialized)
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Critical: allows size mismatch
    )

    # Save to temp directory
    temp_dir = tempfile.mkdtemp(prefix="pretrained_base_")
    model.save_pretrained(temp_dir)

    return temp_dir
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_labels` | Match fine-tuned models (e.g., 60) | Correct classifier dimensions |
| `ignore_mismatched_sizes` | `True` | Allow loading despite head mismatch |

---

## Verification

After applying this fix, IncTar models should:

1. **Match target-only performance** (within ~1%)
2. **Learn coefficients that heavily favor target** (~98% weight)

If IncTar accuracy is dramatically lower than target-only (e.g., 40% vs 85%), the base model is likely wrong.

---

## Code Location

Implementation in: `merginguriel/run_merging_pipeline_refactored.py`
- Method: `_create_pretrained_base_for_task_vectors()`
- Called from: `_perform_standard_merge()` for task-vector methods

---

## Related Skills

- `adamerging-cross-lingual`: When to use AdaMerging
- `source-selection-cross-lingual`: REAL vs URIEL similarity
