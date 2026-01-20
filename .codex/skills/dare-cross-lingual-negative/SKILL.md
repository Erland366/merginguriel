---
name: dare-cross-lingual-negative
description: >
  DARE preprocessing hurts cross-lingual model merging. Negative result skill.
  Use when: considering DARE for cross-lingual transfer experiments.
metadata:
  short-description: "DARE destroys shared multilingual representations"
  tags:
    - cross-lingual
    - model-merging
    - dare
    - negative-result
  domain: research
  created: 2026-01-20
  author: MergingUriel
---

# DARE is Harmful for Cross-Lingual Merging

## General Description

DARE (Drop And REscale) randomly drops delta parameters before merging to reduce interference. While effective for same-language multi-task merging, DARE catastrophically hurts cross-lingual transfer. This skill documents the negative result to prevent repeated experimentation.

## When to Apply

Use this knowledge when:
- Considering DARE preprocessing for cross-lingual model merging
- Reading papers about DARE and wondering if it applies to your multilingual setup
- Debugging why merged multilingual models have poor performance

Do NOT use when:
- Merging models fine-tuned on different tasks in the SAME language
- Working with English-only or monolingual models

## Results Summary

| Locale | Without DARE | With DARE (0.9) | Delta | Relative Drop |
|--------|--------------|-----------------|-------|---------------|
| sq-AL | 66.54% | 24.21% | -42.33% | -63.6% |
| sw-KE | 42.74% | 35.71% | -7.03% | -16.4% |
| vi-VN | 74.45% | 32.08% | -42.37% | -56.9% |

**Consistent finding**: DARE hurts performance across all tested locales.

## Recommended Practice

### Step 1: Do NOT use DARE for cross-lingual merging

```bash
# WRONG - will destroy performance
python -m merginguriel.run_merging_pipeline_refactored \
  --target-locale sq-AL \
  --mode ties \
  --dare-drop-rate 0.9  # DO NOT USE

# CORRECT - no DARE
python -m merginguriel.run_merging_pipeline_refactored \
  --target-locale sq-AL \
  --mode ties \
  --dare-drop-rate 0.0  # default, no DARE
```

### Step 2: Understand why DARE fails here

**Original DARE context**: Same-language, different-task merging
- Task vectors represent task-specific knowledge
- Random dropping reduces task interference
- Rescaling preserves expected magnitude

**Cross-lingual context**: Different-language, same-task merging
- Task vectors encode shared multilingual representations
- Random dropping destroys cross-lingual alignment
- XLM-RoBERTa's multilingual capacity relies on dense representations

**Key insight**: Cross-lingual transfer depends on shared subspaces that get randomly corrupted by DARE's dropout mechanism.

### Step 3: If you must experiment with sparsification

Consider these alternatives (not yet tested):
- **Magnitude-based pruning**: Keep largest deltas instead of random
- **Layer-selective DARE**: Only apply to task-specific layers, not encoder
- **Lower drop rates**: Try 0.1-0.3 instead of 0.9

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| DARE drop_rate=0.9 | Destroys shared multilingual representations | Do not use DARE for cross-lingual |
| STS-B proxy evaluation | Showed minor impact (-0.5%) vs actual MASSIVE (-63%) | Always evaluate on actual task |
| Assuming DARE generalizes | Original paper was same-language multi-task | Cross-lingual is fundamentally different |

## Configuration

```yaml
# Cross-lingual merging - NO DARE
dare_drop_rate: 0.0  # disabled

# If experimenting with alternatives (untested):
# magnitude_pruning: true
# pruning_threshold: 0.1
```

## References

- DARE paper: `compiled_resources/MergingUriel/2311_03099_full.md`
- Original finding: DARE effective for same-language multi-task (Table 1)
- This finding: DARE harmful for cross-lingual same-task
- Related skill: `source-selection-cross-lingual`
