---
name: source-selection-cross-lingual
description: >
  Guidance for selecting source language models in cross-lingual merging.
  Use when: choosing which fine-tuned models to merge for a target language.
metadata:
  short-description: "REAL > URIEL for cross-lingual source selection"
  tags:
    - cross-lingual
    - model-merging
    - source-selection
  domain: research
  created: 2026-01-20
  author: MergingUriel
---

# Source Selection for Cross-Lingual Merging

## General Description

Selecting which source language models to merge is the most critical decision in cross-lingual model merging. Poor source selection leads to catastrophic performance (11% vs 66% accuracy difference observed). This skill captures findings on empirical vs linguistic similarity for source selection.

## When to Apply

Use this knowledge when:
- Selecting source models for cross-lingual merging experiments
- Choosing between URIEL (linguistic) and REAL (empirical) similarity
- Deciding how to weight multiple source models

Do NOT use when:
- Merging models for the same language (different tasks)
- Working with monolingual models only

## Results Summary

| Similarity Type | sq-AL Accuracy | Sources Selected | Notes |
|-----------------|----------------|------------------|-------|
| URIEL (linguistic) | 11.63% | ar-SA, cy-GB, fi-FI | Catastrophic failure |
| REAL (empirical) | 66.54% | de-DE, it-IT, tr-TR | Beats baseline (66.44%) |

| Locale | Best Empirical Sources | REAL Selected | Gap |
|--------|------------------------|---------------|-----|
| sw-KE | sq-AL (46.7%), en-US (46.7%) | az-AZ (41.5%), th-TH (39.1%) | -5% transfer loss |

## Recommended Practice

### Step 1: Use REAL similarity, not URIEL

URIEL measures linguistic feature similarity but this doesn't predict transfer performance. REAL similarity uses actual NxN evaluation matrix scores.

```bash
python -m merginguriel.run_merging_pipeline_refactored \
  --target-locale sq-AL \
  --similarity-type REAL \  # NOT URIEL
  --mode ties \
  --num-languages 3
```

### Step 2: Consider top-k selection over Sinkhorn

Current implementation uses Sinkhorn normalization which optimizes for "balanced similarity distribution" rather than picking the best sources.

**Problem observed**: For sw-KE, Sinkhorn picked az-AZ (41.5% transfer) instead of sq-AL (46.7% transfer).

**Proposed fix**: Implement simple top-k selection:
```python
# Instead of Sinkhorn normalization
top_k_indices = similarity_scores.argsort()[-k:]
weights = torch.ones(k) / k  # or proportional to similarity
```

### Step 3: Validate with NxN matrix

Before running expensive merging experiments, check the NxN evaluation matrix:

```
nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv
```

- Row = source model
- Column = target language
- Value = accuracy

Best sources for target X = highest values in column X.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| URIEL similarity | Linguistic features ≠ transfer performance | Always use empirical (REAL) similarity |
| Sinkhorn normalization | Optimizes distribution, not best transfer | Consider top-k selection instead |
| Trusting "constructive" label | sw-KE labeled constructive but failed | Verify actual source selection matches best empirical sources |

## Configuration

```yaml
# Recommended settings for source selection
similarity_type: REAL  # empirical transfer scores
num_languages: 3       # top 3 sources
# TODO: Implement top-k selection instead of Sinkhorn
```

## References

- Related reports: `training_reports/dare-ablation-2026-01-20.md` (to be created)
- NxN matrix: `nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv`
- Pipeline: `merginguriel/run_merging_pipeline_refactored.py`
