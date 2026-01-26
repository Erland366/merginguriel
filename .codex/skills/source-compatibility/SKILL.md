---
name: source-compatibility
description: >
  Analyze pairwise compatibility of source models before merging using Task Vector Cosine similarity.
  Use when: selecting sources for cross-lingual model merging to improve merge quality.
metadata:
  short-description: "Task Vector Cosine predicts merge compatibility"
  tags:
    - model-merging
    - cross-lingual
    - task-vectors
    - compatibility
  domain: research
  created: 2026-01-21
  author: Claude
---

# Source Compatibility Analysis

## General Description

When merging multiple source models, not all good sources combine well together. Two models might each transfer well individually but interfere when merged. This skill captures how to analyze **pairwise source compatibility** using Task Vector Cosine similarity to improve merge quality.

**Key finding**: Task Vector Cosine (parameter-space) provides a weak but consistent positive signal (~0.3% improvement). CKA (representation-space) provides no benefit.

## When to Apply

Use this knowledge when:
- Planning cross-lingual model merging experiments
- Marginal accuracy gains matter (e.g., production deployments)
- You have pre-computed compatibility matrices available
- You want a lightweight compatibility signal without running full merges

Do NOT use when:
- You need to flip a failing merge to success (signal too weak)
- You're optimizing for speed (adds matrix lookup overhead)

## Results Summary

| Metric | cy-GB Δ | sw-KE Δ | Recommendation |
|--------|---------|---------|----------------|
| Task Vector Cosine × Similarity | **+0.24%** | **+0.37%** | Use as default |
| CKA × Similarity | -0.06% | +0.07% | Skip |
| Baseline Similarity | 0% | 0% | — |

**Matrix statistics** (48 locales):
- TV Cosine: mean=0.072, std=0.041, range=[0.00, 0.33]
- CKA: mean=0.649, std=0.308, range=[0.00, 0.89]

## Recommended Practice

### Integration Method

Use **multiplicative weighting**:

```
final_weight[i] = similarity_weight[i] × avg_compatibility[i]
```

Where `avg_compatibility[i]` = average pairwise TV Cosine with other selected sources.

### Configuration

```yaml
method: "similarity_x_tv_compatibility"
similarity_type: "REAL"  # or URIEL
num_languages: 5
include_target: false
```

### Pre-computing Matrices

Run once per model family:

```bash
python -m merginguriel.compute_compatibility_matrix \
  --models-dir haryos_model \
  --pretrained xlm-roberta-base \
  --metric both \
  --output-dir nxn_results/compatibility_matrix
```

Outputs:
- `task_vector_cosine_matrix.csv`
- `cka_matrix.csv`
- `computation_log.json`

### Using Pre-computed Matrices

The `CompatibilityWeightCalculator` automatically loads from default paths:
- TV Cosine: `nxn_results/compatibility_matrix/task_vector_cosine_matrix.csv`
- CKA: `nxn_results/compatibility_matrix/cka_matrix.csv`

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| CKA compatibility | Representation similarity ≠ structural compatibility | Use parameter-space metrics |
| Expecting large gains | TV Cosine is a weak signal (~0.3%) | Use as tiebreaker, not primary |
| cy-GB still failing | Compatibility can't fix fundamental interference | Don't rely on compatibility alone |

## Algorithm Details

### Task Vector Cosine

```python
def compute_task_vector_cosine(model_a, model_b, pretrained):
    # Compute task vectors (exclude classifier head)
    tv_a = flatten(model_a.params - pretrained.params)
    tv_b = flatten(model_b.params - pretrained.params)

    # Cosine similarity
    return dot(tv_a, tv_b) / (norm(tv_a) * norm(tv_b))
```

### Compatibility Score

```python
def compute_compatibility_score(source, other_sources, matrix):
    # Average pairwise compatibility with other selected sources
    scores = [matrix[source][other] for other in other_sources]
    return mean(scores)
```

### Final Weight

```python
def compute_final_weight(similarity_weight, compatibility_score):
    return similarity_weight * compatibility_score
    # Then renormalize all weights to sum to 1
```

## Implementation Files

| File | Purpose |
|------|---------|
| `merginguriel/compatibility.py` | Core compatibility metrics |
| `merginguriel/compute_compatibility_matrix.py` | Pre-computation CLI |
| `merginguriel/run_merging_pipeline_refactored.py` | `CompatibilityWeightCalculator` |
| `configs/ablations/source_compatibility.yaml` | Ablation config |

## Why Parameter-Space Beats Representation-Space

1. **Direct interference measurement**: Task vectors represent actual weight changes. When merged, similar vectors reinforce; orthogonal vectors cancel.

2. **Structural vs behavioral**: CKA measures whether models produce similar outputs, not whether their internal structures are compatible for averaging.

3. **Computational efficiency**: TV Cosine requires only parameter comparison (fast). CKA requires forward passes on shared inputs (slow).

## References

- Ablation database: `experiments_compatibility.db`
- Pre-computed matrices: `nxn_results/compatibility_matrix/`
- Related skill: `merging-when-constructive`
