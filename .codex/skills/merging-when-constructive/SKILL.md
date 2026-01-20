---
name: merging-when-constructive
description: >
  Guidance on when cross-lingual model merging improves over baselines.
  Use when: planning model merging experiments for cross-lingual NLU.
metadata:
  short-description: "Model merging only helps when zero-shot is poor"
  tags:
    - model-merging
    - cross-lingual
    - xlm-roberta
    - massive-dataset
  domain: research
  created: 2026-01-20
  author: Claude
---

# When Model Merging is Constructive

## General Description

Cross-lingual model merging does NOT universally improve performance. Our ablation study (Jan 2026) shows merging is constructive only when zero-shot transfer is already poor—meaning no single source model dominates. When good sources exist, merging dilutes their contribution through averaging.

This skill captures the key predictor: the **Zero-shot/Self ratio**. When this ratio is low (<60%), merging can combine partial knowledge from multiple mediocre sources. When high (>80%), merging just averages away the best features.

## When to Apply

**Use model merging when:**
- Zero-shot/Self accuracy ratio < 60%
- No single source achieves > 50% of target's self-accuracy
- Target language is low-resource with poor cross-lingual transfer
- All candidate source models perform similarly (low variance)

**Do NOT use merging when:**
- Best zero-shot source achieves > 70% accuracy on target
- Zero-shot/Self ratio > 80%
- One source clearly dominates others

## Results Summary

### Ablation Results (MASSIVE Intent Classification, 3 source languages)

| Target | ZS/Self Ratio | Best Zero-Shot | Best Merged | Δ vs Baseline | Effect |
|--------|--------------|----------------|-------------|---------------|--------|
| sw-KE | **56.4%** | 0.4670 | 0.5796 | **+11.3%** | CONSTRUCTIVE |
| sq-AL | 77.0% | 0.6644 | 0.5821 | -8.2% | destructive |
| vi-VN | 86.2% | 0.7431 | 0.5658 | -17.7% | destructive |
| th-TH | 86.9% | 0.7371 | N/A | — | (not tested ExcTar) |
| fi-FI | 88.7% | 0.7518 | N/A | — | (not tested ExcTar) |
| tr-TR | 85.5% | 0.7310 | N/A | — | (not tested ExcTar) |

### Merging Effect Analysis

| Scenario | Sources Individually | Merged Result | Effect |
|----------|---------------------|---------------|--------|
| sw-KE (URIEL) | 0.41, 0.42, 0.35 | **0.58** | Constructive (+0.16) |
| vi-VN (REAL) | 0.74, 0.70, 0.71 | 0.55 | Destructive (-0.19) |

## Recommended Practice

### Before Running Experiments

1. **Check NxN baseline matrix first**
   ```python
   nxn = pd.read_csv('nxn_results/*/evaluation_matrix.csv', index_col=0)
   target = 'th-TH'

   self_acc = nxn.loc[target, target]
   best_zs = nxn[target].drop(target).max()
   ratio = best_zs / self_acc

   if ratio > 0.7:
       print(f"Skip merging for {target}: ratio={ratio:.1%}, just use best source")
   else:
       print(f"Merging may help for {target}: ratio={ratio:.1%}")
   ```

2. **Use REAL similarity over URIEL for source selection**
   - URIEL picks linguistically similar but poorly performing sources
   - REAL picks empirically better performers

3. **For high-transfer targets, use single best source**
   - Simpler and often better than merging

### Source Selection Quality

| Method | sq-AL Sources | Their Ranks | Result |
|--------|---------------|-------------|--------|
| URIEL | ar-SA, cy-GB, fi-FI | 31, 43, 27 | Poor selection |
| REAL | de-DE, it-IT, tr-TR | 5, 10, 20 | Better selection |
| Optimal | fr-FR (rank 1) | 1 | Best single model |

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| URIEL source selection | Linguistic similarity ≠ cross-lingual transfer quality | Use empirical performance metrics (REAL or NxN) |
| Merging vi-VN sources | Sources individually good (0.74), merged dropped to 0.55 | Averaging dilutes good models—skip merge |
| IncTar to beat diagonal | Target model alone (0.83-0.86) always outperformed merged | Don't try to "enhance" already-good models via merging |
| Only 3 source languages | May be insufficient for constructive averaging effect | Test with 5-7 languages for poor-transfer targets |

## Configuration

### Ablation Runner Config (for constructive targets only)

```yaml
ablation:
  name: "constructive_targets_only"
  description: "Merge only for low zero-shot/self ratio targets"

  fixed:
    method: "similarity"
    num_languages: 5  # Try more sources for poor-transfer targets
    model_family: "xlm-roberta-base"
    # Only include targets with ZS/Self < 60%
    locales:
      - "sw-KE"  # 56.4% ratio - good candidate
      # Exclude: sq-AL (77%), vi-VN (86%), th-TH (87%), etc.

  sweep:
    similarity_type: ["REAL"]  # REAL outperforms URIEL
    include_target: [false]    # ExcTar for Goal 1
```

## Open Questions

1. Would selecting sources directly by NxN column (actual performance on target) outperform REAL similarity?
2. Does increasing to 5-7 source languages improve constructive effect?
3. Can we build a classifier to predict constructive vs destructive before merging?
4. Do other merge methods (Fisher, TIES, DARE) show different patterns?

## References

- Ablation database: `experiments.db`, `experiments_overnight.db`
- NxN baseline: `nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv`
- Experiment log: `references/experiment-log.md` (2026-01-20 entry)
