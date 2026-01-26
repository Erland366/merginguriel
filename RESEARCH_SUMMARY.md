# Merging Effect Prediction: Research Summary

## Research Question

**Can we predict whether cross-lingual model merging will result in SYNERGY or INTERFERENCE before running the expensive merge operation?**

## Key Finding

**PARTIAL SUCCESS** - Source diversity predicts Merging Effect in ~67% of cases.

| Validation Set | Predictions Correct |
|----------------|---------------------|
| Initial 3 targets (sw-KE, cy-GB, vi-VN) | 3/3 (100%) |
| New 6 targets | 4/6 (67%) |
| **Combined** | **7/9 (78%)** |

## Validation Experiment Results (Jan 20, 2026)

### Full Results Table

| Target | Rank | Synergy Score | Predicted | Merged Acc | Expected | Effect | Actual | Match |
|--------|------|---------------|-----------|------------|----------|--------|--------|-------|
| az-AZ | 1 | 0.1514 | SYNERGY | 0.6627 | 0.6371 | +2.57% | SYNERGY | ✓ |
| tr-TR | 2 | 0.1255 | SYNERGY | 0.7290 | 0.6883 | +4.07% | SYNERGY | ✓ |
| af-ZA | 4 | 0.1192 | SYNERGY | 0.5740 | 0.6018 | -2.78% | INTERFERENCE | ✗ |
| am-ET | 47 | 0.0600 | INTERFERENCE | 0.4361 | 0.4658 | -2.97% | INTERFERENCE | ✓ |
| tl-PH | 48 | 0.0417 | INTERFERENCE | 0.5921 | 0.5272 | +6.49% | SYNERGY | ✗ |
| id-ID | 49 | 0.0375 | INTERFERENCE | 0.7091 | 0.7386 | -2.95% | INTERFERENCE | ✓ |

### Failure Analysis

**af-ZA (Predicted SYNERGY, Actual INTERFERENCE -2.78%)**
- High diversity (0.158) but weak source quality
- Sources: id-ID, hy-AM, hu-HU, ka-GE, lv-LV
- Source accuracies: 0.547-0.635 vs target self-perf 0.849
- **Lesson**: Diversity doesn't help when sources are much weaker than target

**tl-PH (Predicted INTERFERENCE, Actual SYNERGY +6.49%)**
- Low diversity (0.058) but sources are regionally coherent
- Sources: vi-VN, km-KH, jv-ID, th-TH, ms-MY (all Southeast Asian)
- **Lesson**: Regional coherence may enable synergy even with low diversity

## The Predictor

### Formula (V1 - Original)
```
synergy_score = diversity_score / (1 + source_acc_std × 10)

where:
  diversity_score = 1 - mean(source_pairwise_similarity)
  source_acc_std = std(source_accuracies_on_target)
```

### Feature Correlations

| Feature | Correlation with Actual Effect | Notes |
|---------|--------------------------------|-------|
| diversity_score | +0.15 | Weak positive |
| source_acc_std | -0.18 | Weak negative |
| source_quality | -0.02 | No signal |

**Note**: Correlations are weak on 6 validation samples. More data needed.

## Insights

### What Works (4/6 correct)
1. **High diversity + Low variance → SYNERGY**: az-AZ, tr-TR
2. **Low diversity + High variance → INTERFERENCE**: am-ET, id-ID

### What Doesn't Fit
1. **High diversity + Weak sources → INTERFERENCE**: af-ZA
2. **Low diversity + Regional coherence → SYNERGY**: tl-PH

### Hypotheses for Refinement

1. **Source Quality Threshold**: Merging only helps when sources are at least 70% as good as target
2. **Regional Coherence**: Sources from same linguistic region may synergize despite low diversity
3. **Complementary Features**: Need to measure what features sources contribute, not just diversity

## Practical Recommendations

Based on validation results:

| Condition | Recommendation | Confidence |
|-----------|----------------|------------|
| High diversity (>0.12) + Low acc std (<0.02) | Merge | High |
| Low diversity (<0.06) + High acc std (>0.04) | Don't merge | High |
| High diversity + Weak sources (<70% of target) | Don't merge | Medium |
| Low diversity + Same region sources | Test empirically | Low |

## Files

| File | Description |
|------|-------------|
| `analysis/merging_effect_analysis.py` | Feature computation |
| `analysis/merging_effect_predictor.py` | All-locale predictions |
| `analysis/merging_effect_predictions.csv` | Full predictions (49 locales) |
| `configs/ablations/synergy_prediction_validation.yaml` | Validation config |
| `experiments_synergy_validation.db` | Validation results |

## Next Steps

1. **Expand validation**: Run on 10+ more targets to improve correlations
2. **Regional coherence metric**: Develop a measure of source regional alignment
3. **Task vector analysis**: Analyze what features each source contributes
4. **Threshold optimization**: Find optimal cutoffs for synergy/interference prediction

## Conclusion

The diversity-based predictor achieves **78% accuracy** (7/9 correct) combining initial and validation results. This is better than random (50%) but not sufficient for production use. The key insight is that **diversity alone is insufficient** - source quality and regional coherence also matter.

For practical use:
- **Use predictor for screening** high-potential targets (ranked top 10)
- **Always validate empirically** before committing to merging
- **Skip merging** for targets where predictor shows low confidence

---

*Generated: 2026-01-20*
*Branch: idea4-merging-effect-prediction*
*Validation: 6 targets tested, 4/6 (67%) correct*
