# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| URIEL selects poor sources | Merged model worse than individual sources; selected sources rank 25-48 in NxN | URIEL measures linguistic similarity, not cross-lingual transfer quality | Use REAL similarity or select directly from NxN matrix column |
| Merged accuracy drops | Sources individually 0.70+, merged result 0.55 | Averaging dilutes good models when one source dominates | Check ZS/Self ratio first; skip merge if >70% |
| IncTar fails to beat diagonal | Merged model with target included still worse than target alone | Target model already optimal; merging adds noise | Use ExcTar mode; IncTar only useful if target model is weak |
| Similarity matrix not found | `FileNotFoundError` for evaluation_matrix.csv | REAL similarity requires pre-computed NxN matrix | Copy from `nxn_results/*/evaluation_matrix.csv` or run `run_nxn_evaluation.py` |
| Wrong evaluation metric | STS-B Spearman ~0.55-0.58, seems similar across configs | Pipeline uses `evaluate_base_encoder.py` (proxy metric) | Use `evaluate_specific_model.py` for MASSIVE intent classification accuracy |
| Locale not found in similarity | `Target locale '' not found in similarity matrix` | Ablation config has `locales` as list in sweep but code expects string | Fixed in ablation_runner.py; ensure locales handled as list |

---

## Model Merging Decision Tree

```
Is ZS/Self ratio < 60%?
├─ YES → Merging may help
│        ├─ Use REAL similarity (not URIEL)
│        ├─ Try 5-7 source languages
│        └─ Evaluate with MASSIVE accuracy
│
└─ NO → Skip merging
         └─ Use best single source from NxN matrix
```

---

## Quick Diagnostic Commands

```bash
# Check ZS/Self ratio for a target
python -c "
import pandas as pd
nxn = pd.read_csv('nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv', index_col=0)
target = 'th-TH'
self_acc = nxn.loc[target, target]
best_zs = nxn[target].drop(target).max()
print(f'{target}: ZS/Self = {best_zs/self_acc:.1%}, Best ZS = {best_zs:.4f} ({nxn[target].drop(target).idxmax()})')
"

# Check what sources were selected
cat merged_models/similarity_URIEL_merge_*_ExcTar/merge_details.txt | grep -A3 "Merged Models"

# Compare against baseline
python -m merginguriel.experiments.query list --db experiments.db
```
