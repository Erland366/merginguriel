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
  updated: 2026-01-20
  author: Claude
---

# When Model Merging is Constructive

## General Description

Cross-lingual model merging does NOT universally improve performance. Our ablation studies (Jan 2026) show merging rarely beats the best zero-shot baseline. Only 1 of 2 constructive candidates (sw-KE) showed improvement, and only with REAL similarity.

**CRITICAL CORRECTION (Jan 20, 2026):**
- ZS/Self ratio alone does NOT predict merging success
- cy-GB has lowest ratio (53.6%) but FAILS to improve with merging
- sw-KE (56.4% ratio) improves ONLY with REAL similarity (not URIEL!)
- Fewer languages does NOT help—5 languages beats 3

## When to Apply

**Use model merging when:**
- Target shows empirically good sources (use REAL similarity)
- The target is sw-KE or similar (low-resource African language)
- Use **5 source languages** with **REAL similarity**

**When to skip merging:**
- Most languages! 47/49 have ZS/Self > 60%
- Even cy-GB (lowest ratio) doesn't benefit
- High-ratio targets (>80%): Use best single source or TIES

## Results Summary

### sw-KE: ONLY locale where merging beats baseline (ExcTar mode)

| Method | Sim | #Lang | Accuracy | Δ vs Baseline | Notes |
|--------|-----|-------|----------|---------------|-------|
| **similarity** | **REAL** | **5** | **0.4832** | **+3.4%** | **BEST - Only config that beats baseline!** |
| similarity | REAL | 3 | 0.4795 | +2.6% | Also beats baseline |
| similarity | URIEL | 5 | 0.4314 | -7.7% | URIEL fails! |
| similarity | URIEL | 3 | 0.4106 | -12.1% | Worst |

**Key insight**: REAL similarity + 5 languages beats baseline. URIEL completely fails.

### cy-GB: Lowest ZS/Self ratio but FAILS to beat baseline

| Method | Sim | #Lang | Accuracy | Δ vs Baseline | Notes |
|--------|-----|-------|----------|---------------|-------|
| similarity | URIEL | 5 | 0.4344 | -2.3% | Best but still fails |
| similarity | URIEL | 3 | 0.4190 | -5.8% | |
| similarity | REAL | 5 | 0.4166 | -6.3% | |
| similarity | REAL | 3 | 0.4025 | -9.5% | Worst |

**Key insight**: Despite lowest ZS/Self (53.6%), cy-GB gets NO benefit from merging. ZS/Self ratio alone does NOT predict success.

### Destructive Target: vi-VN (86% ZS/Self ratio)

| Method | Sim | #Lang | Accuracy | Δ vs Baseline | Notes |
|--------|-----|-------|----------|---------------|-------|
| **ties** | REAL | 5 | **0.7424** | **-0.07%** | **Nearly breaks even!** |
| similarity | REAL | 5 | 0.6769 | -6.6% | |
| similarity | REAL | 7 | 0.6698 | -7.3% | |
| average | REAL | 5 | 0.6614 | -8.2% | |
| similarity | URIEL | 3 | 0.5658 | -17.7% | |
| similarity | REAL | 3 | 0.5460 | -19.7% | Worst |

**Key insight**: TIES preserves features; more languages (5) helps destructive case

## Recommended Practice

### Decision Tree (CORRECTED Jan 20, 2026)

```
Model merging is generally NOT recommended. Only sw-KE showed improvement.

├─ Target is sw-KE or similar African low-resource language
│   ├─ Use: similarity method
│   ├─ Use: REAL similarity (NOT URIEL!)
│   ├─ Use: 5 source languages
│   └─ Expected: +3% improvement
│
├─ Target has low ZS/Self (like cy-GB)
│   └─ DO NOT MERGE - doesn't help despite low ratio
│
└─ All other targets (47/49 locales)
    ├─ Option A: Skip merging, use best single source
    └─ Option B: Use TIES method (nearly breaks even)
```

### Configuration for sw-KE (ONLY proven success)

```yaml
method: "similarity"
similarity_type: "REAL"   # REAL works, URIEL fails!
num_languages: 5          # 5 beats 3!
include_target: false
```

### Configuration for destructive targets (if you must merge)

```yaml
method: "ties"            # TIES preserves features
similarity_type: "REAL"
num_languages: 5
include_target: false
```

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| URIEL for sw-KE | Linguistic similarity ≠ cross-lingual transfer quality | Use REAL, not URIEL |
| cy-GB merging | Low ZS/Self doesn't guarantee merging helps | Test empirically, don't trust ratio alone |
| Fewer languages for sw-KE | Not enough diverse signal | Use 5 languages, not 3 |
| Linear averaging for vi-VN | Averages away good features | Use TIES for destructive |
| Fisher merging | Too resource-intensive, crashes machine | Avoid Fisher |

## Method Comparison Summary

| Method | Best For | Worst For | Notes |
|--------|----------|-----------|-------|
| **similarity** | sw-KE ONLY | Most other locales | MUST use REAL similarity |
| **ties** | Destructive (vi-VN) | — | Preserves features, nearly breaks even |
| **average** | Neither | Both | Mediocre across the board |
| **fisher** | SKIP | — | Too resource-intensive |

## NeuroMerging: Addressing the Interference Problem (Jan 20, 2026)

### The Problem

Our ablations revealed that cy-GB suffers **interference** (-1.58% Merging Effect) while sw-KE achieves **synergy** (+2.30%). Standard merging methods apply uniform weights across all parameters, which doesn't account for how task-specific knowledge is encoded.

### The Solution: Neuronal Subspace Decomposition

NeuroMerging (Fang et al., 2025) decomposes task vectors into two subspaces:

```
τ_parallel = (w_pretrained · τ) / ||w_pretrained||² × w_pretrained
τ_orthogonal = τ - τ_parallel
```

- **Parallel subspace**: Input sensitivity (how the model responds to input patterns)
- **Orthogonal subspace**: Task adaptability (novel task-specific capabilities)

The key insight: **88% of task-specific capabilities reside in the orthogonal subspace**. Interference often occurs when conflicting parallel components are merged.

### Algorithm

1. Create neuronal task vectors: `τ = w_finetuned - w_pretrained`
2. Decompose each neuron into parallel and orthogonal components
3. Apply magnitude masking (keep top 85% by default)
4. Merge orthogonal subspace using elect+mean (TIES-style)
5. Scale by λ₂ = 1/(1-σ) where σ is the L1-norm ratio

**Default configuration**:
- λ₁ = 0 (ignore parallel subspace - has little impact)
- λ₂ = auto-computed from masked element ratio
- mask_rate = 0.15 (mask smallest 15%)

### Expected Outcomes

| Target | Current Best | Expected with NeuroMerging |
|--------|-------------|---------------------------|
| sw-KE | +3.4% (REAL/5) | +3-5% (amplify synergy) |
| cy-GB | -2.3% (URIEL/5) | -1% to +1% (reduce interference) |
| vi-VN | -0.07% (TIES) | Similar or better |

### Configuration

```yaml
method: "neuromerging"
similarity_type: "REAL"
num_languages: 5
include_target: false
# NeuroMerging-specific (optional, defaults are usually good):
# neuro_mask_rate: 0.15
# neuro_lambda_1: 0.0
# neuro_lambda_2: null  # auto-computed
```

### Why NeuroMerging Should Help cy-GB

cy-GB's interference likely comes from conflicting parallel components being merged. By:
1. Ignoring the parallel subspace (λ₁=0)
2. Merging only orthogonal components (task-specific adaptations)
3. Using elect+mean to resolve sign conflicts

We expect to reduce interference from -1.58% toward 0% or positive.

### Implementation

Files:
- `submodules/auto_merge_llm/auto_merge_llm/methods/neuromerging.py`
- `submodules/auto_merge_llm/auto_merge_llm/utils/neuronal_task_vector.py`
- Config: `configs/ablations/neuromerging_validation.yaml`

### Validation Results (NEGATIVE)

| Target | ZS Baseline | NeuroMerging | Similarity | TIES |
|--------|-------------|--------------|------------|------|
| sw-KE | 0.4670 | 0.4549 (-1.2%) | **0.4832 (+1.6%)** | 0.4297 (-3.7%) |
| cy-GB | 0.4445 | 0.4008 (-4.4%) | 0.4166 (-2.8%) | 0.4008 (-4.4%) |
| vi-VN | 0.7431 | 0.6358 (-10.7%) | 0.6769 (-6.6%) | **0.7424 (-0.07%)** |

**Result**: NeuroMerging performed WORSE than similarity and TIES on all three targets.

### Why NeuroMerging Failed

1. **Orthogonal subspace may not contain cross-lingual knowledge**: The paper validates on single-task merging (same language, different tasks). Cross-lingual transfer may work differently.

2. **Task vectors are not aligned**: Source models are trained on different languages. Their task vectors may be in different subspaces, making decomposition less meaningful.

3. **λ₂ auto-computation gives λ₂ ≈ 1.0**: This effectively removes the scaling benefit, making NeuroMerging similar to TIES with extra decomposition overhead.

### Recommendation

**Do NOT use NeuroMerging for cross-lingual model merging.**
- For constructive targets (sw-KE): Use **similarity** with REAL/5-lang
- For destructive targets (vi-VN): Use **TIES**
- For most targets: Don't merge at all

---

## Key Corrections (Jan 20, 2026)

| Previous Claim | Actual Finding |
|----------------|----------------|
| "ZS/Self < 60% = constructive" | **WRONG** - cy-GB (53.6%) fails to improve |
| "URIEL beats REAL for constructive" | **WRONG** - REAL beats URIEL for sw-KE |
| "3 languages better than 5" | **WRONG** - 5 beats 3 for sw-KE |
| "Can identify constructive targets from ratio" | **WRONG** - must test empirically |

## Deep Analysis: Why sw-KE Works but cy-GB Doesn't (Jan 20, 2026)

The key finding is the **Merging Effect** - whether combining sources creates synergy or interference.

### The Merging Effect

```
Merging Effect = Merged Model Accuracy - Average of Source Accuracies
```

| Target | REAL/5 Merged | Source Avg | Merging Effect | Outcome |
|--------|---------------|------------|----------------|---------|
| sw-KE | 0.4832 | 0.4602 | **+2.30%** | SYNERGY |
| cy-GB | 0.4166 | 0.4324 | **-1.58%** | INTERFERENCE |

### Why Does sw-KE Achieve Synergy?

1. Sources have **COMPLEMENTARY** features that combine constructively
2. The target (Swahili/Bantu) benefits from multi-source averaging
3. REAL selection finds sources that work well **together**

### Why Does cy-GB Suffer Interference?

1. Sources have **CONFLICTING** features that destructively interfere
2. The target (Welsh/Celtic) has unique features that get diluted
3. Even REAL-selected sources don't combine well

### What We Can't Predict From

Both targets are similar on these metrics, yet behave oppositely:
- ZS/Self ratio: 56.4% vs 53.6% (both low)
- Source diversity: 4/5 vs 5/5 unique families (both diverse)
- Top sources: Share lv-LV, tl-PH, sq-AL, fr-FR (similar pools)

### The Unpredictable Nature of Merging

**Conclusion**: Cannot predict merging success from proxy metrics alone.
Must empirically test the **Merging Effect** for each target.

### New Recommendation

Before committing to merging:
1. Compute source average: `avg(source_perf_on_target)`
2. Run a pilot merge experiment
3. If `merged_perf < source_avg`, **do not merge**
4. Positive merging effect indicates potential success

## Directional Consensus: Variance Reduction for Interference (Jan 20, 2026)

### The Idea

**Hypothesis**: Task vectors from different languages point in different directions. Projecting them onto a consensus direction should reduce interference.

**Algorithm** (per-parameter):
```
1. Extract task vectors: τ_i = model_i - xlmr_base
2. Compute consensus direction: d = normalize(Σ τ_i)
3. Project each vector: τ_aligned = (τ · d) × d
4. Weighted aggregation: merged = base + Σ(w_i × τ_aligned_i)
```

### Validation Results

| Target | Effect Type | directional_consensus | similarity | Δ vs similarity |
|--------|-------------|----------------------|------------|-----------------|
| cy-GB | INTERFERENCE | **42.60%** | 41.66% | **+0.94%** |
| sw-KE | SYNERGY | 46.87% | **48.32%** | -1.45% |

### Key Insight

Directional consensus is a **variance reduction** technique:
- **Helps interference** (cy-GB): Forces conflicting directions to align, reducing destructive interference
- **Hurts synergy** (sw-KE): Removes beneficial orthogonal components that create constructive combination

### When to Use

| Merging Effect | Recommended Method | Rationale |
|----------------|-------------------|-----------|
| NEGATIVE (interference) | `directional_consensus` | Reduces variance, +0.94% for cy-GB |
| POSITIVE (synergy) | `similarity` | Preserves beneficial variance |

### Updated Decision Tree

```
├─ Target has NEGATIVE Merging Effect (interference)
│   ├─ Use: directional_consensus
│   ├─ Similarity: REAL
│   ├─ Languages: 5
│   └─ Expected: Reduce interference by ~1%
│
├─ Target has POSITIVE Merging Effect (synergy, e.g., sw-KE)
│   ├─ Use: similarity (NOT directional_consensus!)
│   ├─ Similarity: REAL
│   ├─ Languages: 5
│   └─ Expected: +3% improvement
│
└─ Unknown Merging Effect
    ├─ Run pilot: compute source_avg and merged_perf
    ├─ If merged < source_avg → use directional_consensus
    └─ If merged > source_avg → use similarity
```

### Configuration

```yaml
method: "directional_consensus"
similarity_type: "REAL"
num_languages: 5
include_target: false
```

### Implementation

- Method: `submodules/auto_merge_llm/auto_merge_llm/methods/directional_consensus.py`
- Config: `configs/ablations/directional_consensus_validation.yaml`
- Database: `experiments_directional_consensus.db`

---

## Open Questions

1. ~~Why does sw-KE benefit while cy-GB doesn't?~~ → **ANSWERED**: Merging Effect (synergy vs interference)
2. ~~Can we reduce interference for cy-GB?~~ → **ANSWERED**: Yes, directional_consensus helps (+0.94%)
3. What makes sw-KE sources combine well? (complementary vs conflicting features)
4. Can we predict Merging Effect without running the full merge?
5. Are there linguistic features that predict positive Merging Effect?

## References

- Ablation databases: `experiments_constructive.db`, `experiments_methods.db`, `experiments_num_lang.db`
- NxN baseline: `nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv`
- Experiment log: `references/experiment-log.md`
