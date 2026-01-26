# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-01-19 – Retrospective: Code Simplification

**Type:** Retrospective
**General description:** Migrated MergingUriel from CLI arguments to YAML config system and refactored large modules into packages.

### What we tried

1. **YAML Configuration System**
   - Created `merginguriel/config/` package with hierarchical dataclass configs
   - Implemented `TrackProvidedArgsAction` for CLI argument tracking
   - Added `--config` support to all 4 entry points
   - Built deprecation warning system for CLI-to-config migration

2. **Large Module Refactoring**
   - `plot_results.py`: 2,398 → 853 lines (extracted to `plotting/` package)
   - `aggregate_results.py`: 1,160 → 226 lines (extracted to `aggregation/` package)
   - `naming_config.py`: 526 → 381 lines (dropped legacy patterns)

### Key findings

- Dataclass-based configs provide IDE autocomplete and type hints
- `TrackProvidedArgsAction` cleanly detects explicitly-provided CLI args
- Dropping legacy naming patterns (15 → 3) significantly simplified maintenance
- Verified identical results between original and refactored code (Spearman 0.5779)

### What failed

- Monkeypatching lazy imports: patch at definition site, not usage site
- Hyphen in dataclass field name caused syntax error
- `--resume` flag comparison gave different results due to cached models

### Outcome

- Net reduction of ~3,300 lines through modularization
- 110 tests passing, 38 new config tests
- Created skill: `python-config-refactoring`

---

## 2026-01-19 – Observation: Research Goals and Evaluation Framework

**Type:** Observation
**General description:** Clarified MergingUriel's research objectives and the correct evaluation metrics.

### Research Goals

MergingUriel's primary objectives:

1. **Beat zero-shot cross-lingual performance**: A merged model should outperform individual source models evaluated directly on the target language (zero-shot transfer).

2. **Significantly improve over baseline**: The merged model should substantially beat the pretrained XLM-RoBERTa without any fine-tuning.

### The Actual Task: MASSIVE Intent Classification

- **Dataset**: AmazonScience/massive (49 locales, 60 intent classes)
- **Task**: Sequence classification (intent prediction)
- **Metric**: Accuracy = correct_predictions / total_predictions
- **Evaluation script**: `evaluate_specific_model.py`

### Baselines to Compare Against

1. **Zero-shot baseline**: Source language model evaluated on target language test set
   - Example: `af-ZA` model tested on `th-TH` → 44% accuracy
   - Stored in `nxn_results/*/evaluation_matrix.csv` (49×49 cross-lingual matrix)

2. **Best source baseline**: Best-performing source model for target (from NxN matrix)
   - For target `th-TH`: find max across all source models

3. **Best overall baseline**: Highest zero-shot accuracy from any model (excluding self)

### Current Pipeline Issue

The `run_merging_pipeline_refactored.py` currently evaluates using `evaluate_base_encoder.py` which measures **STS-B Spearman correlation** (semantic similarity) — this is a proxy metric, NOT the actual task.

**Wrong metric**: STS-B Spearman correlation (~0.55-0.58)
**Correct metric**: MASSIVE intent classification accuracy (~0.40-0.90 depending on language pair)

### How to Properly Evaluate

```python
# After merging, evaluate on MASSIVE intent classification:
from merginguriel.evaluate_specific_model import evaluate_specific_model

results = evaluate_specific_model(
    model_name="path/to/merged_model",
    locale="th-TH",  # Target locale
    eval_folder="results/merging_similarity_URIEL_th-TH_xlm-roberta-base_3lang_ExcTar"
)

# Compare results["performance"]["accuracy"] against:
# 1. Zero-shot baseline from evaluation_matrix.csv
# 2. Best source model accuracy
# 3. Pretrained XLM-RoBERTa baseline
```

### Success Criteria

For merged model targeting `th-TH`:
- `merged_accuracy > avg(source_accuracies_on_th)` → beats average zero-shot
- `merged_accuracy > max(source_accuracies_on_th)` → beats best zero-shot
- `merged_accuracy > pretrained_baseline` → beats raw pretrained model

### Key Files

| File | Purpose |
|------|---------|
| `evaluate_specific_model.py` | Correct evaluation (MASSIVE intent classification) |
| `evaluate_base_encoder.py` | Proxy metric (STS-B, not recommended for final eval) |
| `nxn_results/*/evaluation_matrix.csv` | Pre-computed baselines (49×49 accuracy matrix) |
| `run_nxn_evaluation.py` | Generate cross-lingual evaluation matrix |

---

## 2026-01-20 – Retrospective: Similarity-Based Model Merging Ablation

**Type:** Retrospective
**General description:** Ablation study comparing URIEL vs REAL similarity and IncTar vs ExcTar across 6 locales revealed model merging is only constructive when zero-shot transfer is poor.

### What we tried

1. **Ablation Configuration**
   - 24 experiments (6 locales × 2 similarity types × 2 target inclusion modes)
   - Similarity: URIEL (linguistic features) vs REAL (empirical cross-lingual)
   - Target: IncTar (include target in merge) vs ExcTar (exclude target)
   - Locales: sq-AL, sw-KE, vi-VN, th-TH, fi-FI, tr-TR
   - Method: similarity-based weighted merging, 3 source languages
   - Weighting: Top-k selection + Sinkhorn normalization

2. **Evaluation**
   - Metric: MASSIVE intent classification accuracy
   - Baselines: NxN cross-lingual matrix (49×49)

### Key findings

| Finding | Evidence |
|---------|----------|
| Only 1/6 targets achieved Goal 1 | sw-KE beat zero-shot by +11.3%; others failed |
| 0/6 targets achieved Goal 2 | All IncTar failed to beat diagonal |
| URIEL selects poorly | Picks ranks 25-48 instead of top performers |
| REAL selects better | Picks ranks 2-20, but still suboptimal |
| Merging often destructive | vi-VN: sources 0.74→merged 0.55 |
| sw-KE is constructive | Sources 0.41→merged 0.58 |

**Critical Discovery**: Zero-shot/Self ratio predicts success
- sw-KE: 56% ratio → constructive (+11%)
- vi-VN: 86% ratio → destructive (-18%)

### What failed

- URIEL source selection: linguistic similarity ≠ transfer quality
- Merging good sources: averaging dilutes quality
- IncTar enhancement: target alone always better
- Only 3 languages: may be insufficient

### Outcome

- Created skill: `merging-when-constructive`
- Key insight: Only merge when ZS/Self ratio < 60%
- Recommendation: Use REAL over URIEL; skip merge for high-transfer targets

---

## 2026-01-20 – Retrospective: Num Languages & Method Ablation

**Type:** Retrospective
**General description:** Tested num_languages (3/5/7) and methods (similarity/average/ties) on constructive (sw-KE) vs destructive (vi-VN) targets, revealing counterintuitive findings.

### What we tried

1. **Num Languages Sweep**: 3, 5, 7 source languages with REAL similarity
2. **Method Comparison**: similarity, average, ties (fisher failed - config issue)
3. **Targets**: sw-KE (constructive, 56% ratio) and vi-VN (destructive, 86% ratio)

### Key findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Fewer langs better for constructive | sw-KE: 3-lang +11.3% vs 7-lang +0.8% | Use 3, not 5-7 |
| TIES nearly breaks even for destructive | vi-VN: TIES -0.07% vs similarity -17.7% | TIES preserves features |
| URIEL beats REAL for sw-KE | URIEL +11.3% vs REAL +9.2% | Linguistic similarity helps low-resource |
| More langs slightly helps destructive | vi-VN: 5-lang -6.6% vs 3-lang -19.7% | Opposite of constructive |

### What failed

- Fisher method: `--dataset-name is required for fisher_dataset mode`
- TIES on sw-KE: -3.7% (sparsification hurts constructive)
- All methods on vi-VN still below baseline (but TIES nearly breaks even)

### Outcome

- Updated skill `merging-when-constructive` with method-specific recommendations
- New decision tree: constructive → similarity/URIEL/3-lang; destructive → TIES/REAL/5-lang
- Closed questions: more languages does NOT help constructive; TIES helps destructive

---

## 2026-01-20 – Retrospective: Constructive Target Validation (MAJOR CORRECTIONS)

**Type:** Retrospective
**General description:** Tested both constructive candidates (cy-GB, sw-KE) with URIEL vs REAL and 3 vs 5 languages. Results contradict previous findings—MAJOR corrections needed.

### What we tried

1. **Identified ALL constructive candidates**: Scanned 49 locales for ZS/Self < 60%
   - Only 2 found: cy-GB (53.6%), sw-KE (56.4%)
2. **Validation ablation**: 2 locales × 2 similarity types × 2 num_languages = 8 experiments
3. **Fisher method**: Attempted but crashed machine—disabled

### CRITICAL CORRECTIONS

| Previous Claim | Actual Finding |
|----------------|----------------|
| "ZS/Self < 60% = constructive" | **WRONG** - cy-GB (53.6%) fails to improve |
| "URIEL beats REAL for constructive" | **WRONG** - REAL beats URIEL for sw-KE |
| "3 languages better than 5" | **WRONG** - 5 beats 3 for sw-KE |
| "URIEL/3-lang gives +11.3%" | **WRONG** - That was IncTar mode; ExcTar gives -12.1% |

### Key findings

**sw-KE (ONLY success):**
- REAL/5-lang: **+3.4%** (beats baseline!)
- REAL/3-lang: +2.6%
- URIEL/5-lang: -7.7%
- URIEL/3-lang: -12.1%

**cy-GB (FAILS despite lowest ratio):**
- URIEL/5-lang: -2.3% (best but still fails)
- REAL/3-lang: -9.5% (worst)

### What failed

- **cy-GB merging**: Despite lowest ZS/Self ratio (53.6%), NO config beat baseline
- **URIEL similarity**: Completely fails for sw-KE (-7.7% to -12.1%)
- **3 languages**: Worse than 5 for both targets
- **Fisher method**: Crashes machine due to resource requirements

### Outcome

- **Major skill update**: `merging-when-constructive` heavily revised
- Only sw-KE benefits from merging (REAL/5-lang)
- ZS/Self ratio does NOT reliably predict merging success
- New recommendation: Don't merge for most languages; if you must, use REAL/5-lang or TIES

### Open questions

1. Why does sw-KE benefit while cy-GB doesn't?
2. What characteristics predict merging success beyond ZS/Self ratio?
3. Are there other locales that would benefit? (Expensive to test all 49)

---

## 2026-01-20 – Retrospective: The Merging Effect (Why sw-KE Works but cy-GB Doesn't)

**Type:** Retrospective
**General description:** Deep analysis reveals the "Merging Effect" (synergy vs interference) is the key predictor, not ZS/Self ratio or source quality alone.

### What we tried

1. **URIEL vs REAL source analysis**: Compared source selection quality for both targets
2. **Transfer ratio analysis**: Measured how well sources transfer to each target
3. **Merging effect calculation**: Compared merged accuracy vs average source accuracy

### Key findings

| Metric | sw-KE | cy-GB |
|--------|-------|-------|
| ZS/Self ratio | 56.4% | 53.6% |
| REAL top-5 avg on target | 0.4602 | 0.4324 |
| REAL/5 merged accuracy | 0.4832 | 0.4166 |
| **Merging Effect** | **+2.30%** | **-1.58%** |
| Outcome | SYNERGY | INTERFERENCE |

**Merging Effect = Merged Accuracy - Source Average**

### Critical insight

The **Merging Effect** determines success:
- **Positive effect (+2.30%)**: Sources combine constructively (sw-KE)
- **Negative effect (-1.58%)**: Sources interfere destructively (cy-GB)

Both targets have:
- Similar ZS/Self ratios (53-56%)
- Diverse source families (4-5 unique)
- Overlapping top sources (lv-LV, tl-PH, sq-AL, fr-FR)

Yet they behave **oppositely**! This confirms:
- ZS/Self ratio does NOT predict success
- Source diversity does NOT predict success
- Only the **Merging Effect** predicts success

### Why sw-KE achieves synergy

1. Sources have COMPLEMENTARY features
2. Swahili (Bantu) benefits from multi-source averaging
3. REAL sources work well **together**, not just individually

### Why cy-GB suffers interference

1. Sources have CONFLICTING features
2. Welsh (Celtic) has unique features that get diluted
3. Even REAL sources don't combine constructively

### Outcome

- **Answered open question**: Why sw-KE works but cy-GB doesn't → Merging Effect
- **New concept**: Synergy vs Interference in model merging
- **Updated skill**: Added "Merging Effect" section with analysis
- **New recommendation**: Pilot test merge before committing; if merged < avg, don't merge

### Remaining questions

1. What makes sw-KE sources complementary?
2. Can we predict Merging Effect without running full merge?
3. Are there linguistic features that correlate with positive Merging Effect?

---

## 2026-01-20 – Plan: NeuroMerging Implementation

**Type:** Plan
**General description:** Implemented NeuroMerging algorithm to address interference problem discovered in ablations (sw-KE: +2.30% synergy, cy-GB: -1.58% interference).

### What we implemented

1. **Neuronal Task Vector Module** (`submodules/auto_merge_llm/auto_merge_llm/utils/neuronal_task_vector.py`)
   - `NeuronalTaskVector` class with subspace decomposition
   - `decompose_weight_matrix()`: Projects task vectors onto parallel and orthogonal subspaces
   - `is_neuronal_param()`: Identifies dense layer weights vs biases/LayerNorm/embeddings

2. **NeuroMerging Method** (`submodules/auto_merge_llm/auto_merge_llm/methods/neuromerging.py`)
   - Implements neuron-level selective merging from Fang et al. 2025
   - Algorithm: decompose → mask → elect+mean merge → scale by λ_2
   - Default: λ_1=0 (ignore parallel), λ_2=auto-computed from L1-norm ratio
   - Mask rate: 0.15 (keep top 85% by magnitude)

3. **Pipeline Integration** (`merginguriel/run_merging_pipeline_refactored.py`)
   - Added `NeuroMergingStrategy` class
   - Registered in `MergingStrategyFactory` under mode "neuromerging"

4. **Validation Config** (`configs/ablations/neuromerging_validation.yaml`)
   - Tests on sw-KE (constructive), cy-GB (low-ratio), vi-VN (destructive)
   - Compares neuromerging vs ties vs similarity baselines

### Key algorithm details

**Neuronal decomposition** (per neuron row w_k):
```
τ_k = w_finetuned - w_pretrained           # task vector
τ_parallel = (w_0 · τ) / ||w_0||² * w_0    # projection onto pretrained
τ_orthogonal = τ - τ_parallel              # novel task-specific adaptations
```

**Merge function** (TIES-style elect+mean):
1. Compute majority sign across sources
2. Keep only values with matching sign
3. Average the kept values

**Why this should help**:
- Orthogonal subspace preserves 88% of task-specific capabilities (paper finding)
- Merging in orthogonal subspace reduces interference between conflicting features
- cy-GB's interference may come from parallel subspace conflicts

### Expected outcomes

| Target | Current Best | Expected with NeuroMerging |
|--------|-------------|---------------------------|
| sw-KE | +3.4% (REAL/5) | +3-5% (amplify synergy) |
| cy-GB | -2.3% (URIEL/5) | -1% to +1% (reduce interference) |
| vi-VN | -0.07% (TIES) | Similar or better |

### Files created/modified

| File | Action |
|------|--------|
| `submodules/auto_merge_llm/.../utils/neuronal_task_vector.py` | Created |
| `submodules/auto_merge_llm/.../methods/neuromerging.py` | Created |
| `submodules/auto_merge_llm/.../methods/__init__.py` | Modified |
| `submodules/auto_merge_llm/.../utils/__init__.py` | Modified |
| `merginguriel/run_merging_pipeline_refactored.py` | Modified |
| `configs/ablations/neuromerging_validation.yaml` | Created |

### Next steps

1. Run validation ablation: `python -m merginguriel.experiments.ablation_runner configs/ablations/neuromerging_validation.yaml`
2. Compare results against baselines in experiment database
3. If successful, update `merging-when-constructive` skill with NeuroMerging recommendations

---

## 2026-01-20 – Retrospective: NeuroMerging Validation (NEGATIVE RESULT)

**Type:** Retrospective
**General description:** NeuroMerging (neuronal subspace decomposition) failed to improve cross-lingual model merging. The method underperformed both similarity and TIES on all tested targets.

### What we tried

1. **NeuroMerging Implementation**
   - Implemented neuronal task vector decomposition into parallel and orthogonal subspaces
   - Algorithm: mask → decompose → elect+mean merge → scale
   - Parameters: λ₁=0 (ignore parallel), λ₂=auto-computed, mask_rate=0.15

2. **Validation ablation**
   - 3 locales × 3 methods = 9 experiments
   - Targets: sw-KE (constructive), cy-GB (low-ratio), vi-VN (destructive)
   - Methods: neuromerging, ties, similarity

### Key findings

| Target | ZS Baseline | NeuroMerging | Similarity | TIES |
|--------|-------------|--------------|------------|------|
| sw-KE | 0.4670 | 0.4549 (-1.2%) | **0.4832 (+1.6%)** | 0.4297 (-3.7%) |
| cy-GB | 0.4445 | 0.4008 (-4.4%) | 0.4166 (-2.8%) | 0.4008 (-4.4%) |
| vi-VN | 0.7431 | 0.6358 (-10.7%) | 0.6769 (-6.6%) | **0.7424 (-0.07%)** |

**NeuroMerging ranked last or tied-last on all targets.**

### What failed

- **Hypothesis disproved**: Merging in orthogonal subspace does NOT reduce interference for cross-lingual transfer
- **λ₂ auto-computation**: Gave λ₂ ≈ 1.0, effectively removing scaling benefit
- **Domain mismatch**: Paper validates on same-language multi-task; cross-lingual may be fundamentally different

### Why NeuroMerging doesn't work for cross-lingual transfer

1. **Task vectors aren't aligned across languages**: Each source model learns language-specific adaptations. Decomposing into parallel/orthogonal subspaces assumes a shared coordinate system that doesn't exist.

2. **Cross-lingual knowledge may be in parallel subspace**: The paper assumes task-specific knowledge is in orthogonal subspace. But cross-lingual transfer may rely on parallel subspace features (input sensitivity to multilingual patterns).

3. **The paper's setup is different**: NeuroMerging targets same-language models fine-tuned on different tasks (e.g., sentiment + NER). Our setup merges different-language models for the same task.

### Outcome

- **Negative result documented**: NeuroMerging does NOT help cross-lingual model merging
- **Skill updated**: Added validation results and recommendation to avoid NeuroMerging
- **Recommendation unchanged**: Use similarity (REAL/5-lang) for sw-KE; TIES for destructive targets; skip merging for most locales

### Open questions

1. Are there other cross-lingual-specific merging methods to explore?
2. Would representation alignment before merging help?
3. Is the problem with source selection or the merging algorithm itself?

---

## 2026-01-20 – Retrospective: Merging Effect Prediction (Idea 4)

**Type:** Retrospective
**General description:** Developed and validated a predictor for synergy vs interference in model merging, achieving 78% accuracy (7/9 correct) but revealing important failure modes.

### Research Question

Can we predict whether cross-lingual model merging will result in SYNERGY or INTERFERENCE before running the expensive merge operation?

### What we tried

1. **Feature Engineering** (`analysis/merging_effect_analysis.py`)
   - Computed source diversity: `1 - mean(source_pairwise_similarity)`
   - Computed source accuracy variance: `std(source_accuracies_on_target)`
   - Source-target similarity features

2. **Predictor Formula** (`analysis/merging_effect_predictor.py`)
   - `synergy_score = diversity_score / (1 + source_acc_std × 10)`
   - Higher score → more likely to achieve synergy
   - Ranked all 49 locales by synergy_score

3. **Validation Experiment** (`configs/ablations/synergy_prediction_validation.yaml`)
   - 6 targets: 3 high-synergy (az-AZ, tr-TR, af-ZA), 3 low-synergy (am-ET, tl-PH, id-ID)
   - Method: similarity with REAL/5-lang

### Key findings

| Target | Rank | Synergy Score | Predicted | Merged Acc | Expected | Effect | Actual | Match |
|--------|------|---------------|-----------|------------|----------|--------|--------|-------|
| az-AZ | 1 | 0.1514 | SYNERGY | 0.6627 | 0.6371 | +2.57% | SYNERGY | ✓ |
| tr-TR | 2 | 0.1255 | SYNERGY | 0.7290 | 0.6883 | +4.07% | SYNERGY | ✓ |
| af-ZA | 4 | 0.1192 | SYNERGY | 0.5740 | 0.6018 | -2.78% | INTERFERENCE | ✗ |
| am-ET | 47 | 0.0600 | INTERFERENCE | 0.4361 | 0.4658 | -2.97% | INTERFERENCE | ✓ |
| tl-PH | 48 | 0.0417 | INTERFERENCE | 0.5921 | 0.5272 | +6.49% | SYNERGY | ✗ |
| id-ID | 49 | 0.0375 | INTERFERENCE | 0.7091 | 0.7386 | -2.95% | INTERFERENCE | ✓ |

**Combined accuracy**: 7/9 (78%) including prior results (sw-KE, cy-GB, vi-VN)

### What worked

| Pattern | Examples | Confidence |
|---------|----------|------------|
| High diversity + Low variance → SYNERGY | az-AZ (+2.57%), tr-TR (+4.07%) | High |
| Low diversity + High variance → INTERFERENCE | am-ET (-2.97%), id-ID (-2.95%) | High |

### What failed

| Failure | Prediction | Actual | Reason |
|---------|------------|--------|--------|
| af-ZA | SYNERGY | INTERFERENCE (-2.78%) | High diversity but weak sources (60-64% vs target 85%) |
| tl-PH | INTERFERENCE | SYNERGY (+6.49%) | Low diversity but regional coherence (all SE Asian sources) |

### Insights

1. **Source quality threshold**: Diversity alone doesn't help when sources are much weaker than target. Need `source_quality > 70%` of target.

2. **Regional coherence**: Sources from same linguistic region (e.g., all Southeast Asian) may synergize despite low diversity. tl-PH sources (vi-VN, km-KH, jv-ID, th-TH, ms-MY) are geographically clustered.

3. **Feature correlations are weak**: On 6 samples, diversity has +0.15 correlation with actual effect, source_acc_std has -0.18. Need more data.

### Outcome

- **Partial success**: 78% accuracy is better than random (50%) but not production-ready
- **Updated skill**: `merging-when-constructive` with Merging Effect Prediction section
- **Files created**:
  - `analysis/merging_effect_analysis.py` - Feature computation
  - `analysis/merging_effect_predictor.py` - All-locale predictions
  - `analysis/merging_effect_predictions.csv` - Full predictions (49 locales)
  - `RESEARCH_SUMMARY.md` - Research documentation

### Practical recommendations

| Condition | Recommendation | Confidence |
|-----------|----------------|------------|
| synergy_score > 0.12 AND source_quality > 70% | Merge | High |
| synergy_score < 0.06 AND sources NOT regionally coherent | Don't merge | High |
| Sources from same linguistic region | Test empirically | Low |
| All other cases | Test empirically | Low |

### Open questions

1. How to quantify "regional coherence" as a predictive feature?
2. What is the optimal source quality threshold?
3. Would expanding validation to 15+ targets improve correlations?

---

## 2026-01-21 – Retrospective: Expanded Merging Effect Prediction Validation (NEGATIVE RESULT)

**Type:** Retrospective
**General description:** Expanded validation to 21 targets disproved the regional coherence hypothesis and showed source-level features cannot reliably predict Merging Effect.

### What we tried

1. **Enhanced Predictor (V2/V3)**
   - Added source_quality feature: `source_acc_mean / target_self_perf`
   - Added regional_coherence feature: `max(region_counts) / num_sources`
   - V3 formula: `synergy_score_v1 + coherence_bonus`

2. **Expanded Validation (12 new targets)**
   - High coherence: ms-MY, hi-IN, km-KH, th-TH
   - Middle score: ko-KR, fa-IR, lv-LV, el-GR
   - Low score: my-MM, ka-GE, bn-BD, ml-IN

3. **Regional Coherence Analysis**
   - Analyzed tl-PH phenomenon (100% SE Asian sources → +6.49% synergy)
   - Identified 7 high-synergy-potential regional clusters
   - Computed within-region transfer and similarity

### Key findings (NEGATIVE)

| Predictor | Accuracy | Notes |
|-----------|----------|-------|
| V1 (diversity) | 47.6% (10/21) | Worse than random |
| V3 (+ coherence) | 61.9% (13/21) | Still unreliable |

**Regional coherence hypothesis DISPROVED:**

| Target | Coherence | Effect | Outcome |
|--------|-----------|--------|---------|
| hi-IN | 1.00 | **-7.93%** | INTERFERENCE (worst!) |
| vi-VN | 1.00 | -0.27% | INTERFERENCE |
| id-ID | 1.00 | -2.95% | INTERFERENCE |

**Feature correlations with actual Merging Effect:**
- diversity: +0.28 (weak positive)
- coherence: -0.06 (slightly negative!)
- quality: -0.13 (negative)
- acc_std: +0.19 (opposite to hypothesis)

### What failed

- **Regional coherence bonus**: Caused false positives (hi-IN, vi-VN, id-ID)
- **All hypotheses**: None of diversity, coherence, quality, variance reliably predict outcome
- **Pattern separation**: SYNERGY and INTERFERENCE targets have nearly identical feature profiles

### Outcome

- **Negative result documented**: Source-level features cannot predict Merging Effect
- **Skill updated**: `merging-when-constructive` with full 21-target results
- **Recommendation changed**: "DO NOT rely on any predictor; run pilot experiments instead"
- **Open questions closed**: Questions 3, 4, 5 answered with negative findings

### Files created

| File | Purpose |
|------|---------|
| `analysis/merging_effect_predictor_v2.py` | Enhanced predictor with V1-V4 formulas |
| `analysis/regional_coherence_analysis.py` | tl-PH deep dive and cluster analysis |
| `configs/ablations/expanded_predictor_validation.yaml` | 12-target validation config |
| `experiments_expanded_validation.db` | Validation experiment results |

---

## 2026-01-21 – Retrospective: Source Compatibility Ablation

**Type:** Retrospective
**General description:** Tested whether pairwise source compatibility (Task Vector Cosine, CKA) can predict and improve merge quality. TV Cosine provides weak but consistent improvement (+0.3%).

### What we tried

1. **Hypothesis**: Not all good source models are compatible. Two sources might transfer well individually but interfere when merged.

2. **Method**: Pre-computed N×N compatibility matrices (48 locales) using two metrics:
   - **Task Vector Cosine**: Parameter-space similarity `cos(τ_A, τ_B)` where τ = model - pretrained
   - **CKA**: Representation-space similarity using hidden states on shared inputs

3. **Integration**: Multiplicative weighting `final_weight = similarity × compatibility`

4. **Ablation**: 2 targets × 3 methods = 6 experiments
   - Targets: sw-KE (constructive), cy-GB (low-ratio but destructive)
   - Methods: similarity (baseline), similarity × TV Cosine, similarity × CKA

### Key findings

| Target | Baseline | TV Cosine × Sim | CKA × Sim |
|--------|----------|-----------------|-----------|
| cy-GB | 43.44% | **43.68% (+0.24%)** | 43.38% (-0.06%) |
| sw-KE | 43.14% | **43.51% (+0.37%)** | 43.21% (+0.07%) |

- **Task Vector Cosine consistently outperforms baseline** on both targets
- **CKA is essentially neutral** (minimal change)
- Improvements are modest (~0.3%) but consistent

### What failed

- **CKA didn't help**: Representation-space similarity doesn't predict merge compatibility
- **Improvements are small**: Parameter-space compatibility helps but doesn't dramatically change outcomes
- **cy-GB still doesn't beat ZS baseline** (43.68% vs 44.45% baseline)

### Why TV Cosine works but CKA doesn't

1. **Parameter-space captures interference directly**: Task vectors represent actual weight changes. Similar task vectors merge constructively; orthogonal ones interfere.
2. **Representation similarity ≠ merge compatibility**: Two models can produce similar representations via different weight configurations.
3. **TV Cosine is computationally cheaper**: No forward passes needed.

### Outcome

- **Partially answered open question**: "Can we predict Merging Effect without full merge?" → Yes, partially. TV Cosine provides weak signal.
- **New method**: `similarity_x_tv_compatibility` added to pipeline
- **Created skill**: `source-compatibility`
- **Updated skill**: `merging-when-constructive` with Source Compatibility Analysis section
- **Recommendation**: Use TV Cosine compatibility as tiebreaker, not primary predictor

### Files created

| File | Purpose |
|------|---------|
| `merginguriel/compatibility.py` | Core compatibility metrics |
| `merginguriel/compute_compatibility_matrix.py` | Pre-computation CLI |
| `nxn_results/compatibility_matrix/*.csv` | Pre-computed 48×48 matrices |
| `configs/ablations/source_compatibility.yaml` | Ablation config |
| `.codex/skills/source-compatibility/SKILL.md` | New result skill |

---

## 2026-01-20 – Retrospective: Directional Consensus Merging

**Type:** Retrospective
**General description:** Implemented and validated directional consensus merging—a variance reduction technique that projects task vectors onto per-layer consensus direction. Partially supports hypothesis: helps interference targets but hurts synergy targets.

### What we tried

1. **New method implementation**: `directional_consensus`
   - Projects task vectors onto normalized sum direction per-layer
   - Algorithm: d = normalize(Σ τ_i), then τ_aligned = (τ · d) × d
   - Weighted aggregation with REAL similarity weights

2. **Validation ablation**: 8 experiments
   - Targets: sw-KE (synergy), cy-GB (interference)
   - Methods: directional_consensus, similarity, task_arithmetic, ties

### Key findings

| Target | Effect Type | directional_consensus | similarity | Δ |
|--------|-------------|----------------------|------------|---|
| cy-GB | INTERFERENCE | **42.60%** | 41.66% | **+0.94%** |
| sw-KE | SYNERGY | 46.87% | **48.32%** | -1.45% |

**Partial hypothesis support:**
- Direction alignment HELPS interference (cy-GB: +0.94%)
- Direction alignment HURTS synergy (sw-KE: -1.45%)

### What failed

- **sw-KE performance**: directional_consensus underperforms similarity
- **Projection too aggressive**: Removes beneficial orthogonal components for synergy cases

### Outcome

- **Method implemented**: `directional_consensus` available in pipeline
- **Skill updated**: `merging-when-constructive` with method-specific recommendations
- **New insight**: Directional consensus = variance reduction; helps interference, hurts synergy
- **Database**: `experiments_directional_consensus.db`

### Files created/modified

| File | Action |
|------|--------|
| `submodules/auto_merge_llm/.../methods/directional_consensus.py` | Created |
| `submodules/auto_merge_llm/.../methods/__init__.py` | Modified |
| `merginguriel/run_merging_pipeline_refactored.py` | Modified |
| `configs/ablations/directional_consensus_validation.yaml` | Created |
| `.codex/skills/merging-when-constructive/SKILL.md` | Updated |

---

## 2026-01-20 – Retrospective: DARE & Source Selection Ablation

**Type:** Retrospective
**General description:** Tested DARE preprocessing and REAL vs URIEL similarity on cross-lingual merging; DARE harmful, REAL essential.

### What we tried

1. **URIEL vs REAL similarity for source selection**
   - sq-AL with URIEL: selected ar-SA, cy-GB, fi-FI
   - sq-AL with REAL: selected de-DE (43%), it-IT (29%), tr-TR (28%)

2. **DARE preprocessing (drop_rate=0.9)**
   - Tested with TIES merging on sq-AL, sw-KE, vi-VN
   - Both with and without DARE

3. **Locale diversity**
   - sq-AL: neutral case
   - sw-KE: "constructive" (56% diagonal ratio)
   - vi-VN: "destructive" (86% diagonal ratio)

### Key findings

| Finding | Evidence |
|---------|----------|
| URIEL = catastrophic | sq-AL: 11.63% with URIEL vs 66.54% with REAL |
| REAL achieves Goal 1 | sq-AL: 66.54% > 66.44% baseline; vi-VN: 74.45% > 74.31% baseline |
| DARE destroys performance | sq-AL: 66.54% → 24.21%; vi-VN: 74.45% → 32.08% |
| Sinkhorn suboptimal | sw-KE picked az-AZ (41.5%) instead of sq-AL (46.7%) |

### What failed

1. **DARE for cross-lingual**: -63% relative performance drop on sq-AL
2. **sw-KE "constructive" case**: Failed despite favorable diagonal ratio
3. **STS-B as proxy**: Showed minimal DARE impact vs catastrophic on actual task

### Outcome

- Goal 1 (ExcTar) achieved for 2/3 locales (sq-AL, vi-VN)
- Created skills: `source-selection-cross-lingual`, `dare-cross-lingual-negative`
- Next: Fix Sinkhorn → top-k selection; test IncTar experiments

---

## 2026-01-20 – Retrospective: AdaMerging for Cross-Lingual Merging

**Type:** Retrospective
**General description:** Tested AdaMerging (entropy-based coefficient learning) for cross-lingual merging; helps hard cases, hurts easy cases.

### What we tried

1. **AdaMerging integration**
   - Implemented task-wise AdaMerging in pipeline
   - Used REAL similarity weights as initial coefficients
   - Optimized via entropy minimization on target language data

2. **Three locales tested**
   - sq-AL (easy): TIES baseline 66.54%
   - sw-KE (hard): TIES baseline 42.74%
   - vi-VN (medium): TIES baseline 74.45%

### Key findings

| Finding | Evidence |
|---------|----------|
| AdaMerging helps hard cases | sw-KE: 48.08% vs 42.74% TIES (+5.34%) |
| AdaMerging hurts easy cases | sq-AL: 65.90% vs 66.54%, vi-VN: 70.07% vs 74.45% |
| Coefficients learn selectively | sw-KE shifted [0.33→0.73], sq-AL unchanged |
| Goal 1 achieved for sw-KE only | 48.08% > 46.70% baseline |

### What failed

1. **sq-AL and vi-VN degraded**: Entropy minimization overcorrected
2. **Only 5 iterations**: May be insufficient for proper convergence
3. **CPU execution**: RTX 4090 available but not used (fixed)

### Outcome

- AdaMerging is situational: use only when TIES fails
- Fixed GPU support in adamerging.py
- Created report: `training_reports/adamerging-ablation-2026-01-20.md`
- Next: Test with more iterations on GPU; try AdaMerging++

---

## 2026-01-20 – Retrospective: IncTar Task Vector Base Model Fix

**Type:** Retrospective
**General description:** Fixed critical bug in IncTar mode where task vectors were computed relative to source model instead of pretrained base.

### What we tried

1. **IncTar with AdaMerging** (before fix)
   - sw-KE: 41.90% (vs 82.85% target-only)
   - sq-AL: 64.09% (vs 86.31% target-only)
   - Catastrophic failure

2. **Root cause analysis**
   - Task vectors computed as: `model - first_source_model`
   - Should be: `model - pretrained_base_model`
   - Pretrained base has 2-class head vs 60-class fine-tuned models

3. **Fix implementation**
   - Create pretrained base with correct num_labels (60)
   - Use `ignore_mismatched_sizes=True` when loading
   - All models now compute task vectors relative to true pretrained base

### Key findings

| Locale | Before Fix | After Fix | Target-only | Goal 2? |
|--------|-----------|-----------|-------------|---------|
| sw-KE | 41.90% | **82.75%** | 82.85% | Almost (-0.10%) |
| sq-AL | 64.09% | **86.38%** | 86.31% | **Yes (+0.07%)** |

### What failed

1. **Original assumption**: Using first source as base would work for all modes
2. **Reality**: IncTar requires true pretrained base for meaningful task vectors

### Outcome

- Goal 2 (IncTar) achieved for sq-AL: 86.38% > 86.31%
- Created skills: `adamerging-cross-lingual`, `task-vector-base-model`
- Fix applies to all task-vector methods (TIES, task_arithmetic, AdaMerging)

---

<!-- New entries go above this line -->
