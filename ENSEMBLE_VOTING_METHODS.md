# Ensemble Voting Methods in MergingUriel

This document explains the four voting techniques used in our URIEL-guided ensemble inference system. These methods combine predictions from multiple language models trained on different source languages to make predictions for a target language.

## Overview

The ensemble inference system combines predictions from multiple models trained on different source languages. Each voting method represents a different strategy for combining these predictions:

1. **Majority Voting** - Simple majority rule
2. **Weighted Majority Voting** - URIEL-weighted votes
3. **Soft Voting** - Probability averaging
4. **URIEL Weighted Logits** - Our core innovation

---

## 1. Majority Voting

**Simple majority rule with equal model weights**

```python
def majority_vote(predictions: List[int]) -> int:
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]
```

### How it works:
1. Each model gets 1 vote (equal weighting)
2. Count votes for each class
3. Class with most votes wins
4. **All models treated equally regardless of language similarity**

### Example:
```
5 models predict: [2, 2, 1, 2, 1]
Votes: Class 2 = 3, Class 1 = 2
→ Class 2 wins (majority)
```

### Characteristics:
- ✅ Simple and robust
- ✅ No weighting complexity
- ❌ Ignores linguistic similarity information
- ❌ All models treated equally

### Best for:
- Baseline comparison
- When all source languages are equally relevant
- Robustness testing

---

## 2. Weighted Majority Voting

**URIEL similarity-weighted votes**

```python
def weighted_majority_vote(predictions: List[int], weights: List[float]) -> int:
    weighted_counts = {}
    for pred, weight in zip(predictions, weights):
        weighted_counts[pred] = weighted_counts.get(pred, 0) + weight
    return max(weighted_counts.items(), key=lambda x: x[1])[0]
```

### How it works:
1. Models get different vote weights based on URIEL similarity to target
2. Weight = similarity score (normalized to sum to 1.0)
3. **More typologically similar languages get stronger votes**
4. Sum weighted votes for each class
5. Class with highest weighted vote count wins

### Example:
```
Target: sq-AL (Albanian)
Source models and weights:
- ar-SA (Arabic): 0.30 similarity → predicts class 2 → 0.30 votes to class 2
- fr-FR (French): 0.15 similarity → predicts class 2 → 0.15 votes to class 2
- cy-GB (Welsh): 0.20 similarity → predicts class 1 → 0.20 votes to class 1
- lv-LV (Latvian): 0.18 similarity → predicts class 1 → 0.18 votes to class 1
- fi-FI (Finnish): 0.17 similarity → predicts class 3 → 0.17 votes to class 3

Weighted totals: Class 1 = 0.38, Class 2 = 0.45, Class 3 = 0.17
→ Class 2 wins (highest weighted votes)
```

### Characteristics:
- ✅ Incorporates linguistic similarity
- ✅ Intuitive weighting scheme
- ✅ Better than simple majority for cross-lingual transfer
- ❌ Only uses final predictions, not confidence levels

### Best for:
- When some source languages are clearly more similar to target
- Cross-lingual scenarios with varying typological distances
- Balancing model influence based on linguistic relevance

---

## 3. Soft Voting

**Probability distribution averaging**

```python
def soft_vote(probabilities: List[torch.Tensor], weights: List[float] = None) -> int:
    avg_prob = torch.zeros_like(probabilities[0])
    for prob, weight in zip(probabilities, weights):
        avg_prob += prob * weight
    return avg_prob.argmax().item()
```

### How it works:
1. Extract probability distributions from each model (post-softmax)
2. Average probabilities across all models
3. Can be unweighted or URIEL-weighted
4. Class with highest average probability wins
5. **Preserves model confidence information**

### Example (unweighted):
```
3 models, 3 intent classes:
Model 1: [0.1, 0.7, 0.2] (confident in class 2)
Model 2: [0.6, 0.3, 0.1] (confident in class 1)
Model 3: [0.4, 0.4, 0.2] (uncertain between classes 1&2)

Average: [(0.1+0.6+0.4)/3, (0.7+0.3+0.4)/3, (0.2+0.1+0.2)/3]
        = [0.37, 0.47, 0.17]
→ Class 2 wins (highest average probability)
```

### Characteristics:
- ✅ Uses full probability distributions
- ✅ Preserves model confidence levels
- ✅ Can incorporate URIEL weighting
- ✅ Theoretically sound ensemble method
- ❌ More computationally expensive
- ❌ May be affected by overconfident models

### Best for:
- When models have different confidence levels
- Uncertainty-aware predictions
- Theoretically grounded ensemble approaches

---

## 4. URIEL Weighted Logits ⭐

**Our core innovation - logit-level weighting with URIEL similarity**

```python
def uriel_weighted_logits(logits_list: List[torch.Tensor], weights: List[float]) -> int:
    weighted_logits = torch.zeros_like(logits_list[0])
    for logits, weight in zip(logits_list, weights):
        weighted_logits += logits * weight  # Multiply by URIEL similarity
    return weighted_logits.argmax().item()
```

### How it works:
1. **Extract raw logits** from each model (pre-softmax activation scores)
2. **Multiply logits** by URIEL similarity score for each model
3. **Sum weighted logits** across all models
4. **Apply argmax** to get final prediction
5. **Preserves both model confidence and linguistic similarity**

### Mathematical formulation:
```
Let:
- L_i = logits from model i
- w_i = URIEL similarity weight for model i
- C = number of classes

Final weighted logits = Σ(w_i × L_i) for i = 1 to n
Final prediction = argmax(Σ(w_i × L_i))
```

### Example:
```
Target: sq-AL (Albanian)

Model A (ar-SA, weight=0.30): logits [1.2, -0.5, 0.8]
→ Weighted: [0.36, -0.15, 0.24]

Model B (fr-FR, weight=0.15): logits [-0.3, 2.1, 0.4]
→ Weighted: [-0.045, 0.315, 0.06]

Model C (cy-GB, weight=0.20): logits [0.8, 1.1, -0.2]
→ Weighted: [0.16, 0.22, -0.04]

Sum of weighted logits: [0.475, 0.385, 0.26]
→ Class 1 wins (argmax)
```

### Characteristics:
- ✅ **Uses raw logits** - preserves pre-softmax confidence
- ✅ **URIEL-weighted** - incorporates linguistic similarity
- ✅ **Maintains distribution** - keeps probabilistic nature
- ✅ **Theoretically grounded** - based on ensemble theory
- ✅ **Our core approach** - designed for cross-lingual transfer
- ❌ Most computationally complex
- ❌ Requires careful weight normalization

### Why it's special:
This method combines the strengths of multiple approaches:
- **Like soft voting**: Uses full probability information (via logits)
- **Like weighted majority**: Incorporates URIEL similarity weighting
- **Beyond both**: Works at logit level, preserving model confidence before softmax transformation

### Best for:
- **Cross-lingual transfer** where linguistic similarity matters
- **Multi-class problems** with complex decision boundaries
- **When confidence levels vary** significantly between models
- **Research scenarios** exploring linguistic similarity effects

---

## Comparative Summary

| Method | Input | Weighting | Preserves Confidence | URIEL-Aware | Complexity | Best For |
|--------|-------|-----------|---------------------|-------------|------------|-----------|
| **Majority** | Final predictions | Equal | ❌ | ❌ | Low | Simple baseline |
| **Weighted Majority** | Final predictions | URIEL similarity | ❌ | ✅ | Medium | Unequal language relevance |
| **Soft** | Probabilities | Equal/URIEL | ✅ | ✅ | High | Varying confidence levels |
| **URIEL Logits** | Raw logits | URIEL similarity | ✅ | ✅ | Highest | Cross-lingual transfer |

## Practical Recommendations

### For Research and Cross-Lingual Transfer:
- **Primary choice**: `urie_logits` - our core innovation designed for this scenario
- **Secondary choice**: `weighted_majority` - simpler but still incorporates linguistic similarity

### For Baseline Comparison:
- Always include `majority` - provides simple baseline
- Include `soft` - shows effect of confidence information

### For Production Systems:
- `urie_logits` if computational resources allow
- `weighted_majority` for simpler implementation
- `majority` for robustness and speed

### For Error Analysis:
- Compare all methods to understand:
  - When linguistic similarity helps (urie_logits vs majority)
  - When confidence matters (soft vs majority)
  - When simple voting suffices (majority vs weighted)

## Implementation Notes

All four methods are implemented in `merginguriel/uriel_ensemble_inference.py` and can be selected using the `--voting-method` parameter:

```bash
# Use our core innovation
python merginguriel/uriel_ensemble_inference.py \
    --target-lang sq-AL \
    --voting-method urie_logits

# Compare with baselines
python merginguriel/uriel_ensemble_inference.py \
    --target-lang sq-AL \
    --voting-method majority

# Test all methods with large-scale runner
python merginguriel/run_large_scale_ensemble_experiments.py \
    --voting-methods majority weighted_majority soft urie_logits
```

The large-scale ensemble experiment runner automatically tests all voting methods across multiple target languages, providing comprehensive comparison of their effectiveness in different cross-lingual scenarios.