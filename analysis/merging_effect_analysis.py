"""
Merging Effect Prediction Analysis

Research Question:
Can we predict whether merging sources will result in SYNERGY or INTERFERENCE
before running the expensive merge operation?

Key Insight (from ablations):
- sw-KE: +2.30% Merging Effect (SYNERGY)
- cy-GB: -1.58% Merging Effect (INTERFERENCE)
- Both have similar ZS/Self ratios (~53-56%), yet behave oppositely

Merging Effect = merged_accuracy - avg(source_accuracies_on_target)

Goal: Find features that predict Merging Effect from source-level data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from merginguriel.similarity_utils import load_and_process_similarity, load_similarity_matrix

# Paths
NXN_MATRIX_PATH = PROJECT_ROOT / "nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv"
SIMILARITY_MATRIX_PATH = PROJECT_ROOT / "language_similarity_matrix_unified.csv"


def load_nxn_matrix() -> pd.DataFrame:
    """Load the NxN evaluation matrix (source x target -> accuracy)."""
    df = pd.read_csv(NXN_MATRIX_PATH, index_col=0)
    return df


def get_pipeline_sources(target: str, num_languages: int = 5) -> List[Tuple[str, float]]:
    """Get sources exactly as the pipeline would select them."""
    return load_and_process_similarity(
        str(SIMILARITY_MATRIX_PATH),
        target,
        num_languages=num_languages,
        top_k=20,
        sinkhorn_iterations=20,
        include_target=False,
        verbose=False
    )


def compute_source_features(
    target: str,
    sources: List[str],
    source_weights: List[float],
    nxn_matrix: pd.DataFrame,
    raw_similarity_matrix: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute features that might predict Merging Effect.

    Features:
    1. Source accuracy features (from NxN matrix)
    2. Source-to-source similarity features (raw REAL)
    3. Source-to-target similarity features
    4. Weight features (from Sinkhorn-processed similarity)
    """
    features = {}

    # 1. Source accuracy features (how well do sources perform on target?)
    source_accuracies = [nxn_matrix.loc[src, target] for src in sources]
    features['source_acc_mean'] = np.mean(source_accuracies)
    features['source_acc_std'] = np.std(source_accuracies)
    features['source_acc_max'] = np.max(source_accuracies)
    features['source_acc_min'] = np.min(source_accuracies)
    features['source_acc_range'] = features['source_acc_max'] - features['source_acc_min']

    # Weighted accuracy (using Sinkhorn weights)
    total_weight = sum(source_weights)
    if total_weight > 0:
        features['source_acc_weighted'] = sum(
            acc * w for acc, w in zip(source_accuracies, source_weights)
        ) / total_weight
    else:
        features['source_acc_weighted'] = features['source_acc_mean']

    # 2. Source-to-source pairwise similarity (are sources similar to each other?)
    pairwise_sims = []
    for i, src1 in enumerate(sources):
        for j, src2 in enumerate(sources):
            if i < j:  # Upper triangle only
                pairwise_sims.append(raw_similarity_matrix.loc[src1, src2])

    features['source_pairwise_sim_mean'] = np.mean(pairwise_sims) if pairwise_sims else 0
    features['source_pairwise_sim_std'] = np.std(pairwise_sims) if pairwise_sims else 0
    features['source_pairwise_sim_min'] = np.min(pairwise_sims) if pairwise_sims else 0
    features['source_pairwise_sim_max'] = np.max(pairwise_sims) if pairwise_sims else 0

    # 3. Source-to-target similarity (how similar are sources to target?)
    source_target_sims = [raw_similarity_matrix.loc[src, target] for src in sources]
    features['source_target_sim_mean'] = np.mean(source_target_sims)
    features['source_target_sim_std'] = np.std(source_target_sims)
    features['source_target_sim_min'] = np.min(source_target_sims)
    features['source_target_sim_max'] = np.max(source_target_sims)

    # 4. Derived features
    # Hypothesis: High source diversity + high target similarity = synergy
    features['diversity_score'] = 1 - features['source_pairwise_sim_mean']
    features['target_alignment'] = features['source_target_sim_mean']
    features['diversity_x_alignment'] = features['diversity_score'] * features['target_alignment']

    # Weight concentration (how evenly distributed are the weights?)
    weight_sum = sum(source_weights)
    if weight_sum > 0:
        normalized_weights = [w / weight_sum for w in source_weights]
        features['weight_entropy'] = -sum(
            w * np.log(w + 1e-10) for w in normalized_weights
        )
        features['weight_max'] = max(normalized_weights)
        features['weight_concentration'] = max(normalized_weights) / np.mean(normalized_weights)
    else:
        features['weight_entropy'] = 0
        features['weight_max'] = 0
        features['weight_concentration'] = 1

    # Coefficient of variation (normalized std)
    if features['source_acc_mean'] > 0:
        features['source_acc_cv'] = features['source_acc_std'] / features['source_acc_mean']
    else:
        features['source_acc_cv'] = 0

    return features


def compute_merging_effect(
    target: str,
    sources: List[str],
    nxn_matrix: pd.DataFrame,
    actual_merged_accuracy: float
) -> Dict[str, float]:
    """
    Compute expected accuracy (average of sources) vs actual merged accuracy.

    Merging Effect = actual - expected
    """
    source_accuracies = [nxn_matrix.loc[src, target] for src in sources]
    expected = np.mean(source_accuracies)

    return {
        'expected_accuracy': expected,
        'actual_merged_accuracy': actual_merged_accuracy,
        'merging_effect': actual_merged_accuracy - expected,
        'merging_effect_pct': (actual_merged_accuracy - expected) * 100,
        'source_accuracies': dict(zip(sources, source_accuracies))
    }


def analyze_target(
    target: str,
    nxn_matrix: pd.DataFrame,
    raw_similarity_matrix: pd.DataFrame,
    actual_merged_accuracy: float = None,
    n_sources: int = 5
) -> Dict:
    """Full analysis for a target locale."""
    # Get sources exactly as pipeline would select them
    source_weights = get_pipeline_sources(target, n_sources)
    sources = [loc for loc, _ in source_weights]
    weights = [w for _, w in source_weights]

    features = compute_source_features(
        target, sources, weights, nxn_matrix, raw_similarity_matrix
    )

    result = {
        'target': target,
        'sources': sources,
        'source_weights': dict(source_weights),
        **features
    }

    if actual_merged_accuracy is not None:
        effect = compute_merging_effect(target, sources, nxn_matrix, actual_merged_accuracy)
        result.update(effect)

    return result


def main():
    print("=" * 70)
    print("MERGING EFFECT PREDICTION ANALYSIS (CORRECTED)")
    print("=" * 70)

    # Load data
    nxn = load_nxn_matrix()

    # Load raw similarity matrix (before Sinkhorn)
    raw_sim = load_similarity_matrix(str(SIMILARITY_MATRIX_PATH), verbose=False)

    print(f"\nLoaded NxN matrix: {nxn.shape}")
    print(f"Loaded Raw Similarity matrix: {raw_sim.shape}")

    # Known results from ablations (REAL similarity, 5 languages, ExcTar)
    known_results = {
        'sw-KE': 0.4832,  # Synergy expected
        'cy-GB': 0.4166,  # Interference expected
        'vi-VN': 0.6769,  # Destructive target (similarity method)
    }

    print("\n" + "=" * 70)
    print("ANALYSIS OF KNOWN TARGETS (Using actual pipeline sources)")
    print("=" * 70)

    results = []
    for target, actual_acc in known_results.items():
        result = analyze_target(target, nxn, raw_sim, actual_merged_accuracy=actual_acc)
        results.append(result)

        print(f"\n--- {target} ---")
        print(f"Sources (as selected by pipeline):")
        for src, w in result['source_weights'].items():
            acc = result['source_accuracies'].get(src, 'N/A')
            print(f"  {src}: weight={w:.4f}, acc_on_target={acc:.4f}")

        print(f"\nSource Accuracy Features:")
        print(f"  Mean:     {result['source_acc_mean']:.4f}")
        print(f"  Weighted: {result['source_acc_weighted']:.4f}")
        print(f"  Std:      {result['source_acc_std']:.4f}")
        print(f"  Range:    {result['source_acc_range']:.4f}")

        print(f"\nSource Pairwise Similarity (raw REAL):")
        print(f"  Mean:  {result['source_pairwise_sim_mean']:.4f}")
        print(f"  Std:   {result['source_pairwise_sim_std']:.4f}")

        print(f"\nSource-Target Similarity (raw REAL):")
        print(f"  Mean:  {result['source_target_sim_mean']:.4f}")
        print(f"  Std:   {result['source_target_sim_std']:.4f}")

        print(f"\nDerived Features:")
        print(f"  Diversity Score:     {result['diversity_score']:.4f}")
        print(f"  Target Alignment:    {result['target_alignment']:.4f}")
        print(f"  Weight Concentration: {result['weight_concentration']:.4f}")

        if 'merging_effect' in result:
            print(f"\nMerging Effect:")
            print(f"  Expected (avg sources): {result['expected_accuracy']:.4f}")
            print(f"  Actual (merged):        {result['actual_merged_accuracy']:.4f}")
            print(f"  Merging Effect:         {result['merging_effect_pct']:+.2f}%")
            if result['merging_effect'] > 0:
                print(f"  Outcome: SYNERGY ✓")
            else:
                print(f"  Outcome: INTERFERENCE ✗")

    # Comparison table
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON TABLE")
    print("=" * 70)

    df = pd.DataFrame(results)
    display_cols = [
        'target', 'source_acc_mean', 'source_acc_std',
        'source_pairwise_sim_mean', 'source_target_sim_mean',
        'diversity_score', 'merging_effect_pct'
    ]
    print(df[display_cols].to_string(index=False))

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    sw_ke = [r for r in results if r['target'] == 'sw-KE'][0]
    cy_gb = [r for r in results if r['target'] == 'cy-GB'][0]
    vi_vn = [r for r in results if r['target'] == 'vi-VN'][0]

    print("\nComparing sw-KE vs cy-GB vs vi-VN:")

    print(f"\n1. Expected Accuracy (avg of sources on target):")
    print(f"   sw-KE: {sw_ke['expected_accuracy']:.4f}")
    print(f"   cy-GB: {cy_gb['expected_accuracy']:.4f}")
    print(f"   vi-VN: {vi_vn['expected_accuracy']:.4f}")

    print(f"\n2. Merging Effect (actual - expected):")
    print(f"   sw-KE: {sw_ke['merging_effect_pct']:+.2f}% → {'SYNERGY' if sw_ke['merging_effect'] > 0 else 'INTERFERENCE'}")
    print(f"   cy-GB: {cy_gb['merging_effect_pct']:+.2f}% → {'SYNERGY' if cy_gb['merging_effect'] > 0 else 'INTERFERENCE'}")
    print(f"   vi-VN: {vi_vn['merging_effect_pct']:+.2f}% → {'SYNERGY' if vi_vn['merging_effect'] > 0 else 'INTERFERENCE'}")

    print(f"\n3. Source Pairwise Similarity (are sources similar to EACH OTHER?):")
    print(f"   sw-KE: {sw_ke['source_pairwise_sim_mean']:.4f}")
    print(f"   cy-GB: {cy_gb['source_pairwise_sim_mean']:.4f}")
    print(f"   vi-VN: {vi_vn['source_pairwise_sim_mean']:.4f}")

    print(f"\n4. Diversity Score (1 - pairwise_sim):")
    print(f"   sw-KE: {sw_ke['diversity_score']:.4f}")
    print(f"   cy-GB: {cy_gb['diversity_score']:.4f}")
    print(f"   vi-VN: {vi_vn['diversity_score']:.4f}")

    print(f"\n5. Source Accuracy Standard Deviation:")
    print(f"   sw-KE: {sw_ke['source_acc_std']:.4f}")
    print(f"   cy-GB: {cy_gb['source_acc_std']:.4f}")
    print(f"   vi-VN: {vi_vn['source_acc_std']:.4f}")

    # Hypothesis testing
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    synergy_targets = [r for r in results if r['merging_effect'] > 0]
    interference_targets = [r for r in results if r['merging_effect'] <= 0]

    if synergy_targets and interference_targets:
        avg_div_synergy = np.mean([r['diversity_score'] for r in synergy_targets])
        avg_div_interference = np.mean([r['diversity_score'] for r in interference_targets])

        print(f"\nH1: Higher source diversity → more synergy")
        print(f"    Avg diversity (synergy targets):      {avg_div_synergy:.4f}")
        print(f"    Avg diversity (interference targets): {avg_div_interference:.4f}")
        if avg_div_synergy > avg_div_interference:
            print("    SUPPORTED: Synergy targets have higher diversity")
        else:
            print("    NOT SUPPORTED")

        avg_std_synergy = np.mean([r['source_acc_std'] for r in synergy_targets])
        avg_std_interference = np.mean([r['source_acc_std'] for r in interference_targets])

        print(f"\nH2: Higher source accuracy variance → more interference (conflicting sources)")
        print(f"    Avg acc std (synergy targets):      {avg_std_synergy:.4f}")
        print(f"    Avg acc std (interference targets): {avg_std_interference:.4f}")
        if avg_std_interference > avg_std_synergy:
            print("    SUPPORTED: Interference targets have higher accuracy variance")
        else:
            print("    NOT SUPPORTED")

    return results


if __name__ == "__main__":
    main()
