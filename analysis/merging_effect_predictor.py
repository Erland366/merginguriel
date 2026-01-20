"""
Merging Effect Predictor

Goal: Predict synergy/interference for ALL 49 locales before running merges.

Key Hypotheses (from analysis):
H1: Higher source diversity → more synergy
H2: Higher source accuracy variance → more interference

Proposed Prediction Score:
  synergy_score = diversity_score / (1 + source_acc_std)
  Higher score = more likely to achieve synergy

This script:
1. Computes features for all 49 locales
2. Ranks them by predicted synergy score
3. Identifies high-potential targets for merging
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from merginguriel.similarity_utils import load_and_process_similarity, load_similarity_matrix

# Paths
NXN_MATRIX_PATH = PROJECT_ROOT / "nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv"
SIMILARITY_MATRIX_PATH = PROJECT_ROOT / "language_similarity_matrix_unified.csv"


def load_nxn_matrix() -> pd.DataFrame:
    """Load the NxN evaluation matrix."""
    return pd.read_csv(NXN_MATRIX_PATH, index_col=0)


def get_pipeline_sources(target: str, num_languages: int = 5) -> List[Tuple[str, float]]:
    """Get sources as pipeline would select them."""
    return load_and_process_similarity(
        str(SIMILARITY_MATRIX_PATH),
        target,
        num_languages=num_languages,
        top_k=20,
        sinkhorn_iterations=20,
        include_target=False,
        verbose=False
    )


def compute_features(
    target: str,
    sources: List[str],
    nxn_matrix: pd.DataFrame,
    raw_similarity_matrix: pd.DataFrame
) -> Dict[str, float]:
    """Compute predictive features for a target locale."""

    # Source accuracy features
    source_accuracies = [nxn_matrix.loc[src, target] for src in sources]
    source_acc_mean = np.mean(source_accuracies)
    source_acc_std = np.std(source_accuracies)
    source_acc_max = np.max(source_accuracies)

    # Self-performance (diagonal of NxN)
    self_perf = nxn_matrix.loc[target, target]

    # ZS/Self ratio
    zs_self_ratio = source_acc_max / self_perf if self_perf > 0 else 0

    # Source pairwise similarity
    pairwise_sims = []
    for i, src1 in enumerate(sources):
        for j, src2 in enumerate(sources):
            if i < j:
                pairwise_sims.append(raw_similarity_matrix.loc[src1, src2])

    source_pairwise_sim_mean = np.mean(pairwise_sims) if pairwise_sims else 0
    diversity_score = 1 - source_pairwise_sim_mean

    # Source-target similarity
    source_target_sims = [raw_similarity_matrix.loc[src, target] for src in sources]
    source_target_sim_mean = np.mean(source_target_sims)

    # Predicted synergy score
    # Based on hypotheses: high diversity + low accuracy variance = synergy
    synergy_score = diversity_score / (1 + source_acc_std * 10)  # Scale std

    # Alternative: weighted combination
    # synergy_score_v2 = diversity_score * 0.6 - source_acc_std * 0.4

    return {
        'target': target,
        'sources': sources,
        'self_perf': self_perf,
        'best_source_perf': source_acc_max,
        'zs_self_ratio': zs_self_ratio,
        'source_acc_mean': source_acc_mean,
        'source_acc_std': source_acc_std,
        'source_pairwise_sim_mean': source_pairwise_sim_mean,
        'diversity_score': diversity_score,
        'source_target_sim_mean': source_target_sim_mean,
        'synergy_score': synergy_score,
        # Expected merged performance (rough estimate)
        'expected_merged': source_acc_mean,
    }


def main():
    print("=" * 80)
    print("MERGING EFFECT PREDICTOR - All 49 Locales")
    print("=" * 80)

    # Load data
    nxn = load_nxn_matrix()
    raw_sim = load_similarity_matrix(str(SIMILARITY_MATRIX_PATH), verbose=False)

    locales = list(nxn.index)
    print(f"\nAnalyzing {len(locales)} locales...")

    # Compute features for all locales
    all_features = []
    for target in locales:
        try:
            sources = get_pipeline_sources(target, num_languages=5)
            source_locales = [loc for loc, _ in sources]
            features = compute_features(target, source_locales, nxn, raw_sim)
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {target}: {e}")

    df = pd.DataFrame(all_features)

    # Rank by synergy score
    df = df.sort_values('synergy_score', ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 TARGETS (Highest Predicted Synergy)")
    print("=" * 80)
    print("\nThese targets are predicted to benefit most from merging:\n")

    top10 = df.head(10)
    display_cols = ['target', 'synergy_score', 'diversity_score', 'source_acc_std',
                    'zs_self_ratio', 'self_perf', 'best_source_perf']
    print(top10[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("BOTTOM 10 TARGETS (Lowest Predicted Synergy - Likely Interference)")
    print("=" * 80)
    print("\nThese targets should AVOID merging (stick with best source):\n")

    bottom10 = df.tail(10)
    print(bottom10[display_cols].to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)

    # Define thresholds
    high_synergy = df[df['synergy_score'] > df['synergy_score'].quantile(0.75)]
    low_synergy = df[df['synergy_score'] < df['synergy_score'].quantile(0.25)]

    print(f"\nHigh synergy candidates ({len(high_synergy)} locales):")
    print(f"  Avg diversity: {high_synergy['diversity_score'].mean():.4f}")
    print(f"  Avg acc std:   {high_synergy['source_acc_std'].mean():.4f}")
    print(f"  Avg ZS/Self:   {high_synergy['zs_self_ratio'].mean():.4f}")

    print(f"\nLow synergy candidates ({len(low_synergy)} locales):")
    print(f"  Avg diversity: {low_synergy['diversity_score'].mean():.4f}")
    print(f"  Avg acc std:   {low_synergy['source_acc_std'].mean():.4f}")
    print(f"  Avg ZS/Self:   {low_synergy['zs_self_ratio'].mean():.4f}")

    # Known validation cases
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST KNOWN RESULTS")
    print("=" * 80)

    known = {
        'sw-KE': {'actual_effect': '+9.06%', 'outcome': 'SYNERGY'},
        'cy-GB': {'actual_effect': '+1.30%', 'outcome': 'SYNERGY'},
        'vi-VN': {'actual_effect': '-0.27%', 'outcome': 'INTERFERENCE'},
    }

    for locale, info in known.items():
        row = df[df['target'] == locale].iloc[0]
        rank = df['target'].tolist().index(locale) + 1
        print(f"\n{locale}:")
        print(f"  Actual: {info['actual_effect']} ({info['outcome']})")
        print(f"  Predicted synergy score: {row['synergy_score']:.4f}")
        print(f"  Rank: {rank}/49 (1=highest synergy)")
        print(f"  Diversity: {row['diversity_score']:.4f}")
        print(f"  Acc Std: {row['source_acc_std']:.4f}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("FEATURE CORRELATIONS WITH SYNERGY SCORE")
    print("=" * 80)

    numeric_cols = ['diversity_score', 'source_acc_std', 'source_target_sim_mean',
                    'zs_self_ratio', 'source_pairwise_sim_mean']
    for col in numeric_cols:
        corr = df['synergy_score'].corr(df[col])
        print(f"  {col:30s}: {corr:+.4f}")

    # Export for further analysis
    output_path = PROJECT_ROOT / "analysis" / "merging_effect_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. HIGH PRIORITY for merging (top 5 by synergy score):")
    for _, row in df.head(5).iterrows():
        print(f"   - {row['target']}: synergy_score={row['synergy_score']:.4f}")

    print("\n2. AVOID merging (bottom 5 - use best single source instead):")
    for _, row in df.tail(5).iterrows():
        print(f"   - {row['target']}: synergy_score={row['synergy_score']:.4f}")

    print("\n3. Proposed validation experiment:")
    print("   Run merging on top 5 and bottom 5 predicted targets")
    print("   Measure actual Merging Effect to validate predictor")

    return df


if __name__ == "__main__":
    main()
