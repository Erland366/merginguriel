"""
Merging Effect Predictor V2

Enhancements over V1:
- Source quality feature: penalize when sources are much weaker than target (af-ZA fix)
- Regional coherence feature: reward geographic clustering of sources (tl-PH fix)

Key Hypotheses:
H1: Higher source diversity → more synergy
H2: Higher source accuracy variance → more interference
H3: Source quality relative to target matters (NEW)
H4: Regional coherence enables synergy despite low diversity (NEW)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from merginguriel.similarity_utils import load_and_process_similarity, load_similarity_matrix

# Paths
NXN_MATRIX_PATH = PROJECT_ROOT / "nxn_results/nxn_eval_20251027_103544/evaluation_matrix.csv"
SIMILARITY_MATRIX_PATH = PROJECT_ROOT / "language_similarity_matrix_unified.csv"

# Regional mapping for locales
LOCALE_REGIONS = {
    # Southeast Asian
    'vi-VN': 'SE_Asian', 'th-TH': 'SE_Asian', 'id-ID': 'SE_Asian',
    'ms-MY': 'SE_Asian', 'tl-PH': 'SE_Asian', 'jv-ID': 'SE_Asian',
    'km-KH': 'SE_Asian', 'my-MM': 'SE_Asian',
    # East Asian
    'zh-CN': 'E_Asian', 'zh-TW': 'E_Asian', 'ja-JP': 'E_Asian',
    'ko-KR': 'E_Asian',
    # South Asian
    'hi-IN': 'S_Asian', 'bn-BD': 'S_Asian', 'ta-IN': 'S_Asian',
    'te-IN': 'S_Asian', 'kn-IN': 'S_Asian', 'ml-IN': 'S_Asian',
    'ur-PK': 'S_Asian', 'nb-NO': 'Nordic',
    # Middle Eastern
    'ar-SA': 'Middle_East', 'he-IL': 'Middle_East', 'fa-IR': 'Middle_East',
    'tr-TR': 'Middle_East',
    # African
    'sw-KE': 'African', 'am-ET': 'African', 'af-ZA': 'African',
    # Western European
    'en-US': 'W_European', 'de-DE': 'W_European', 'fr-FR': 'W_European',
    'es-ES': 'W_European', 'pt-PT': 'W_European', 'it-IT': 'W_European',
    'nl-NL': 'W_European', 'da-DK': 'Nordic', 'sv-SE': 'Nordic',
    'fi-FI': 'Nordic', 'is-IS': 'Nordic',
    # Eastern European
    'pl-PL': 'E_European', 'ru-RU': 'E_European', 'uk-UA': 'E_European',
    'cs-CZ': 'E_European', 'sk-SK': 'E_European', 'sl-SI': 'E_European',
    'hr-HR': 'E_European', 'hu-HU': 'E_European', 'ro-RO': 'E_European',
    'bg-BG': 'E_European', 'sr-RS': 'E_European',
    # Baltic
    'lv-LV': 'Baltic', 'lt-LT': 'Baltic', 'et-EE': 'Baltic',
    # Other European
    'el-GR': 'S_European', 'sq-AL': 'S_European', 'cy-GB': 'Celtic',
    'ga-IE': 'Celtic', 'ca-ES': 'W_European',
    # Caucasian
    'hy-AM': 'Caucasian', 'ka-GE': 'Caucasian', 'az-AZ': 'Caucasian',
}


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


def compute_regional_coherence(sources: List[str]) -> Dict[str, float]:
    """Compute regional coherence metrics for sources."""
    regions = [LOCALE_REGIONS.get(src, 'Unknown') for src in sources]
    region_counts = Counter(regions)

    most_common_region, most_common_count = region_counts.most_common(1)[0]

    # Coherence = fraction of sources from the same region
    coherence = most_common_count / len(sources)

    # Number of unique regions
    num_regions = len(region_counts)

    return {
        'regional_coherence': coherence,
        'dominant_region': most_common_region,
        'num_regions': num_regions,
        'source_regions': regions,
    }


def compute_features(
    target: str,
    sources: List[str],
    nxn_matrix: pd.DataFrame,
    raw_similarity_matrix: pd.DataFrame
) -> Dict[str, float]:
    """Compute predictive features for a target locale (V2 with new features)."""

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

    # NEW: Source quality relative to target
    source_quality = source_acc_mean / self_perf if self_perf > 0 else 0

    # NEW: Regional coherence
    regional_info = compute_regional_coherence(sources)
    regional_coherence = regional_info['regional_coherence']
    dominant_region = regional_info['dominant_region']
    num_regions = regional_info['num_regions']

    # Target region
    target_region = LOCALE_REGIONS.get(target, 'Unknown')

    # NEW: Regional alignment (do sources match target's region?)
    sources_in_target_region = sum(1 for src in sources if LOCALE_REGIONS.get(src) == target_region)
    regional_alignment = sources_in_target_region / len(sources)

    # ===== PREDICTOR FORMULAS =====

    # V1: Original (baseline)
    synergy_score_v1 = diversity_score / (1 + source_acc_std * 10)

    # V2: Add source quality penalty
    # If sources are much weaker than target, penalize
    quality_factor = min(source_quality / 0.7, 1.0)  # Threshold at 70%
    synergy_score_v2 = synergy_score_v1 * quality_factor

    # V3: Add regional coherence bonus
    # High coherence can compensate for low diversity
    coherence_bonus = regional_coherence * 0.5 if regional_coherence >= 0.8 else 0
    synergy_score_v3 = synergy_score_v2 + coherence_bonus

    # V4: Combined formula
    # diversity matters, but quality and coherence can override
    synergy_score_v4 = (
        diversity_score * quality_factor * 0.6 +
        regional_coherence * 0.3 -
        source_acc_std * 2
    )

    return {
        'target': target,
        'target_region': target_region,
        'sources': sources,
        'source_regions': regional_info['source_regions'],
        'self_perf': self_perf,
        'best_source_perf': source_acc_max,
        'zs_self_ratio': zs_self_ratio,
        'source_acc_mean': source_acc_mean,
        'source_acc_std': source_acc_std,
        'source_pairwise_sim_mean': source_pairwise_sim_mean,
        'diversity_score': diversity_score,
        'source_target_sim_mean': source_target_sim_mean,
        # NEW features
        'source_quality': source_quality,
        'regional_coherence': regional_coherence,
        'dominant_region': dominant_region,
        'num_regions': num_regions,
        'regional_alignment': regional_alignment,
        # Multiple predictor versions
        'synergy_score_v1': synergy_score_v1,  # Original
        'synergy_score_v2': synergy_score_v2,  # + quality
        'synergy_score_v3': synergy_score_v3,  # + quality + coherence bonus
        'synergy_score_v4': synergy_score_v4,  # Combined
        # Default to V3 for ranking
        'synergy_score': synergy_score_v3,
        # Expected merged performance (rough estimate)
        'expected_merged': source_acc_mean,
    }


def analyze_known_results(df: pd.DataFrame):
    """Analyze predictor performance on known validation results."""

    # All known results from validation experiments
    known_results = {
        # Original 3
        'sw-KE': {'merged': 0.4832, 'expected': 0.4602, 'effect': +2.30, 'outcome': 'SYNERGY'},
        'cy-GB': {'merged': 0.4166, 'expected': 0.4324, 'effect': -1.58, 'outcome': 'INTERFERENCE'},
        'vi-VN': {'merged': 0.6769, 'expected': 0.6796, 'effect': -0.27, 'outcome': 'INTERFERENCE'},
        # Validation 6
        'az-AZ': {'merged': 0.6627, 'expected': 0.6371, 'effect': +2.57, 'outcome': 'SYNERGY'},
        'tr-TR': {'merged': 0.7290, 'expected': 0.6883, 'effect': +4.07, 'outcome': 'SYNERGY'},
        'af-ZA': {'merged': 0.5740, 'expected': 0.6018, 'effect': -2.78, 'outcome': 'INTERFERENCE'},
        'am-ET': {'merged': 0.4361, 'expected': 0.4658, 'effect': -2.97, 'outcome': 'INTERFERENCE'},
        'tl-PH': {'merged': 0.5921, 'expected': 0.5272, 'effect': +6.49, 'outcome': 'SYNERGY'},
        'id-ID': {'merged': 0.7091, 'expected': 0.7386, 'effect': -2.95, 'outcome': 'INTERFERENCE'},
    }

    print("\n" + "=" * 100)
    print("VALIDATION ON KNOWN RESULTS (9 targets)")
    print("=" * 100)

    # Compare all predictor versions
    versions = ['synergy_score_v1', 'synergy_score_v2', 'synergy_score_v3', 'synergy_score_v4']

    for version in versions:
        print(f"\n--- {version} ---")

        # Sort by this version's score
        df_sorted = df.sort_values(version, ascending=False).reset_index(drop=True)

        correct = 0
        results = []

        for locale, info in known_results.items():
            row = df_sorted[df_sorted['target'] == locale].iloc[0]
            rank = df_sorted[df_sorted['target'] == locale].index[0] + 1
            score = row[version]

            # Predict: top half = synergy, bottom half = interference
            predicted = 'SYNERGY' if rank <= 25 else 'INTERFERENCE'
            actual = info['outcome']
            match = predicted == actual
            if match:
                correct += 1

            results.append({
                'locale': locale,
                'rank': rank,
                'score': score,
                'predicted': predicted,
                'actual': actual,
                'effect': info['effect'],
                'match': '✓' if match else '✗',
                'source_quality': row['source_quality'],
                'regional_coherence': row['regional_coherence'],
            })

        # Print results table
        print(f"\n{'Locale':<8} {'Rank':>4} {'Score':>8} {'Pred':<12} {'Actual':<12} {'Effect':>8} {'Match':>5} {'Quality':>8} {'Coherence':>10}")
        print("-" * 95)
        for r in sorted(results, key=lambda x: x['rank']):
            print(f"{r['locale']:<8} {r['rank']:>4} {r['score']:>8.4f} {r['predicted']:<12} {r['actual']:<12} {r['effect']:>+7.2f}% {r['match']:>5} {r['source_quality']:>8.2f} {r['regional_coherence']:>10.2f}")

        print(f"\nAccuracy: {correct}/9 ({correct/9*100:.1f}%)")

    return known_results


def main():
    print("=" * 100)
    print("MERGING EFFECT PREDICTOR V2 - Enhanced with Source Quality & Regional Coherence")
    print("=" * 100)

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

    # Analyze on known results
    known_results = analyze_known_results(df)

    # Rank by best predictor (V3)
    df = df.sort_values('synergy_score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("TOP 15 TARGETS (Highest Predicted Synergy - V3)")
    print("=" * 100)

    display_cols = ['target', 'synergy_score', 'diversity_score', 'source_quality',
                    'regional_coherence', 'dominant_region', 'source_acc_std']
    print(df.head(15)[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("BOTTOM 15 TARGETS (Lowest Predicted Synergy - Likely Interference)")
    print("=" * 100)

    print(df.tail(15)[display_cols].to_string(index=False))

    # Feature importance analysis
    print("\n" + "=" * 100)
    print("FEATURE ANALYSIS")
    print("=" * 100)

    # Correlation with actual Merging Effect (for known targets)
    known_df = df[df['target'].isin(known_results.keys())].copy()
    known_df['actual_effect'] = known_df['target'].map(lambda x: known_results[x]['effect'])

    print("\nCorrelation with Actual Merging Effect (9 known targets):")
    feature_cols = ['diversity_score', 'source_quality', 'regional_coherence',
                    'source_acc_std', 'zs_self_ratio', 'regional_alignment']
    for col in feature_cols:
        corr = known_df['actual_effect'].corr(known_df[col])
        print(f"  {col:25s}: {corr:+.4f}")

    # Identify targets for expanded validation
    print("\n" + "=" * 100)
    print("RECOMMENDED TARGETS FOR EXPANDED VALIDATION")
    print("=" * 100)

    # Exclude already-tested targets
    tested = set(known_results.keys())
    untested = df[~df['target'].isin(tested)]

    # Select middle-ranked targets (ranks 10-40) for better calibration
    mid_ranked = untested[(untested.index >= 10) & (untested.index <= 40)]

    print("\nMiddle-ranked untested targets (for calibration):")
    print(mid_ranked[['target', 'synergy_score', 'diversity_score', 'source_quality',
                      'regional_coherence']].head(15).to_string(index=False))

    # Export
    output_path = PROJECT_ROOT / "analysis" / "merging_effect_predictions_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return df


if __name__ == "__main__":
    main()
