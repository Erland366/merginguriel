"""
Regional Coherence Analysis

Deep-dive into tl-PH phenomenon:
- Low diversity (0.058) but achieved +6.49% synergy
- All 5 sources are Southeast Asian (vi-VN, km-KH, jv-ID, th-TH, ms-MY)

Research Questions:
1. Why does regional coherence enable synergy despite low diversity?
2. What features do SE Asian languages share that combine well?
3. Can we identify other regional clusters that might synergize?
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

# Regional mapping
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


def analyze_regional_cluster(
    targets: List[str],
    nxn_matrix: pd.DataFrame,
    raw_similarity_matrix: pd.DataFrame
) -> Dict:
    """Analyze transfer patterns within a regional cluster."""

    region = LOCALE_REGIONS.get(targets[0], 'Unknown')

    # Within-region transfer matrix
    print(f"\n{'='*60}")
    print(f"WITHIN-REGION TRANSFER: {region}")
    print(f"{'='*60}")

    print("\nNxN matrix (source → target accuracy):")
    sub_matrix = nxn_matrix.loc[targets, targets]
    print(sub_matrix.round(3).to_string())

    # Average within-region transfer
    off_diag = []
    for src in targets:
        for tgt in targets:
            if src != tgt:
                off_diag.append(nxn_matrix.loc[src, tgt])

    avg_transfer = np.mean(off_diag)
    std_transfer = np.std(off_diag)

    print(f"\nWithin-region transfer stats:")
    print(f"  Average: {avg_transfer:.4f}")
    print(f"  Std dev: {std_transfer:.4f}")
    print(f"  Range:   [{min(off_diag):.4f}, {max(off_diag):.4f}]")

    # Within-region similarity
    print("\nWithin-region similarity (REAL):")
    sim_matrix = raw_similarity_matrix.loc[targets, targets]
    print(sim_matrix.round(3).to_string())

    sim_off_diag = []
    for i, src in enumerate(targets):
        for j, tgt in enumerate(targets):
            if i < j:
                sim_off_diag.append(raw_similarity_matrix.loc[src, tgt])

    avg_sim = np.mean(sim_off_diag)
    print(f"\nAverage pairwise similarity: {avg_sim:.4f}")

    return {
        'region': region,
        'targets': targets,
        'avg_transfer': avg_transfer,
        'std_transfer': std_transfer,
        'avg_similarity': avg_sim,
    }


def analyze_tl_ph_case(nxn_matrix: pd.DataFrame, raw_sim: pd.DataFrame):
    """Deep analysis of tl-PH synergy case."""

    print("\n" + "="*80)
    print("DEEP DIVE: tl-PH Regional Coherence Phenomenon")
    print("="*80)

    target = 'tl-PH'
    sources = get_pipeline_sources(target, num_languages=5)
    source_locales = [loc for loc, _ in sources]

    print(f"\nTarget: {target}")
    print(f"Region: {LOCALE_REGIONS.get(target)}")
    print(f"Sources: {source_locales}")
    print(f"Source regions: {[LOCALE_REGIONS.get(s) for s in source_locales]}")

    # Merging Effect calculation
    source_accs = [nxn_matrix.loc[src, target] for src in source_locales]
    expected = np.mean(source_accs)
    actual_merged = 0.5921  # From validation experiment

    print(f"\nSource accuracies on tl-PH:")
    for src, acc in zip(source_locales, source_accs):
        print(f"  {src}: {acc:.4f}")
    print(f"  Average: {expected:.4f}")
    print(f"  Merged:  {actual_merged:.4f}")
    print(f"  Merging Effect: {(actual_merged - expected)*100:+.2f}%")

    # Cross-source transfer within SE Asian cluster
    print("\n" + "-"*60)
    print("Cross-source transfer (how well do sources transfer to each other):")
    print("-"*60)

    cross_transfer = []
    print(f"\n{'Source':<8} → {'Other sources avg':<20}")
    for src in source_locales:
        others = [s for s in source_locales if s != src]
        accs_on_others = [nxn_matrix.loc[src, other] for other in others]
        avg = np.mean(accs_on_others)
        cross_transfer.append(avg)
        print(f"{src:<8} → {avg:.4f}")

    print(f"\nAverage cross-source transfer: {np.mean(cross_transfer):.4f}")
    print(f"Std dev: {np.std(cross_transfer):.4f}")

    # Source pairwise similarity
    print("\n" + "-"*60)
    print("Source pairwise similarity (REAL):")
    print("-"*60)

    pairwise_sims = []
    print(f"\n{'Pair':<15} {'Similarity':>10}")
    for i, src1 in enumerate(source_locales):
        for j, src2 in enumerate(source_locales):
            if i < j:
                sim = raw_sim.loc[src1, src2]
                pairwise_sims.append(sim)
                print(f"{src1}-{src2:<7} {sim:.4f}")

    diversity = 1 - np.mean(pairwise_sims)
    print(f"\nAverage pairwise similarity: {np.mean(pairwise_sims):.4f}")
    print(f"Diversity score: {diversity:.4f}")

    return {
        'target': target,
        'sources': source_locales,
        'source_accs': source_accs,
        'expected': expected,
        'actual': actual_merged,
        'merging_effect': actual_merged - expected,
        'diversity': diversity,
        'cross_transfer': np.mean(cross_transfer),
    }


def compare_coherent_vs_diverse(nxn_matrix: pd.DataFrame, raw_sim: pd.DataFrame):
    """Compare tl-PH (coherent) vs az-AZ (diverse)."""

    print("\n" + "="*80)
    print("COMPARISON: Coherent (tl-PH) vs Diverse (az-AZ)")
    print("="*80)

    cases = {
        'tl-PH': {'actual': 0.5921, 'effect': +6.49},
        'az-AZ': {'actual': 0.6627, 'effect': +2.57},
    }

    for target, info in cases.items():
        sources = get_pipeline_sources(target, num_languages=5)
        source_locales = [loc for loc, _ in sources]
        regions = [LOCALE_REGIONS.get(s, 'Unknown') for s in source_locales]
        region_counts = Counter(regions)

        source_accs = [nxn_matrix.loc[src, target] for src in source_locales]

        # Pairwise similarity
        pairwise_sims = []
        for i, src1 in enumerate(source_locales):
            for j, src2 in enumerate(source_locales):
                if i < j:
                    pairwise_sims.append(raw_sim.loc[src1, src2])

        print(f"\n--- {target} ---")
        print(f"Sources: {source_locales}")
        print(f"Regions: {regions}")
        print(f"Region counts: {dict(region_counts)}")
        print(f"Coherence: {max(region_counts.values())/5:.2f}")
        print(f"Source acc mean: {np.mean(source_accs):.4f}")
        print(f"Source acc std: {np.std(source_accs):.4f}")
        print(f"Pairwise sim mean: {np.mean(pairwise_sims):.4f}")
        print(f"Diversity: {1 - np.mean(pairwise_sims):.4f}")
        print(f"Merging Effect: {info['effect']:+.2f}%")

    # Key insight
    print("\n" + "-"*60)
    print("KEY INSIGHT")
    print("-"*60)
    print("""
tl-PH achieves HIGHER synergy (+6.49%) than az-AZ (+2.57%) despite:
- LOWER diversity (0.058 vs 0.172)
- LOWER source accuracy

Why? All 5 sources are from the same linguistic/geographic region (SE Asian).
They share common features that COMPLEMENT rather than CONFLICT during merging.

Hypothesis: Regional coherence enables synergy when sources share:
1. Similar grammatical structures
2. Similar writing systems or script families
3. Geographic/cultural proximity
4. Shared loanwords and linguistic influences
""")


def identify_coherent_clusters(nxn_matrix: pd.DataFrame, raw_sim: pd.DataFrame):
    """Identify other potentially coherent regional clusters."""

    print("\n" + "="*80)
    print("IDENTIFYING HIGH-COHERENCE REGIONAL CLUSTERS")
    print("="*80)

    # Group locales by region
    regions = {}
    for locale, region in LOCALE_REGIONS.items():
        if locale in nxn_matrix.index:
            if region not in regions:
                regions[region] = []
            regions[region].append(locale)

    print("\nRegional clusters (3+ members):")
    for region, locales in sorted(regions.items(), key=lambda x: -len(x[1])):
        if len(locales) >= 3:
            # Compute within-region transfer
            off_diag = []
            for src in locales:
                for tgt in locales:
                    if src != tgt:
                        off_diag.append(nxn_matrix.loc[src, tgt])
            avg_transfer = np.mean(off_diag)

            # Compute within-region similarity
            pairwise_sims = []
            for i, src in enumerate(locales):
                for j, tgt in enumerate(locales):
                    if i < j:
                        pairwise_sims.append(raw_sim.loc[src, tgt])
            avg_sim = np.mean(pairwise_sims) if pairwise_sims else 0

            print(f"\n{region} ({len(locales)} members):")
            print(f"  Locales: {locales}")
            print(f"  Avg within-region transfer: {avg_transfer:.4f}")
            print(f"  Avg within-region similarity: {avg_sim:.4f}")
            print(f"  Diversity: {1 - avg_sim:.4f}")

            # Predict: high transfer + high similarity = likely synergy candidate
            if avg_transfer > 0.55 and avg_sim > 0.8:
                print(f"  *** HIGH SYNERGY POTENTIAL ***")


def main():
    print("="*80)
    print("REGIONAL COHERENCE ANALYSIS")
    print("="*80)

    # Load data
    nxn = load_nxn_matrix()
    raw_sim = load_similarity_matrix(str(SIMILARITY_MATRIX_PATH), verbose=False)

    # Deep dive into tl-PH
    tl_ph_result = analyze_tl_ph_case(nxn, raw_sim)

    # Compare coherent vs diverse
    compare_coherent_vs_diverse(nxn, raw_sim)

    # Analyze SE Asian cluster
    se_asian_locales = [loc for loc, reg in LOCALE_REGIONS.items()
                        if reg == 'SE_Asian' and loc in nxn.index]
    analyze_regional_cluster(se_asian_locales, nxn, raw_sim)

    # Identify other coherent clusters
    identify_coherent_clusters(nxn, raw_sim)

    return tl_ph_result


if __name__ == "__main__":
    main()
