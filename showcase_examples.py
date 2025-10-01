#!/usr/bin/env python3
"""
Quick showcase of interesting cross-lingual merging examples for presentation
"""

import pandas as pd
import matplotlib.pyplot as plt

def create_showcase():
    """Create a showcase of interesting examples"""

    # Load the summary data
    df = pd.read_csv('cross_lingual_summary.csv')

    # Select interesting examples
    showcase_languages = ['cy-GB', 'jv-ID', 'tl-PH', 'ar-SA', 'ja-JP', 'vi-VN']
    showcase_df = df[df['locale'].isin(showcase_languages)].copy()

    print("=== Cross-Lingual Merging Showcase ===\n")

    print("1. BEST PERFORMER: Welsh (cy-GB)")
    cy_gb = showcase_df[showcase_df['locale'] == 'cy-GB'].iloc[0]
    print(f"   - Zero-shot performance: {cy_gb['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {cy_gb['similarity_merge']:.4f}")
    print(f"   - Improvement: +{cy_gb['sim_improvement_vs_zero_shot']:.4f}")
    print(f"   - Sources: {cy_gb['num_similarity_sources']} languages")

    print("\n2. LOW-RESOURCE SUCCESS: Javanese (jv-ID)")
    jv_id = showcase_df[showcase_df['locale'] == 'jv-ID'].iloc[0]
    print(f"   - Zero-shot performance: {jv_id['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {jv_id['similarity_merge']:.4f}")
    print(f"   - Improvement: +{jv_id['sim_improvement_vs_zero_shot']:.4f}")
    print(f"   - Sources: {jv_id['num_similarity_sources']} languages")

    print("\n3. METHOD DIFFERENCE: Tagalog (tl-PH)")
    tl_ph = showcase_df[showcase_df['locale'] == 'tl-PH'].iloc[0]
    print(f"   - Zero-shot performance: {tl_ph['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {tl_ph['similarity_merge']:.4f} (-{abs(tl_ph['sim_improvement_vs_zero_shot']):.4f})")
    print(f"   - Average merge: {tl_ph['average_merge']:.4f} (+{tl_ph['avg_improvement_vs_zero_shot']:.4f})")
    print(f"   - Method difference: {abs(tl_ph['sim_improvement_vs_zero_shot'] - tl_ph['avg_improvement_vs_zero_shot']):.4f}")

    print("\n4. STRATEGY MATTERS: Arabic (ar-SA)")
    ar_sa = showcase_df[showcase_df['locale'] == 'ar-SA'].iloc[0]
    print(f"   - Zero-shot performance: {ar_sa['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {ar_sa['similarity_merge']:.4f} (-{abs(ar_sa['sim_improvement_vs_zero_shot']):.4f})")
    print(f"   - Average merge: {ar_sa['average_merge']:.4f} (+{ar_sa['avg_improvement_vs_zero_shot']:.4f})")
    print(f"   - Best method: Average merge")

    print("\n5. HIGH-RESOURCE CHALLENGE: Japanese (ja-JP)")
    ja_jp = showcase_df[showcase_df['locale'] == 'ja-JP'].iloc[0]
    print(f"   - Zero-shot performance: {ja_jp['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {ja_jp['similarity_merge']:.4f} (-{abs(ja_jp['sim_improvement_vs_zero_shot']):.4f})")
    print(f"   - Average merge: {ja_jp['average_merge']:.4f} (-{abs(ja_jp['avg_improvement_vs_zero_shot']):.4f})")
    print(f"   - Both methods perform worse than zero-shot")

    print("\n6. CONSISTENT UNDERPERFORMANCE: Vietnamese (vi-VN)")
    vi_vn = showcase_df[showcase_df['locale'] == 'vi-VN'].iloc[0]
    print(f"   - Zero-shot performance: {vi_vn['similarity_zero_shot_avg']:.4f}")
    print(f"   - Similarity merge: {vi_vn['similarity_merge']:.4f} (-{abs(vi_vn['sim_improvement_vs_zero_shot']):.4f})")
    print(f"   - Average merge: {vi_vn['average_merge']:.4f} (-{abs(vi_vn['avg_improvement_vs_zero_shot']):.4f})")
    print(f"   - Both methods identical and worse than zero-shot")

    # Create a simple visualization
    plt.figure(figsize=(12, 6))

    x = range(len(showcase_languages))
    width = 0.25

    similarities = showcase_df['similarity_merge'].values
    averages = showcase_df['average_merge'].values
    zero_shots = showcase_df['similarity_zero_shot_avg'].values

    plt.bar([i - width for i in x], zero_shots, width, label='Zero-shot', alpha=0.7, color='gray')
    plt.bar([i for i in x], similarities, width, label='Similarity Merge', alpha=0.8, color='blue')
    plt.bar([i + width for i in x], averages, width, label='Average Merge', alpha=0.8, color='red')

    plt.xlabel('Languages')
    plt.ylabel('Performance Score')
    plt.title('Cross-Lingual Merging Showcase Examples')
    plt.xticks(x, showcase_languages)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('showcase_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nShowcase visualization saved as 'showcase_examples.png'")

if __name__ == "__main__":
    create_showcase()