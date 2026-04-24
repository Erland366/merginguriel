#!/usr/bin/env python3
"""
Verification script to ensure both ensemble inference and merging pipeline
correctly normalize weights to sum to 1.0.
"""

import os
import json
import sys

def check_ensemble_weights():
    """Check ensemble inference weights."""
    try:
        results_dir = "urie_ensemble_results"
        if not os.path.exists(results_dir):
            print("❌ No ensemble results found")
            return False

        # Find the latest results
        target_lang = "en-US"
        results_path = os.path.join(results_dir, f"urie_uriel_logits_{target_lang}", "results.json")

        if not os.path.exists(results_path):
            print(f"❌ No ensemble results found for {target_lang}")
            return False

        with open(results_path, 'r') as f:
            data = json.load(f)

        weights = data['metadata']['weights']
        weight_sum = sum(weights)

        print("🔍 Ensemble Inference Weights:")
        for name, weight in zip(data['metadata']['model_names'], weights):
            print(f"  {name}: {weight:.6f}")
        print(f"  Sum: {weight_sum:.6f}")

        if abs(weight_sum - 1.0) < 1e-6:
            print("✅ Ensemble weights correctly sum to 1.0")
            return True
        else:
            print(f"❌ Ensemble weights sum to {weight_sum:.6f}, expected 1.0")
            return False

    except Exception as e:
        print(f"❌ Error checking ensemble weights: {e}")
        return False

def check_merging_weights():
    """Check merging pipeline weights."""
    try:
        merge_details_path = "merged_models/similarity_merge_en-US/merge_details.txt"

        if not os.path.exists(merge_details_path):
            print("❌ No merging results found")
            return False

        with open(merge_details_path, 'r') as f:
            content = f.read()

        weights = []
        for line in content.split('\n'):
            if 'Weight:' in line and 'Total Weight:' not in line and '%' in line:
                # Extract weight value (e.g., "0.617225" from "   - Weight: 0.617225 (61.72% of total)")
                try:
                    # Extract the weight value
                    weight_str = line.split('Weight:')[1].strip().split(' ')[0]
                    weights.append(float(weight_str))
                except (ValueError, IndexError):
                    continue

        weight_sum = sum(weights)

        print("🔍 Merging Pipeline Weights:")
        for weight in weights:
            print(f"  {weight:.6f}")
        print(f"  Sum: {weight_sum:.6f}")

        if abs(weight_sum - 1.0) < 1e-6:
            print("✅ Merging weights correctly sum to 1.0")
            return True
        else:
            print(f"❌ Merging weights sum to {weight_sum:.6f}, expected 1.0")
            return False

    except Exception as e:
        print(f"❌ Error checking merging weights: {e}")
        return False

def main():
    print("=" * 60)
    print("WEIGHT NORMALIZATION VERIFICATION")
    print("=" * 60)

    ensemble_ok = check_ensemble_weights()
    print()
    merging_ok = check_merging_weights()

    print()
    print("=" * 60)
    if ensemble_ok and merging_ok:
        print("✅ ALL SYSTEMS: Weight normalization is working correctly")
        sys.exit(0)
    else:
        print("❌ ISSUES FOUND: Weight normalization needs attention")
        sys.exit(1)

if __name__ == "__main__":
    main()