#!/usr/bin/env python3
"""
Filter for xlm-roberta-base_massive_k_ models with locale endings.
"""

import pandas as pd
import re

def filter_k_models():
    """Filter for xlm-roberta-base_massive_k_ models."""
    
    # Read the CSV we just created
    df = pd.read_csv("haryoaw_massive_models.csv")
    
    # Filter for models matching the pattern xlm-roberta-base_massive_k_[locale]
    pattern = r'^haryoaw/xlm-roberta-base_massive_k_[a-z]{2}-[A-Z]{2}$'
    k_models = []
    
    for _, row in df.iterrows():
        model_name = row['model_name']
        if re.match(pattern, model_name):
            k_models.append({
                'model_name': model_name,
                'locale': row['locale'],
                'author': 'haryoaw',
                'base_model': 'xlm-roberta-base_massive_k'
            })
    
    # Sort by locale
    k_models_sorted = sorted(k_models, key=lambda x: x['locale'])
    
    # Save to CSV
    if k_models_sorted:
        result_df = pd.DataFrame(k_models_sorted)
        result_df.to_csv("haryoaw_k_models.csv", index=False)
        
        print(f"Found {len(k_models_sorted)} models matching pattern 'xlm-roberta-base_massive_k_[locale]':")
        print()
        for model in k_models_sorted:
            print(f"  {model['model_name']}")
        
        print(f"\nResults saved to: haryoaw_k_models.csv")
    else:
        print("No models found matching the pattern!")
    
    return k_models_sorted

if __name__ == "__main__":
    k_models = filter_k_models()