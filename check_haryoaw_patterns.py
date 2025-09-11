#!/usr/bin/env python3
"""
Check haryoaw's actual model patterns.
"""

import requests
import re

def get_haryoaw_models():
    """Get all models from haryoaw and show patterns."""
    
    # Get models
    url = "https://huggingface.co/api/models?author=haryoaw&limit=500"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json()
        
        print(f"Retrieved {len(models)} models from haryoaw")
        
        # Show all unique model name patterns
        patterns = {}
        for model in models:
            model_id = model.get('id', '')
            # Extract base name without author prefix
            if model_id.startswith('haryoaw/'):
                base_name = model_id[8:]  # Remove 'haryoaw/'
                
                # Group by pattern
                if 'xlm-roberta-base_massive' in base_name:
                    pattern = 'xlm-roberta-base_massive*'
                    if pattern not in patterns:
                        patterns[pattern] = []
                    patterns[pattern].append(base_name)
        
        # Show patterns
        print("\nModel patterns found:")
        for pattern, model_list in patterns.items():
            print(f"\n{pattern} ({len(model_list)} models):")
            for model in sorted(model_list):
                print(f"  haryoaw/{model}")
        
        return patterns
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    patterns = get_haryoaw_models()