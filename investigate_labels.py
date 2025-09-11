#!/usr/bin/env python3
"""
Script to investigate label mapping issues and create the correct mapping.
"""

import json
from datasets import load_dataset
from transformers import AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate_label_mapping(model_name: str, subfolder: str, locale: str = "cy-GB"):
    """Investigate the label mapping between model and dataset."""
    
    print("="*60)
    print("LABEL MAPPING INVESTIGATION")
    print("="*60)
    
    # Load model config
    if subfolder:
        config = AutoConfig.from_pretrained(model_name, subfolder=subfolder)
    else:
        config = AutoConfig.from_pretrained(model_name)
    
    print(f"\n1. MODEL CONFIG LABELS:")
    print(f"   Model type: {config.model_type}")
    print(f"   Number of labels: {config.num_labels}")
    
    if hasattr(config, 'id2label'):
        print(f"   Model id2label mapping:")
        for i, label in config.id2label.items():
            print(f"     {i} -> '{label}'")
    else:
        print("   ❌ No id2label mapping found in model config")
    
    if hasattr(config, 'label2id'):
        print(f"   Model label2id mapping:")
        for label, i in config.label2id.items():
            print(f"     '{label}' -> {i}")
    else:
        print("   ❌ No label2id mapping found in model config")
    
    # Load dataset to get actual intents
    print(f"\n2. DATASET LABELS:")
    dataset = load_dataset("qanastek/MASSIVE", locale, split="train", trust_remote_code=True)
    
    intents = dataset['intent']
    unique_intents = sorted(list(set(intents)))
    
    print(f"   Total unique intents: {len(unique_intents)}")
    print(f"   Intent range: {min(unique_intents)} to {max(unique_intents)}")
    print(f"   All intents: {unique_intents}")
    
    # Check if intents are continuous
    expected_intents = list(range(min(unique_intents), max(unique_intents) + 1))
    if unique_intents == expected_intents:
        print(f"   ✅ Intents are continuous: {expected_intents}")
    else:
        print(f"   ❌ Intents are NOT continuous")
        print(f"      Expected: {expected_intents}")
        print(f"      Actual:   {unique_intents}")
        missing = set(expected_intents) - set(unique_intents)
        if missing:
            print(f"      Missing:   {sorted(missing)}")
    
    # Analyze the mismatch
    print(f"\n3. MISMATCH ANALYSIS:")
    
    if hasattr(config, 'id2label'):
        model_labels = set(config.id2label.values())
        dataset_labels = set(str(i) for i in unique_intents)
        
        print(f"   Model labels: {sorted(model_labels)}")
        print(f"   Dataset labels: {sorted(dataset_labels)}")
        
        overlap = model_labels & dataset_labels
        print(f"   Overlapping labels: {sorted(overlap)}")
        
        model_only = model_labels - dataset_labels
        dataset_only = dataset_labels - model_labels
        
        if model_only:
            print(f"   Labels only in model: {sorted(model_only)}")
        if dataset_only:
            print(f"   Labels only in dataset: {sorted(dataset_only)}")
    
    # Try to create a mapping
    print(f"\n4. SUGGESTED FIXES:")
    
    if hasattr(config, 'id2label'):
        # Check if model labels are numeric strings
        all_numeric = all(label.isdigit() for label in config.id2label.values())
        if all_numeric:
            print(f"   ✅ All model labels are numeric strings")
            print(f"   💡 Try converting predictions to int before comparison")
        else:
            print(f"   ❌ Model labels are not all numeric")
            print(f"   💡 Need to create a custom mapping")
    
    # Check if we need to remap
    if hasattr(config, 'id2label') and len(config.id2label) == len(unique_intents):
        print(f"   ✅ Label counts match ({len(config.id2label)} == {len(unique_intents)})")
        print(f"   💡 Model might be trained with 0-indexed labels while dataset uses different indexing")
    
    return {
        'model_labels': dict(config.id2label) if hasattr(config, 'id2label') else None,
        'dataset_labels': unique_intents,
        'model_num_labels': config.num_labels,
        'dataset_num_labels': len(unique_intents)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Investigate label mapping issues")
    parser.add_argument("--model", type=str, default="haryoaw/xlm-roberta-base_massive_k_cy-GB", 
                       help="Base model repository")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Specific model directory (e.g., alpha_0.5_cy-GB_epoch-9)")
    parser.add_argument("--locale", type=str, default="cy-GB", help="Locale to evaluate")
    
    args = parser.parse_args()
    
    investigation = investigate_label_mapping(args.model, args.model_dir, args.locale)
    
    # Save investigation results
    with open("label_investigation.json", "w") as f:
        json.dump(investigation, f, indent=2)
    
    print(f"\nInvestigation results saved to: label_investigation.json")