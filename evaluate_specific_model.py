#!/usr/bin/env python3
"""
Script to evaluate a specific hyperparameter setting from the MASSIVE model repository.
"""

import argparse
import torch
import json
import os
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_results_folder(base_model: str = None, model_dir: str = None, locale: str = None, prefix: str = None):
    """Create results folder with structured naming."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create structured folder name
    if prefix:
        folder_name = f"{prefix}_{locale}"
    elif base_model and model_dir:
        # Extract base model name from full path
        base_name = base_model.split('/')[-1] if '/' in base_model else base_model
        folder_name = f"{base_name}_{model_dir}_{locale}"
    elif base_model:
        base_name = base_model.split('/')[-1] if '/' in base_model else base_model
        folder_name = f"{base_name}_{locale}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"eval_{timestamp}"
    
    eval_folder = os.path.join(results_dir, folder_name)
    os.makedirs(eval_folder, exist_ok=True)
    
    return eval_folder

def save_evaluation_results(results: dict, eval_folder: str):
    """Save evaluation results to JSON file."""
    results_path = os.path.join(eval_folder, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Results saved to: {results_path}")

def map_locale_to_massive_format(locale: str):
    """Map language code to MASSIVE locale format using model_mapping.csv."""
    try:
        model_mapping_path = os.path.join(os.path.dirname(__file__), "model_mapping.csv")
        if os.path.exists(model_mapping_path):
            model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)
            
            # If locale is already in MASSIVE format, return as-is
            if locale in model_mapping_df['locale'].values:
                return locale
            
            # If locale is a language code, map it to MASSIVE format
            if locale in model_mapping_df.index:
                massive_locale = model_mapping_df.loc[locale, 'locale']
                logger.info(f"Mapped locale '{locale}' to '{massive_locale}'")
                return massive_locale
                
        # If no mapping found, return original locale
        logger.warning(f"No mapping found for locale '{locale}', using as-is")
        return locale
    except Exception as e:
        logger.warning(f"Could not load locale mapping: {e}")
        return locale

def evaluate_specific_model(model_name: str, subfolder: str | None = None, locale: str = "cy-GB", eval_folder: str = None):
    """Evaluate a specific model from the hyperparameter tuning results."""
    
    try:
        # Map locale to MASSIVE format if needed
        massive_locale = map_locale_to_massive_format(locale)
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Testing on locale: {locale} (mapped to: {massive_locale})")
        
        # Load tokenizer
        if subfolder:
            tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        try:
            # Try loading with device_map first (preferred)
            if subfolder:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    subfolder=subfolder,
                    device_map=device if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else None
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    device_map=device if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else None
                )
        except Exception as e:
            logger.warning(f"Device mapping failed ({e}), falling back to CPU loading then moving to {device}")
            # Fallback: load on CPU first, then move to device
            if subfolder:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    subfolder=subfolder,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True
                )
            # Then move to target device
            model = model.to(device)

        logger.info("✓ Model loaded successfully")
        logger.info(f"✓ Model moved to {device}")
        
        # Load test data
        dataset = load_dataset("AmazonScience/massive", massive_locale, split="test", trust_remote_code=True)
        logger.info(f"✓ Loaded {len(dataset)} test examples")
        
        # Get intent labels from model config
        if hasattr(model.config, 'id2label'):
            intent_labels = list(model.config.id2label.values())
            logger.info(f"✓ Model has {len(intent_labels)} intent classes")
            logger.info(f"✓ Sample labels: {intent_labels[:5]}")
        else:
            logger.warning("Model config doesn't have id2label mapping")
            # Load the dataset to get the actual intent labels
            dataset = load_dataset("qanastek/MASSIVE", massive_locale, split="train[:100]", trust_remote_code=True)
            intents = dataset['intent']
            intent_labels = sorted(list(set(intents)))
            logger.info(f"✓ Found {len(intent_labels)} unique intents from dataset")
            logger.info(f"✓ Sample intents: {intent_labels[:5]}")
            
            # Create mapping if model uses numeric labels
            if hasattr(model.config, 'label2id'):
                logger.info(f"✓ Model config has label2id: {model.config.label2id}")
            else:
                logger.info("✓ Creating default numeric mapping")
        
        # Evaluate on a few examples
        logger.info("Testing on first 5 examples:")
        for i in range(min(5, len(dataset))):
            text = dataset[i]['utt']
            true_intent = dataset[i]['intent']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                predicted_class = predictions[0].item()
                
            # Get predicted label - handle LABEL_X format
            if hasattr(model.config, 'id2label') and predicted_class in model.config.id2label:
                predicted_label = model.config.id2label[predicted_class]
                # Convert LABEL_X format to just X
                if predicted_label.startswith('LABEL_'):
                    predicted_label = predicted_label.replace('LABEL_', '')
            elif intent_labels and predicted_class < len(intent_labels):
                predicted_label = intent_labels[predicted_class]
            else:
                predicted_label = str(predicted_class)  # Use as-is if no mapping
                
            # Debug info
            logger.info(f"Debug: predicted_class={predicted_class}, predicted_label='{predicted_label}', true_intent='{true_intent}'")
            
            print(f"  Example {i+1}:")
            print(f"    Text: {text}")
            print(f"    True: {true_intent}")
            print(f"    Predicted: {predicted_label}")
            print(f"    Correct: {true_intent == predicted_label}")
            print()
        
        # Calculate accuracy on full test set
        logger.info("Evaluating on full test set...")
        correct = 0
        total = 0
        
        batch_size = 32
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i:i+batch_size]['utt']
            batch_intents = dataset[i:i+batch_size]['intent']
            
            # Tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                
            # Compare with true labels
            for j, (pred, true_intent) in enumerate(zip(predictions, batch_intents)):
                predicted_class = pred.item()
                
                # Use same logic as above for getting predicted label
                if hasattr(model.config, 'id2label') and predicted_class in model.config.id2label:
                    predicted_label = model.config.id2label[predicted_class]
                    # Convert LABEL_X format to just X
                    if predicted_label.startswith('LABEL_'):
                        predicted_label = predicted_label.replace('LABEL_', '')
                elif intent_labels and predicted_class < len(intent_labels):
                    predicted_label = intent_labels[predicted_class]
                else:
                    predicted_label = str(predicted_class)  # Use as-is if no mapping
                
                # Convert true_intent to string for comparison if needed
                true_label = str(true_intent)
                
                if predicted_label == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"✓ Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Prepare comprehensive results
        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'subfolder': subfolder,
                'locale': locale,
                'massive_locale': massive_locale,
                'dataset': 'MASSIVE',
                'split': 'test'
            },
            'model_info': {
                'num_labels': model.num_labels,
                'model_type': model.config.model_type,
                'architecture': model.config.architectures[0] if model.config.architectures else None,
                'id2label': dict(model.config.id2label) if hasattr(model.config, 'id2label') else None,
                'label2id': dict(model.config.label2id) if hasattr(model.config, 'label2id') else None
            },
            'dataset_info': {
                'total_examples': total,
                'unique_intents': len(intent_labels) if intent_labels else None,
                'intent_labels': intent_labels if intent_labels else None
            },
            'performance': {
                'accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': total,
                'error_rate': (total - correct) / total if total > 0 else 0
            },
            'examples': []
        }
        
        # Add example predictions
        logger.info("Collecting example predictions...")
        for i in range(min(10, len(dataset))):  # Save first 10 examples
            text = dataset[i]['utt']
            true_intent = dataset[i]['intent']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                predicted_class = predictions[0].item()
            
            # Get predicted label - handle LABEL_X format
            if hasattr(model.config, 'id2label') and predicted_class in model.config.id2label:
                predicted_label = model.config.id2label[predicted_class]
                # Convert LABEL_X format to just X
                if predicted_label.startswith('LABEL_'):
                    predicted_label = predicted_label.replace('LABEL_', '')
            elif intent_labels and predicted_class < len(intent_labels):
                predicted_label = intent_labels[predicted_class]
            else:
                predicted_label = str(predicted_class)  # Use as-is if no mapping
            
            results['examples'].append({
                'example_id': i,
                'text': text,
                'true_intent': true_intent,
                'predicted_label': predicted_label,
                'predicted_class': predicted_class,
                'correct': str(predicted_label) == str(true_intent)
            })
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def list_available_models(base_model: str):
    """List available models in the repository."""
    from huggingface_hub import list_repo_files
    
    files = list_repo_files(base_model)
    model_dirs = set()
    
    for file in files:
        if '/' in file:
            parts = file.split('/')
            if len(parts) > 1 and parts[1] == 'config.json':
                model_dirs.add(parts[0])
    
    print(f"Available models in {base_model}:")
    for model_dir in sorted(model_dirs):
        print(f"  {model_dir}")
    
    return sorted(model_dirs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate specific MASSIVE model")
    parser.add_argument("--base-model", type=str, default="haryoaw/xlm-roberta-base_massive_k_cy-GB", 
                       help="Base model repository")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Specific model directory (e.g., alpha_0.5_cy-GB_epoch-9)")
    parser.add_argument("--locale", type=str, default="cy-GB", help="Locale to evaluate")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Custom results directory (default: creates structured folder)")
    parser.add_argument("--prefix", type=str, default=None,
                       help="Prefix for results folder (e.g., 'baseline', 'similarity', 'average')")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models(args.base_model)
    else:
        # Create results folder
        if args.results_dir:
            eval_folder = args.results_dir
            if not os.path.exists(eval_folder):
                os.makedirs(eval_folder)
        else:
            eval_folder = create_results_folder(args.base_model, args.model_dir, args.locale, args.prefix)
        
        logger.info(f"Results will be saved to: {eval_folder}")
        
        results = evaluate_specific_model(args.base_model, args.model_dir, args.locale, eval_folder)
        
        # If base model fails and no model_dir was specified, list available models
        if results is None and args.model_dir is None:
            logger.info("Base model failed. Available models:")
            available_models = list_available_models(args.base_model)
            if available_models:
                logger.info(f"Try using one of the model directories with --model-dir")
                logger.info(f"Example: python evaluate_specific_model.py --model-dir {available_models[-1]}")
        elif results:
            # Save results
            save_evaluation_results(results, eval_folder)
            
            print(f"\n🎉 Evaluation Results:")
            print(f"  Model: {results['evaluation_info']['model_name']}")
            print(f"  Subfolder: {results['evaluation_info']['subfolder']}")
            print(f"  Locale: {results['evaluation_info']['locale']}")
            print(f"  Dataset: {results['evaluation_info']['dataset']} ({results['evaluation_info']['split']} split)")
            print(f"  Accuracy: {results['performance']['accuracy']:.4f}")
            print(f"  Correct: {results['performance']['correct_predictions']}/{results['performance']['total_predictions']}")
            print(f"  Error Rate: {results['performance']['error_rate']:.4f}")
            print(f"  Results saved to: {eval_folder}/results.json")