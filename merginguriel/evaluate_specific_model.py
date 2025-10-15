#!/usr/bin/env python3
"""
Script to evaluate a specific MASSIVE model (local folder or repo path).
Expects MASSIVE-style locale codes (e.g., 'fr-FR') and that the model config
includes id2label/label2id mappings.
"""

import argparse
import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_results_folder(base_model: str = None, locale: str = None, prefix: str = None):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    if prefix:
        folder_name = f"{prefix}_{locale}"
    elif base_model:
        base_name = base_model.split('/')[-1] if '/' in base_model else base_model
        folder_name = f"{base_name}_{locale}"
    else:
        folder_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_folder = os.path.join(results_dir, folder_name)
    os.makedirs(eval_folder, exist_ok=True)
    return eval_folder


def save_evaluation_results(results: dict, eval_folder: str):
    results_path = os.path.join(eval_folder, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Results saved to: {results_path}")


def evaluate_specific_model(model_name: str, locale: str = "cy-GB", eval_folder: str = None):
    try:
        massive_locale = locale
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Testing on locale: {locale}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer loaded successfully")

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map=("auto" if device == "cuda" else None),
                torch_dtype=(torch.bfloat16 if device == "cuda" else None),
            )
        except Exception as e:
            logger.warning(f"Auto device mapping failed ({e}); loading on CPU then moving to {device}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
            ).to(device)
        logger.info("✓ Model loaded")

        # Load test data
        dataset = load_dataset("AmazonScience/massive", massive_locale, split="test", trust_remote_code=True)
        logger.info(f"✓ Loaded {len(dataset)} test examples")

        # Require label mappings
        if not hasattr(model.config, 'id2label') or not model.config.id2label:
            raise ValueError("Model config missing id2label/label2id; please save label mappings during training.")
        intent_labels = list(model.config.id2label.values())
        logger.info(f"✓ Model has {len(intent_labels)} intent classes")

        # Determine label format for comparison
        sample_labels = list(model.config.id2label.values())[:5]
        uses_label_format = any(
            isinstance(label, str) and not label.startswith('LABEL_') and not label.isdigit()
            for label in sample_labels
        )

        if uses_label_format:
            # Model uses string labels (intent names), dataset uses integers
            # Create mapping from model label to integer
            label_to_int = {label: idx for idx, label in model.config.id2label.items()}
            logger.info("Model uses string labels, comparing predicted class directly with true intent")
        else:
            # Model uses LABEL_0, LABEL_1 format
            logger.info("Model uses LABEL_N format, using standard comparison")

        # Quick sample preview
        logger.info("Previewing first 5 examples...")
        for i in range(min(5, len(dataset))):
            text = dataset[i]['utt']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls = int(torch.argmax(outputs.logits, dim=1)[0].item())

            if uses_label_format:
                pred = model.config.id2label.get(cls, str(cls))
            else:
                pred = model.config.id2label.get(cls, str(cls))
                if isinstance(pred, str) and pred.startswith('LABEL_'):
                    pred = pred.replace('LABEL_', '')
            logger.info(f"  [{i}] → {pred}")

        # Full evaluation
        correct = 0
        total = 0
        batch_size = 32
        for i in range(0, len(dataset), batch_size):
            texts = dataset[i:i+batch_size]['utt']
            true_intents = dataset[i:i+batch_size]['intent']
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
            for pred_id, true_intent in zip(preds, true_intents):
                cls = int(pred_id.item())

                if uses_label_format:
                    # Model uses string labels, compare predicted class directly with true intent
                    if cls == true_intent:
                        correct += 1
                else:
                    # Model uses LABEL_N format
                    pred = model.config.id2label.get(cls, str(cls))
                    if isinstance(pred, str) and pred.startswith('LABEL_'):
                        pred = pred.replace('LABEL_', '')
                    if str(pred) == str(true_intent):
                        correct += 1
                total += 1

        accuracy = correct / total if total else 0.0
        logger.info(f"✓ Test Accuracy: {accuracy:.4f} ({correct}/{total})")

        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'locale': locale,
                'massive_locale': massive_locale,
                'dataset': 'AmazonScience/massive',
                'split': 'test',
            },
            'model_info': {
                'num_labels': model.num_labels,
                'model_type': model.config.model_type,
                'architecture': model.config.architectures[0] if model.config.architectures else None,
                'id2label': dict(model.config.id2label),
                'label2id': dict(model.config.label2id) if hasattr(model.config, 'label2id') else None,
            },
            'dataset_info': {
                'total_examples': total,
                'unique_intents': len(intent_labels),
                'intent_labels': intent_labels,
            },
            'performance': {
                'accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': total,
                'error_rate': (total - correct) / total if total else 0.0,
            },
            'examples': [],
        }

        # Add example predictions
        for i in range(min(10, len(dataset))):
            text = dataset[i]['utt']
            true_intent = dataset[i]['intent']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls = int(torch.argmax(outputs.logits, dim=1)[0].item())
            pred = model.config.id2label.get(cls, str(cls))
            if isinstance(pred, str) and pred.startswith('LABEL_'):
                pred = pred.replace('LABEL_', '')
            results['examples'].append({
                'example_id': i,
                'text': text,
                'true_intent': true_intent,
                'predicted_label': pred,
                'predicted_class': cls,
                'correct': str(pred) == str(true_intent),
            })

        return results
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MASSIVE model on a locale")
    parser.add_argument("--base-model", type=str, required=True, help="Local path or HF repo id")
    parser.add_argument("--locale", type=str, default="cy-GB", help="MASSIVE locale to evaluate")
    parser.add_argument("--results-dir", type=str, default=None, help="Custom results dir")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for results folder")
    args = parser.parse_args()

    eval_folder = args.results_dir or create_results_folder(args.base_model, args.locale, args.prefix)
    logger.info(f"Results will be saved to: {eval_folder}")
    results = evaluate_specific_model(args.base_model, args.locale, eval_folder)
    if results:
        save_evaluation_results(results, eval_folder)
        print("\n🎉 Evaluation complete.")
        print(f"  Accuracy: {results['performance']['accuracy']:.4f}")
        print(f"  Saved: {eval_folder}/results.json")
