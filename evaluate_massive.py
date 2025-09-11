#!/usr/bin/env python3
"""
Evaluation script for MASSIVE dataset finetuned models.
Supports intent classification and NER evaluation for multilingual NLU models.
"""

import argparse
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
import torch
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MASSIVEEvaluator:
    def __init__(self, model_name: str, locale: str = "en-US", device: str = "auto"):
        """
        Initialize the MASSIVE evaluator.
        
        Args:
            model_name: HuggingFace model name or path
            locale: Locale to evaluate (e.g., "en-US", "es-ES", "fr-FR")
            device: Device to use for evaluation
        """
        self.model_name = model_name
        self.locale = locale
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset
        logger.info(f"Loading MASSIVE dataset for locale: {locale}")
        self.dataset = load_dataset("qanastek/MASSIVE", locale, trust_remote_code=True)
        
        # Analyze dataset structure
        self._analyze_dataset()
        
        # Initialize models and tokenizers
        self.intent_tokenizer = None
        self.intent_model = None
        self.ner_tokenizer = None
        self.ner_model = None
        
        # Label mappings
        self.intent_labels = None
        self.ner_labels = None
        
    def _analyze_dataset(self):
        """Analyze the dataset structure and extract label mappings."""
        logger.info("Analyzing dataset structure...")
        
        # Get intent labels
        intents = self.dataset['train']['intent']
        self.intent_labels = sorted(list(set(intents)))
        self.intent2id = {label: i for i, label in enumerate(self.intent_labels)}
        self.id2intent = {i: label for label, i in self.intent2id.items()}
        
        # Get NER labels
        all_ner_tags = [tag for example in self.dataset['train']['ner_tags'] for tag in example]
        self.ner_labels = sorted(list(set(all_ner_tags)))
        self.ner2id = {label: i for i, label in enumerate(self.ner_labels)}
        self.id2ner = {i: label for label, i in self.ner2id.items()}
        
        logger.info(f"Found {len(self.intent_labels)} intent classes: {self.intent_labels[:5]}...")
        logger.info(f"Found {len(self.ner_labels)} NER labels: {self.ner_labels}")
        
    def load_intent_model(self, model_path: Optional[str] = None):
        """Load intent classification model."""
        model_path = model_path or self.model_name
        logger.info(f"Loading intent classification model from: {model_path}")
        
        self.intent_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=len(self.intent_labels),
            id2label=self.id2intent,
            label2id=self.intent2id
        )
        self.intent_model.to(self.device)
        
    def load_ner_model(self, model_path: Optional[str] = None):
        """Load NER model."""
        model_path = model_path or self.model_name
        logger.info(f"Loading NER model from: {model_path}")
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(self.ner_labels),
            id2label=self.id2ner,
            label2id=self.ner2id
        )
        self.ner_model.to(self.device)
        
    def prepare_intent_data(self, split: str = "test"):
        """Prepare data for intent classification."""
        dataset = self.dataset[split]
        
        texts = dataset['utt']
        labels = [self.intent2id[intent] for intent in dataset['intent']]
        
        # Tokenize
        encodings = self.intent_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }
        
    def prepare_ner_data(self, split: str = "test"):
        """Prepare data for NER."""
        dataset = self.dataset[split]
        
        texts = dataset['utt']
        tokens_list = dataset['tokens']
        ner_tags_list = dataset['ner_tags']
        
        # Tokenize and align labels
        tokenized_inputs = self.ner_tokenizer(
            texts,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=128,
            return_tensors="pt"
        )
        
        labels = []
        for i, tags in enumerate(ner_tags_list):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(self.ner2id[tags[word_idx]])
                else:
                    label_ids.append(-100)  # Sub-word tokens
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': torch.tensor(labels)
        }
        
    def evaluate_intent(self, split: str = "test") -> Dict:
        """Evaluate intent classification."""
        if self.intent_model is None:
            self.load_intent_model()
            
        logger.info(f"Evaluating intent classification on {split} split...")
        
        # Prepare data
        data = self.prepare_intent_data(split)
        dataset = self.dataset[split]
        
        # Evaluate
        self.intent_model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = 32
            for i in tqdm(range(0, len(data['input_ids']), batch_size)):
                batch_input_ids = data['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = data['attention_mask'][i:i+batch_size].to(self.device)
                
                outputs = self.intent_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        # Calculate metrics
        true_labels = data['labels'].numpy()
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        results = {
            'accuracy': acc,
            'f1_weighted': f1,
            'f1_per_class': dict(zip(self.intent_labels, f1_per_class)),
            'classification_report': classification_report(
                true_labels, predictions, target_names=self.intent_labels, zero_division=0
            )
        }
        
        logger.info(f"Intent Classification Results:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  F1 (weighted): {f1:.4f}")
        
        return results
        
    def evaluate_ner(self, split: str = "test") -> Dict:
        """Evaluate NER."""
        if self.ner_model is None:
            self.load_ner_model()
            
        logger.info(f"Evaluating NER on {split} split...")
        
        # Prepare data
        data = self.prepare_ner_data(split)
        
        # Evaluate
        self.ner_model.eval()
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            batch_size = 16
            for i in tqdm(range(0, len(data['input_ids']), batch_size)):
                batch_input_ids = data['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = data['attention_mask'][i:i+batch_size].to(self.device)
                batch_labels = data['labels'][i:i+batch_size]
                
                outputs = self.ner_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=2)
                
                # Convert to label sequences (ignoring padding and subwords)
                for j in range(len(batch_input_ids)):
                    pred_sequence = []
                    true_sequence = []
                    
                    for k in range(len(batch_input_ids[j])):
                        if batch_labels[j][k] != -100:  # Not padding/subword
                            pred_sequence.append(self.id2ner[preds[j][k].item()])
                            true_sequence.append(self.id2ner[batch_labels[j][k].item()])
                    
                    all_predictions.append(pred_sequence)
                    all_true_labels.append(true_sequence)
        
        # Calculate metrics (excluding 'O' tag for entity-specific metrics)
        all_true_flat = [tag for seq in all_true_labels for tag in seq]
        all_pred_flat = [tag for seq in all_predictions for tag in seq]
        
        # Overall accuracy
        overall_acc = accuracy_score(all_true_flat, all_pred_flat)
        
        # Entity-level metrics (excluding 'O')
        entity_true = [tag for tag in all_true_flat if tag != 'O']
        entity_pred = [tag for tag in all_pred_flat if tag != 'O']
        
        if entity_true:  # Only calculate if there are entities
            entity_f1 = f1_score(entity_true, entity_pred, average='weighted')
            entity_precision, entity_recall, entity_f1_per_class, _ = precision_recall_fscore_support(
                entity_true, entity_pred, average=None, zero_division=0
            )
        else:
            entity_f1 = 0.0
            entity_f1_per_class = []
            
        results = {
            'overall_accuracy': overall_acc,
            'entity_f1_weighted': entity_f1,
            'classification_report': classification_report(
                all_true_flat, all_pred_flat, target_names=self.ner_labels, zero_division=0
            )
        }
        
        if entity_f1_per_class:
            entity_labels = [label for label in self.ner_labels if label != 'O']
            results['entity_f1_per_class'] = dict(zip(entity_labels, entity_f1_per_class))
        
        logger.info(f"NER Results:")
        logger.info(f"  Overall Accuracy: {overall_acc:.4f}")
        logger.info(f"  Entity F1 (weighted): {entity_f1:.4f}")
        
        return results
        
    def evaluate_full(self, split: str = "test") -> Dict:
        """Evaluate both intent classification and NER."""
        results = {
            'model': self.model_name,
            'locale': self.locale,
            'split': split,
            'intent_classification': self.evaluate_intent(split),
            'ner': self.evaluate_ner(split)
        }
        
        return results
        
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MASSIVE dataset models")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--locale", type=str, default="en-US", help="Locale to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file path")
    parser.add_argument("--task", type=str, choices=["intent", "ner", "both"], default="both", help="Task to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MASSIVEEvaluator(args.model, args.locale, args.device)
    
    # Evaluate based on task
    if args.task == "intent":
        results = {
            'model': args.model,
            'locale': args.locale,
            'split': args.split,
            'intent_classification': evaluator.evaluate_intent(args.split)
        }
    elif args.task == "ner":
        results = {
            'model': args.model,
            'locale': args.locale,
            'split': args.split,
            'ner': evaluator.evaluate_ner(args.split)
        }
    else:  # both
        results = evaluator.evaluate_full(args.split)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Locale: {args.locale}")
    print(f"Split: {args.split}")
    
    if 'intent_classification' in results:
        ic_results = results['intent_classification']
        print(f"\nIntent Classification:")
        print(f"  Accuracy: {ic_results['accuracy']:.4f}")
        print(f"  F1 (weighted): {ic_results['f1_weighted']:.4f}")
    
    if 'ner' in results:
        ner_results = results['ner']
        print(f"\nNamed Entity Recognition:")
        print(f"  Overall Accuracy: {ner_results['overall_accuracy']:.4f}")
        print(f"  Entity F1 (weighted): {ner_results['entity_f1_weighted']:.4f}")

if __name__ == "__main__":
    main()