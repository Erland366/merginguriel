#!/usr/bin/env python3
"""
Fine-tune XLM-RoBERTa on MASSIVE dataset for joint intent classification and slot filling.
Uses auto batch size finder and mixed precision training.
"""

import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    logging
)
from sklearn.metrics import classification_report, f1_score
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
import gc
import warnings
warnings.filterwarnings('ignore')

# Set logging
logging.set_verbosity_error()

def prepare_dataset(tokenizer, dataset, intent_label2id, slot_label2id):
    """Prepare dataset for training"""
    def tokenize_and_align_labels(examples):
        # Tokenize text
        tokenized_inputs = tokenizer(
            examples['tokens'], 
            truncation=True, 
            is_split_into_words=True,
            padding=False
        )
        
        labels = []
        for i, labels_in in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_in[word_idx])
                else:
                    # Same word as previous token, assign -100 or B-I transition
                    if labels_in[word_idx] % 2 == 1:  # B- tag
                        label_ids.append(labels_in[word_idx] + 1)  # Convert to I- tag
                    else:
                        label_ids.append(labels_in[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        tokenized_inputs["intent_labels"] = [intent_label2id[intent] for intent in examples['intent']]
        return tokenized_inputs
    
    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names
    )

def compute_metrics(p, intent_id2label, slot_id2label):
    """Compute metrics for both tasks"""
    predictions, labels = p
    
    # Intent classification metrics
    intent_preds = np.argmax(predictions[0], axis=1)
    intent_labels = labels[0]
    
    # Remove padding (-100) from slot predictions
    slot_preds = np.argmax(predictions[1], axis=2)
    slot_labels = labels[1]
    
    true_slot_preds = [
        [slot_id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(slot_preds, slot_labels)
    ]
    true_slot_labels = [
        [slot_id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(slot_preds, slot_labels)
    ]
    
    # Calculate metrics
    intent_f1 = f1_score(intent_labels, intent_preds, average='weighted')
    slot_f1 = seq_f1_score(true_slot_labels, true_slot_preds)
    
    return {
        'intent_f1': intent_f1,
        'slot_f1': slot_f1,
        'combined_f1': (intent_f1 + slot_f1) / 2
    }

class JointTrainer(Trainer):
    """Custom trainer for joint intent classification and slot filling"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        intent_labels = inputs.pop("intent_labels")
        slot_labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get losses
        intent_loss = outputs.intent_loss
        slot_loss = outputs.slot_loss
        
        # Combined loss with weight (can be tuned)
        total_loss = 0.3 * intent_loss + 0.7 * slot_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

class JointXLMRModel(torch.nn.Module):
    """Joint model for intent classification and slot filling"""
    
    def __init__(self, model_name, num_intents, num_slots):
        super().__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_slots
        )
        
        # Intent classification head
        self.intent_classifier = torch.nn.Linear(
            self.base_model.config.hidden_size, 
            num_intents
        )
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, **kwargs):
        # Get token embeddings
        outputs = self.base_model.roberta(**kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Slot filling logits
        slot_logits = self.base_model.classifier(sequence_output)
        
        # Intent classification (use CLS token)
        cls_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_output)
        
        loss = None
        if 'labels' in kwargs:
            slot_labels = kwargs['labels']
            intent_labels = kwargs['intent_labels']
            
            # Calculate losses
            loss_fct = torch.nn.CrossEntropyLoss()
            slot_loss = loss_fct(
                slot_logits.view(-1, slot_logits.shape[-1]),
                slot_labels.view(-1)
            )
            intent_loss = loss_fct(
                intent_logits.view(-1, intent_logits.shape[-1]),
                intent_labels.view(-1)
            )
            
            loss = 0.3 * intent_loss + 0.7 * slot_loss
        
        return {
            'loss': loss,
            'slot_logits': slot_logits,
            'intent_logits': intent_logits,
            'intent_loss': intent_loss if loss is not None else None,
            'slot_loss': slot_loss if loss is not None else None
        }

def find_optimal_batch_size(model, tokenizer, dataset, device):
    """Find optimal batch size using memory optimization"""
    from transformers import Trainer, TrainingArguments
    
    batch_sizes = [2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        try:
            # Create temporary training args
            training_args = TrainingArguments(
                output_dir='./temp',
                per_device_train_batch_size=batch_size,
                fp16=True,
                report_to='none',
                disable_tqdm=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset.select(range(10)),  # Test on small subset
                tokenizer=tokenizer
            )
            
            # Try to train
            trainer.train()
            
            # Clean up
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            return batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"Batch size {batch_size} too large, trying smaller...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    return 1  # Fallback

def main():
    parser = argparse.ArgumentParser(description='Fine-tune XLM-R on MASSIVE dataset')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--dataset_lang', type=str, default='en-US')
    parser.add_argument('--output_dir', type=str, default='./xlmr-massive')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--load_best_model_at_end', action='store_true')
    parser.add_argument('--metric_for_best_model', type=str, default='combined_f1')
    parser.add_argument('--greater_is_better', action='store_true')
    parser.add_argument('--push_to_hub', action='store_true')
    parser.add_argument('--hub_model_id', type=str, default=None)
    parser.add_argument('--find_batch_size', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading MASSIVE dataset...")
    dataset = load_dataset("qanastek/MASSIVE", args.dataset_lang, trust_remote_code=True)
    
    # Get label mappings
    intent_labels = dataset['train'].features['intent'].names
    slot_labels = dataset['train'].features['ner_tags'].feature.names
    
    intent_label2id = {label: i for i, label in enumerate(intent_labels)}
    intent_id2label = {i: label for i, label in enumerate(intent_labels)}
    slot_label2id = {label: i for i, label in enumerate(slot_labels)}
    slot_id2label = {i: label for i, label in enumerate(slot_labels)}
    
    print(f"Intent classes: {len(intent_labels)}")
    print(f"Slot classes: {len(slot_labels)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Prepare datasets
    print("Preparing datasets...")
    tokenized_datasets = {}
    for split in dataset.keys():
        tokenized_datasets[split] = prepare_dataset(
            tokenizer, 
            dataset[split], 
            intent_label2id, 
            slot_label2id
        )
    
    # Create model
    model = JointXLMRModel(
        args.model_name,
        num_intents=len(intent_labels),
        num_slots=len(slot_labels)
    )
    
    # Find optimal batch size if requested
    if args.find_batch_size:
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(
            model, tokenizer, tokenized_datasets['train'], 'cuda'
        )
        args.batch_size = optimal_batch_size
        print(f"Optimal batch size: {args.batch_size}")
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Custom data collator for joint training
    def joint_data_collator(features):
        batch = data_collator(features)
        intent_labels = [f['intent_labels'] for f in features]
        batch['intent_labels'] = torch.tensor(intent_labels, dtype=torch.long)
        return batch
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        fp16=True,  # Mixed precision
        gradient_accumulation_steps=1,
        logging_steps=100,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy='every_save',
        report_to='tensorboard'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=joint_data_collator,
        compute_metrics=lambda p: compute_metrics(p, intent_id2label, slot_id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating on test set...")
    eval_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test results: {eval_results}")
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    
    # Save label mappings
    import json
    label_mappings = {
        'intent_label2id': intent_label2id,
        'intent_id2label': intent_id2label,
        'slot_label2id': slot_label2id,
        'slot_id2label': slot_id2label
    }
    with open(os.path.join(args.output_dir, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    # Push to hub if requested
    if args.push_to_hub:
        print("Pushing model to hub...")
        trainer.push_to_hub()
    
    print("Training complete!")

if __name__ == '__main__':
    main()