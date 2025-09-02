#!/usr/bin/env python3
"""
Script to analyze the MASSIVE dataset structure and understand the task type.
"""

from datasets import load_dataset
import numpy as np
from collections import Counter

# Load the dataset
dataset = load_dataset("qanastek/MASSIVE", "en-US", split="train", trust_remote_code=True)
print("Dataset features:", dataset.features)
print("Dataset size:", len(dataset))

# Check the first few examples
print("\nFirst example:")
print(dataset[0])

# Analyze intents
intents = dataset['intent']
intent_counts = Counter(intents)
print(f"\nNumber of unique intents: {len(intent_counts)}")
print("Top 10 intents:")
for intent, count in intent_counts.most_common(10):
    print(f"  {intent}: {count}")

# Analyze NER tags
ner_tags = dataset['ner_tags']
all_tags = [tag for example_tags in ner_tags for tag in example_tags]
tag_counts = Counter(all_tags)
print(f"\nNumber of unique NER tags: {len(tag_counts)}")
print("NER tags:", list(tag_counts.keys()))

# Check if there are slots/entities
print("\nChecking for slot information:")
if 'annot_utt' in dataset.features:
    print("Found 'annot_utt' - checking for slot annotations...")
    for i in range(3):
        print(f"Example {i}:")
        print(f"  Utterance: {dataset[i]['utt']}")
        print(f"  Annotated: {dataset[i]['annot_utt']}")

# Determine the task type
print("\n" + "="*50)
print("TASK ANALYSIS:")
print("="*50)
print("This appears to be an INTENT CLASSIFICATION + NER task:")
print(f"- Intent classification: {len(intent_counts)} classes")
print(f"- NER/Slot filling: {len(tag_counts)} entity types")
print("\nThis is a typical NLU (Natural Language Understanding) task.")
print("We should fine-tune the model for both intent classification and slot filling.")