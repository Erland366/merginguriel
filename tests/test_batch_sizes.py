#!/usr/bin/env python
"""
Test script to demonstrate how to use different batch sizes with training_bert.py

This script shows examples of using different batch sizes for training and evaluation.
"""

import subprocess
import sys

def test_batch_size_commands():
    """Generate example commands for testing different batch sizes."""

    print("=== Batch Size Configuration Examples ===\n")

    # Default batch sizes
    print("1. Default batch sizes (HuggingFace defaults):")
    cmd1 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_default_bs",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--max_train_samples", "200",
        "--max_eval_samples", "100",
        # No batch size specified - will use HuggingFace defaults (usually 8)
    ]
    print(" ".join(cmd1))
    print("→ Will use default batch sizes (typically 8 per device)")
    print()

    # Small batch sizes
    print("2. Small batch sizes (good for memory-constrained environments):")
    cmd2 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_small_bs",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--per-device-train-batch-size", "4",
        "--per-device-eval-batch-size", "8",
        "--max_train_samples", "200",
        "--max_eval_samples", "100",
        "--learning-rate", "3e-5",
        "--early_stopping_patience", "2"
    ]
    print(" ".join(cmd2))
    print("→ Training batch size: 4, Evaluation batch size: 8")
    print()

    # Medium batch sizes
    print("3. Medium batch sizes (balanced performance):")
    cmd3 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_medium_bs",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--per-device-train-batch-size", "16",
        "--per-device-eval-batch-size", "32",
        "--max_train_samples", "300",
        "--max_eval_samples", "150",
        "--learning-rate", "5e-5",
        "--early_stopping_patience", "3"
    ]
    print(" ".join(cmd3))
    print("→ Training batch size: 16, Evaluation batch size: 32")
    print()

    # Large batch sizes
    print("4. Large batch sizes (for GPUs with lots of memory):")
    cmd4 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_large_bs",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--per-device-train-batch-size", "32",
        "--per-device-eval-batch-size", "64",
        "--max_train_samples", "400",
        "--max_eval_samples", "200",
        "--learning-rate", "1e-4",  # Higher LR for larger batches
        "--early_stopping_patience", "3"
    ]
    print(" ".join(cmd4))
    print("→ Training batch size: 32, Evaluation batch size: 64")
    print()

    # Gradient accumulation for effective larger batch sizes
    print("5. Gradient accumulation (effective larger batch size with less memory):")
    cmd5 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_grad_acc",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--per-device-train-batch-size", "8",
        "--per-device-eval-batch-size", "16",
        "--gradient_accumulation_steps", "4",  # Accumulate 4 steps
        "--max_train_samples", "300",
        "--max_eval_samples", "150",
        "--learning-rate", "5e-5",
        "--early_stopping_patience", "3"
    ]
    print(" ".join(cmd5))
    print("→ Training batch size: 8, but effective batch size: 8 × 4 = 32")
    print()

    # Different train and eval batch sizes
    print("6. Different train and eval batch sizes:")
    cmd6 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_diff_bs",
        "--do_train", "--do_eval",
        "--num-train-epochs", "3",  # HuggingFace field name,
        "--per-device-train-batch-size", "12",  # Smaller for training
        "--per-device-eval-batch-size", "48",  # Larger for evaluation (no gradients)
        "--max_train_samples", "250",
        "--max_eval_samples", "125",
        "--learning-rate", "4e-5",
        "--early_stopping_patience", "2"
    ]
    print(" ".join(cmd6))
    print("→ Training batch size: 12, Evaluation batch size: 48")
    print()

def show_batch_size_tips():
    """Provide tips for choosing batch sizes."""

    print("=== Batch Size Selection Tips ===")
    print("💾 **Memory Considerations**:")
    print("   - Start with batch size 8 and increase if you have enough GPU memory")
    print("   - Monitor GPU memory usage during training")
    print("   - If you get OOM errors, reduce batch size")
    print()
    print("🎯 **Performance Trade-offs**:")
    print("   - Larger batches = faster training per epoch")
    print("   - Smaller batches = more frequent updates, potentially better convergence")
    print("   - Gradient accumulation can simulate larger batches with less memory")
    print()
    print("📊 **General Guidelines**:")
    print("   - Training batch size: 8-32 (depending on GPU memory)")
    print("   - Evaluation batch size: 16-64 (can be larger than training)")
    print("   - RoBERTa-base: Start with 16, adjust based on memory")
    print("   - BERT-base: Start with 16, adjust based on memory")
    print()
    print("⚡ **Gradient Accumulation**:")
    print("   - Use when you want effective large batch size but limited memory")
    print("   - Formula: effective_batch_size = per_device_batch_size × gradient_accumulation_steps")
    print("   - Example: batch_size=8, accumulation=4 → effective batch=32")
    print()

def show_memory_estimates():
    """Show memory estimates for different batch sizes."""

    print("=== Memory Usage Estimates (approximate) ===")
    print("For RoBERTa-base with max_length=128:")
    print()
    print("📏 **Model memory**:")
    print("   - RoBERTa-base: ~500MB (base model)")
    print("   - + Gradients: ~500MB")
    print("   - + Optimizer states: ~1GB")
    print("   - Base requirement: ~2GB")
    print()
    print("💾 **Batch memory addition**:")
    print("   - Batch size 8: ~1GB additional")
    print("   - Batch size 16: ~2GB additional")
    print("   - Batch size 32: ~4GB additional")
    print("   - Batch size 64: ~8GB additional")
    print()
    print("🖥️ **Total estimated VRAM needed**:")
    print("   - Batch size 8: ~3GB total")
    print("   - Batch size 16: ~4GB total")
    print("   - Batch size 32: ~6GB total")
    print("   - Batch size 64: ~10GB total")
    print()
    print("⚠️  These are estimates. Actual usage may vary!")
    print()

if __name__ == "__main__":
    print("Batch Size Configuration Guide for training_bert.py\n")

    # Show example commands
    test_batch_size_commands()

    # Show tips
    show_batch_size_tips()

    # Show memory estimates
    show_memory_estimates()

    print("🚀 Ready to configure your batch sizes!")
    print("💡 Start with --per-device-train-batch-size 16 and adjust based on your GPU memory")