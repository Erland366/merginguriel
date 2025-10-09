#!/usr/bin/env python
"""
Test script to verify early stopping integration with training_bert.py

This script demonstrates how to use the early stopping feature and shows example commands.
"""

import subprocess
import sys

def test_early_stopping_commands():
    """Generate example commands for testing early stopping integration."""

    print("=== Early Stopping Integration Test Commands ===\n")

    # Basic training with early stopping (default patience=3)
    print("1. Basic training with early stopping (default settings):")
    cmd1 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_early_stop",
        "--do_train", "--do_eval",
        "--num_train_epochs", "10",  # Set high to allow early stopping
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "16",
        "--logging_steps", "50",
        "--max_train_samples", "200",
        "--max_eval_samples", "100",
        "--early_stopping_patience", "3",  # Default: wait 3 epochs without improvement
        # Run name will be auto-generated: roberta-base_massive_en-US_lr5e-5_ep10
    ]
    print(" ".join(cmd1))
    print("→ Expected behavior: Training will stop if no improvement for 3 evaluations")
    print("→ Best model will be loaded at the end")
    print()

    # Training with custom early stopping settings
    print("2. Training with custom early stopping settings:")
    cmd2 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_custom_early_stop",
        "--do_train", "--do_eval",
        "--num_train_epochs", "15",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--logging_steps", "25",
        "--max_train_samples", "300",
        "--max_eval_samples", "150",
        "--learning_rate", "3e-5",
        "--early_stopping_patience", "5",  # More patient: wait 5 epochs
        "--early_stopping_threshold", "0.001",  # Require minimum improvement of 0.001
        "--wandb_tags", "early-stopping,patient,roberta"
    ]
    print(" ".join(cmd2))
    print("→ Expected behavior: Training will stop if no improvement > 0.001 for 5 evaluations")
    print()

    # Training with very aggressive early stopping
    print("3. Training with aggressive early stopping:")
    cmd3 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_aggressive_early_stop",
        "--do_train", "--do_eval",
        "--num_train_epochs", "8",
        "--per_device_train_batch_size", "16",
        "--logging_steps", "20",
        "--max_train_samples", "150",
        "--max_eval_samples", "75",
        "--learning_rate", "2e-5",
        "--early_stopping_patience", "1",  # Very aggressive: stop after 1 bad evaluation
        "--wandb_tags", "early-stopping,aggressive,quick"
    ]
    print(" ".join(cmd3))
    print("→ Expected behavior: Training will stop immediately after first non-improving evaluation")
    print()

    # Training with different model and early stopping
    print("4. BERT with early stopping:")
    cmd4 = [
        "python", "training_bert.py",
        "--model_name_or_path", "bert-base-uncased",
        "--output_dir", "./test_results_bert_early_stop",
        "--do_train", "--do_eval",
        "--num_train_epochs", "12",
        "--per_device_train_batch_size", "16",
        "--logging_steps", "30",
        "--max_train_samples", "250",
        "--max_eval_samples", "125",
        "--learning_rate", "2e-5",
        "--early_stopping_patience", "4",
        "--wandb_tags", "bert,early-stopping,comparison"
    ]
    print(" ".join(cmd4))
    print("→ Expected run name: bert-base-uncased_massive_en-US_lr2e-5_ep12")
    print()

    # Training without early stopping (baseline comparison)
    print("5. Training without early stopping (baseline):")
    cmd5 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_no_early_stop",
        "--do_train", "--do_eval",
        "--num_train_epochs", "5",  # Fixed number of epochs
        "--per_device_train_batch_size", "16",
        "--logging_steps", "40",
        "--max_train_samples", "200",
        "--max_eval_samples", "100",
        "--evaluation_strategy", "epoch",  # Still evaluate each epoch
        "--save_strategy", "epoch",
        "--load_best_model_at_end", "True",
        "--wandb_tags", "baseline,no-early-stopping"
        # Note: early_stopping_patience not set, so no early stopping
    ]
    print(" ".join(cmd5))
    print("→ Expected behavior: Training will run for all 5 epochs regardless of performance")
    print()

def check_early_stopping_benefits():
    """Explain the benefits of early stopping."""

    print("=== Early Stopping Benefits ===")
    print("🎯 **Prevents Overfitting**: Stops training when validation performance degrades")
    print("⏱️ **Saves Training Time**: Avoids unnecessary training epochs")
    print("💾 **Best Model Selection**: Automatically keeps the best performing model")
    print("📊 **Resource Efficient**: Reduces GPU usage and compute costs")
    print("🔧 **Configurable**: Adjustable patience and improvement thresholds")
    print("📈 **Better Generalization**: Typically results in better test performance")
    print()

    print("=== Configuration Options ===")
    print("`--early_stopping_patience`: Number of evaluations without improvement before stopping")
    print("  - Default: 3 (wait for 3 evaluations without improvement)")
    print("  - Higher values = more patient (longer training)")
    print("  - Lower values = more aggressive (quicker stopping)")
    print()
    print("`--early_stopping_threshold`: Minimum improvement required")
    print("  - Default: 0.0 (any improvement counts)")
    print("  - Higher values = require more significant improvement")
    print()

    print("=== Automatic Configuration ===")
    print("When early stopping is enabled, the script automatically configures:")
    print("✅ `evaluation_strategy=epoch` (evaluate each epoch)")
    print("✅ `save_strategy=epoch` (save checkpoint each epoch)")
    print("✅ `load_best_model_at_end=True` (load best model at end)")
    print("✅ `metric_for_best_model=eval_accuracy` (monitor validation accuracy)")
    print("✅ `greater_is_better=True` (higher accuracy is better)")
    print()

def show_monitoring_tips():
    """Provide tips for monitoring early stopping experiments."""

    print("=== Monitoring Tips ===")
    print("📊 **Watch the validation curve**: Look for overfitting patterns")
    print("🔍 **Check patience setting**: If stopping too early, increase patience")
    print("📈 **Set appropriate threshold**: Avoid stopping on insignificant improvements")
    print("💾 **Best model is saved**: The final model is the best one, not the last epoch")
    print("🌐 **Wandb integration**: Early stopping decisions are logged to wandb")
    print()

if __name__ == "__main__":
    print("Testing Early Stopping Integration for training_bert.py\n")

    # Show example commands
    test_early_stopping_commands()

    # Explain benefits
    check_early_stopping_benefits()

    # Provide monitoring tips
    show_monitoring_tips()

    print("🚀 Ready to run experiments with early stopping!")
    print("💡 Tip: Start with default patience=3 and adjust based on your results")