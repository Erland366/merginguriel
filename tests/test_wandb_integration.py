#!/usr/bin/env python
"""
Test script to verify wandb integration with training_bert.py

This script demonstrates how to use the wandb-enabled training script with various options.
"""

import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

def test_wandb_command():
    """Generate example commands for testing wandb integration"""

    print("=== Wandb Integration Test Commands ===\n")

    # Basic training with wandb (project: MergingUriel) - auto-named
    print("1. Basic training with auto-generated run name:")
    cmd1 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results",
        "--do_train", "--do_eval",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "16",
        "--logging_steps", "10",
        "--max_train_samples", "100",  # Small sample for testing
        "--max_eval_samples", "50"
        # Run name will be auto-generated: roberta-base_massive_en-US_lr5e-5_ep1
    ]
    print(" ".join(cmd1))
    print("→ Expected run name: roberta-base_massive_en-US_lr5e-5_ep1")
    print()

    # Training with custom wandb settings - auto-named
    print("2. Training with custom learning rate and epochs:")
    cmd2 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_custom",
        "--do_train", "--do_eval",
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--logging_steps", "5",
        "--max_train_samples", "200",
        "--max_eval_samples", "100",
        "--wandb_project", "MergingUriel",
        "--wandb_tags", "roberta,massive,intent-classification,test",
        "--learning_rate", "3e-5"
        # Run name will be auto-generated: roberta-base_massive_en-US_lr3e-5_ep2
    ]
    print(" ".join(cmd2))
    print("→ Expected run name: roberta-base_massive_en-US_lr3e-5_ep2")
    print()

    # Offline mode testing - auto-named
    print("3. Offline mode testing:")
    cmd3 = [
        "python", "training_bert.py",
        "--output_dir", "./test_results_offline",
        "--do_train", "--do_eval",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "16",
        "--logging_steps", "20",
        "--max_train_samples", "50",
        "--max_eval_samples", "25",
        "--wandb_offline"
        # Run name will be auto-generated: roberta-base_massive_en-US_lr5e-5_ep1
    ]
    print(" ".join(cmd3))
    print("→ Expected run name: roberta-base_massive_en-US_lr5e-5_ep1")
    print()

    # With different model - auto-named
    print("4. Testing with different model:")
    cmd4 = [
        "python", "training_bert.py",
        "--model_name_or_path", "bert-base-uncased",
        "--output_dir", "./test_results_bert",
        "--do_train", "--do_eval",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "16",
        "--logging_steps", "15",
        "--max_train_samples", "100",
        "--max_eval_samples", "50",
        "--wandb_tags", "bert,baseline,massive",
        "--learning_rate", "2e-5"
        # Run name will be auto-generated: bert-base-uncased_massive_en-US_lr2e-5_ep1
    ]
    print(" ".join(cmd4))
    print("→ Expected run name: bert-base-uncased_massive_en-US_lr2e-5_ep1")
    print()

    # With different dataset config
    print("5. Testing with different dataset config:")
    cmd5 = [
        "python", "training_bert.py",
        "--model_name_or_path", "FacebookAI/roberta-base",
        "--dataset_config_name", "fr-FR",
        "--output_dir", "./test_results_french",
        "--do_train", "--do_eval",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "16",
        "--learning_rate", "1e-4",
        "--max_train_samples", "100",
        "--max_eval_samples", "50",
        "--wandb_tags", "french,roberta,massive"
        # Run name will be auto-generated: roberta-base_massive_fr-FR_lr1e-4_ep3
    ]
    print(" ".join(cmd5))
    print("→ Expected run name: roberta-base_massive_fr-FR_lr1e-4_ep3")
    print()

def check_wandb_availability():
    """Check if wandb is available and configured"""
    try:
        import wandb
        print("✅ wandb is available")

        # Check if user is logged in
        try:
            api = wandb.Api()
            print("✅ wandb API access available")
        except Exception:
            print("⚠️  wandb not logged in. Run 'wandb login' to sync experiments online")
            print("   (Offline mode will still work)")

    except ImportError:
        print("❌ wandb not available. Install with: pip install wandb")
        return False

    return True

if __name__ == "__main__":
    print("Testing wandb integration for training_bert.py\n")

    # Check wandb availability
    wandb_available = check_wandb_availability()

    print("\n" + "="*50)

    # Show example commands
    test_wandb_command()

    print("=== Key Features Added ===")
    print("✅ Native wandb integration via Hugging Face Trainer")
    print("✅ Project name: MergingUriel")
    print("✅ Auto-generated descriptive run names")
    print("✅ Format: {model}_{dataset}_{config}_lr{lr}_ep{epochs}")
    print("✅ Automatic logging of:")
    print("   - Training metrics (loss, accuracy, learning rate)")
    print("   - Model parameters and architecture")
    print("   - Dataset statistics and class distribution")
    print("   - System metrics (GPU usage, memory)")
    print("   - Final evaluation results")
    print("✅ Customizable tags and project settings")
    print("✅ Offline mode support")
    print("✅ Error handling for graceful fallback")

    print("\n=== Auto-Generated Run Name Examples ===")
    print("• roberta-base_massive_en-US_lr5e-5_ep3")
    print("• bert-base-uncased_massive_en-US_lr3e-5_ep5")
    print("• roberta-large_massive_fr-FR_lr1e-4_ep2")

    print("\n=== Usage Tips ===")
    print("1. Set up wandb: wandb login")
    print("2. View experiments: https://wandb.ai/your-username/MergingUriel")
    print("3. Use --wandb_offline for runs without internet")
    print("4. Customize project with --wandb_project")
    print("5. Add team workspace with --wandb_entity")
    print("6. Add tags with --wandb_tags (e.g., 'bert,experiment,test')")
    print("7. Run names are auto-generated from model/dataset/lr/epochs")

    if wandb_available:
        print("\n🚀 Ready to run experiments with wandb tracking!")
    else:
        print("\n⚠️  Install wandb first to enable experiment tracking")