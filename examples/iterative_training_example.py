#!/usr/bin/env python
"""
Example script demonstrating iterative training and merging.

This script shows how to use the iterative training system with a simple
configuration for demonstration purposes.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.iterative_config import create_default_config
from merginguriel.run_iterative_training import create_orchestrator_config_from_args, print_experiment_summary
from merginguriel.training_state import TrainingMetrics
from merginguriel.adaptive_merging import PerformanceTracker, AdaptiveMergeScheduler


def create_example_config():
    """Create an example configuration for demonstration."""
    # Use a temporary directory for this example
    temp_dir = tempfile.mkdtemp(prefix="iterative_training_example_")

    # Create configuration for 3 locales with frequent merging for demo
    config = create_default_config(
        locales=["en-US", "fr-FR", "de-DE"],
        base_output_dir=temp_dir,
        merge_frequency=2,  # Merge every 2 epochs for demo
        merge_algorithm="linear",
        weight_calculation="similarity"
    )

    # Update training parameters for faster demo
    for training_config in config.training_configs:
        training_config.max_epochs = 6  # Short training for demo
        training_config.batch_size = 32  # Smaller batch size
        training_config.learning_rate = 1e-4  # Higher learning rate

    # Enable advanced features
    config.adaptive_merge_frequency = True
    config.performance_merge_trigger = True
    config.enable_wandb = False  # Disable wandb for demo
    config.validate_merge_integrity = True

    return config, temp_dir


def demo_performance_tracking():
    """Demonstrate performance tracking functionality."""
    print("="*60)
    print("DEMO: Performance Tracking")
    print("="*60)

    # Create performance tracker
    tracker = PerformanceTracker(max_history=100)

    # Simulate training progress for one locale
    locale = "en-US"
    print(f"Simulating training progress for {locale}...")

    for epoch in range(5):
        for step in range(10):
            # Simulate improving metrics
            train_loss = 1.0 - (epoch * 10 + step) * 0.005
            eval_accuracy = 0.5 + (epoch * 10 + step) * 0.003

            from merginguriel.adaptive_merging import PerformanceMetrics
            metrics = PerformanceMetrics(
                locale=locale,
                timestamp=epoch * 1000 + step * 100,
                epoch=epoch,
                step=epoch * 10 + step,
                train_loss=train_loss,
                eval_accuracy=eval_accuracy,
                learning_rate=5e-5
            )

            tracker.add_metrics(metrics)

            if step % 5 == 0:
                print(f"  Epoch {epoch}, Step {epoch * 10 + step}: "
                      f"Loss={train_loss:.3f}, Acc={eval_accuracy:.3f}")

    # Analyze convergence
    convergence_status = tracker.get_convergence_status(locale)
    print(f"\nConvergence Status for {locale}:")
    print(f"  Status: {convergence_status['status']}")
    print(f"  Convergence Rate: {convergence_status['convergence_rate']:.6f}")
    print(f"  Recent Loss: {convergence_status['recent_loss']:.3f}")
    print(f"  Recent Accuracy: {convergence_status['recent_accuracy']:.3f}")
    print(f"  Trend: {convergence_status['trend']:.6f}")


def demo_adaptive_scheduling():
    """Demonstrate adaptive merge scheduling."""
    print("\n" + "="*60)
    print("DEMO: Adaptive Merge Scheduling")
    print("="*60)

    # Create adaptive scheduler
    scheduler = AdaptiveMergeScheduler(
        base_merge_frequency=3,
        convergence_threshold=1e-4
    )

    # Simulate multiple locales with different convergence patterns
    locales = ["en-US", "fr-FR", "de-DE"]

    print("Simulating training progress for multiple locales...")

    for epoch in range(4):
        for locale in locales:
            # Simulate different convergence patterns
            if locale == "en-US":
                # Fast convergence
                train_loss = 1.0 - epoch * 0.2
                eval_accuracy = 0.5 + epoch * 0.1
            elif locale == "fr-FR":
                # Slow convergence
                train_loss = 1.0 - epoch * 0.1
                eval_accuracy = 0.5 + epoch * 0.05
            else:  # de-DE
                # Plateau after initial improvement
                train_loss = max(0.5, 1.0 - epoch * 0.15)
                eval_accuracy = min(0.7, 0.5 + epoch * 0.08)

            from merginguriel.adaptive_merging import PerformanceMetrics
            metrics = PerformanceMetrics(
                locale=locale,
                timestamp=epoch * 1000,
                epoch=epoch,
                step=epoch * 100,
                train_loss=train_loss,
                eval_accuracy=eval_accuracy,
                learning_rate=5e-5
            )

            scheduler.performance_tracker.add_metrics(metrics)

        # Evaluate merge necessity at each epoch
        decision = scheduler.evaluate_merge_necessity(
            current_epoch=epoch,
            active_locales=locales
        )

        print(f"\nEpoch {epoch}:")
        print(f"  Should Merge: {decision.should_merge}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Reason: {decision.reason}")
        print(f"  Recommended Frequency: {decision.recommended_merge_frequency}")
        print(f"  Suggested Algorithm: {decision.suggested_algorithm}")

        if decision.should_merge:
            scheduler.record_merge(
                epoch=epoch,
                locales=locales,
                algorithm=decision.suggested_algorithm or "linear",
                outcome="success"
            )
            print(f"  -> MERGE EXECUTED at epoch {epoch}")

    # Show final statistics
    stats = scheduler.get_merge_statistics()
    print(f"\nFinal Merge Statistics:")
    print(f"  Total Merges: {stats['total_merges']}")
    print(f"  Successful Merges: {stats['successful_merges']}")
    print(f"  Success Rate: {stats['overall_success_rate']:.2%}")
    print(f"  Algorithm Performance: {stats['algorithm_statistics']}")


def demo_configuration():
    """Demonstrate configuration creation and management."""
    print("\n" + "="*60)
    print("DEMO: Configuration Management")
    print("="*60)

    # Create example configuration
    config, temp_dir = create_example_config()

    print(f"Created example configuration in: {temp_dir}")
    print_experiment_summary(config)

    # Save configuration
    config_path = os.path.join(temp_dir, "example_config.json")
    config.save_config(config_path)
    print(f"\nConfiguration saved to: {config_path}")

    # Demonstrate loading configuration
    print("Loading configuration from file...")
    loaded_config = config.__class__.load_config(config_path)
    print(f"Loaded configuration for {len(loaded_config.training_configs)} locales")

    # Clean up
    print(f"\nCleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_demonstrations():
    """Run all demonstrations."""
    print("Iterative Training and Merging System - Demonstrations")
    print("=" * 60)
    print("This script demonstrates the key features of the iterative training system.")
    print("Note: These are simulations - no actual model training is performed.")
    print()

    try:
        # Run demonstrations
        demo_configuration()
        demo_performance_tracking()
        demo_adaptive_scheduling()

        print("\n" + "="*60)
        print("DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review the generated code and documentation")
        print("2. Run the actual training with your own data:")
        print("   python merginguriel/run_iterative_training.py --locales en-US,fr-FR --max-epochs 10")
        print("3. Monitor training progress and merge operations")
        print("4. Analyze results and compare with traditional post-training merging")

    except Exception as e:
        print(f"\nError during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demonstrations()