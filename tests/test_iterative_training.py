"""
Test suite for iterative training and merging functionality.

This module provides comprehensive tests for the iterative training system,
including configuration, state management, merge coordination, and the full pipeline.
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.iterative_config import (
    IterativeOrchestratorConfig,
    IterativeTrainingConfig,
    IterativeMergeConfig,
    create_default_config
)
from merginguriel.training_state import (
    TrainingState,
    TrainingMetrics,
    CheckpointManager,
    TrainingStateManager
)
from merginguriel.merge_coordinator import MergeCoordinator
from merginguriel.adaptive_merging import (
    PerformanceTracker,
    AdaptiveMergeScheduler,
    EnhancedMonitor,
    PerformanceMetrics,
    MergeDecision
)


class TestIterativeConfig(unittest.TestCase):
    """Test configuration system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_default_config(self):
        """Test creating default configuration."""
        locales = ["en-US", "fr-FR", "de-DE"]
        config = create_default_config(
            locales=locales,
            base_output_dir=self.temp_dir
        )

        self.assertEqual(len(config.training_configs), 3)
        self.assertEqual(config.training_configs[0].locale, "en-US")
        self.assertEqual(config.training_configs[1].locale, "fr-FR")
        self.assertEqual(config.training_configs[2].locale, "de-DE")
        self.assertEqual(config.merge_config.target_languages, locales)

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        locales = ["en-US", "fr-FR"]
        config = create_default_config(
            locales=locales,
            base_output_dir=self.temp_dir,
            merge_frequency=5,
            merge_algorithm="fisher_simple"
        )

        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        config.save_config(config_path)
        self.assertTrue(os.path.exists(config_path))

        # Load configuration
        loaded_config = IterativeOrchestratorConfig.load_config(config_path)

        # Verify loaded config matches original
        self.assertEqual(len(loaded_config.training_configs), 2)
        self.assertEqual(loaded_config.merge_config.merge_frequency, 5)
        self.assertEqual(loaded_config.merge_config.merge_algorithm, "fisher_simple")

    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "orchestrator_name": "test_experiment",
            "base_output_dir": self.temp_dir,
            "training_configs": [
                {
                    "locale": "en-US",
                    "dataset_config_name": "en-US",
                    "max_epochs": 10,
                    "learning_rate": 3e-5,
                    "batch_size": 64
                }
            ],
            "merge_config": {
                "merge_frequency": 4,
                "merge_algorithm": "linear",
                "weight_calculation": "similarity"
            }
        }

        config = IterativeOrchestratorConfig.from_dict(config_dict)

        self.assertEqual(config.orchestrator_name, "test_experiment")
        self.assertEqual(len(config.training_configs), 1)
        self.assertEqual(config.training_configs[0].locale, "en-US")
        self.assertEqual(config.merge_config.merge_frequency, 4)
        self.assertEqual(config.merge_config.merge_algorithm, "linear")


class TestTrainingState(unittest.TestCase):
    """Test training state management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_metrics(self):
        """Test training metrics creation and serialization."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            eval_loss=0.6,
            eval_accuracy=0.85,
            learning_rate=5e-5
        )

        metrics_dict = metrics.to_dict()
        self.assertEqual(metrics_dict['epoch'], 1)
        self.assertEqual(metrics_dict['eval_accuracy'], 0.85)

        # Test round-trip serialization
        restored_metrics = TrainingMetrics.from_dict(metrics_dict)
        self.assertEqual(restored_metrics.epoch, metrics.epoch)
        self.assertEqual(restored_metrics.eval_accuracy, metrics.eval_accuracy)

    def test_model_state(self):
        """Test model state creation and integrity validation."""
        # Create a simple model state dict
        model_state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 5)
        }

        state = TrainingState(
            locale="en-US",
            model_name_or_path="test_model",
            checkpoint_path="/path/to/checkpoint",
            epoch=5,
            step=500,
            total_steps=1000,
            model_state_dict=model_state_dict
        )

        # Test checksum calculation
        checksum1 = state.calculate_checksum()
        checksum2 = state.calculate_checksum()
        self.assertEqual(checksum1, checksum2)

        # Test integrity validation
        self.assertTrue(state.validate_integrity())

        # Test integrity validation failure
        state.checksum = "invalid_checksum"
        self.assertFalse(state.validate_integrity())

    def test_checkpoint_manager(self):
        """Test checkpoint manager functionality."""
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        manager = CheckpointManager(checkpoint_dir, max_checkpoints=2)

        # Create a test model state
        model_state_dict = {
            "layer1.weight": torch.randn(10, 10)
        }
        state = ModelState(
            locale="en-US",
            model_name_or_path="test_model",
            checkpoint_path="",
            epoch=1,
            step=100,
            total_steps=1000,
            model_state_dict=model_state_dict
        )

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(state)
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "checkpoint_metadata.json")))

        # Load checkpoint
        loaded_state = manager.load_checkpoint(checkpoint_path)
        self.assertEqual(loaded_state.locale, state.locale)
        self.assertEqual(loaded_state.epoch, state.epoch)
        self.assertTrue(torch.equal(
            loaded_state.model_state_dict["layer1.weight"],
            state.model_state_dict["layer1.weight"]
        ))

        # Test checkpoint listing
        checkpoints = manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]['locale'], "en-US")

        # Test latest checkpoint
        latest = manager.get_latest_checkpoint("en-US")
        self.assertEqual(latest, checkpoint_path)

    def test_training_state_manager(self):
        """Test training state manager."""
        base_dir = os.path.join(self.temp_dir, "state_manager")
        manager = TrainingStateManager(base_dir)

        # Register models
        training_config = {
            "model_name_or_path": "test_model",
            "max_epochs": 10
        }
        manager.register_model("en-US", training_config)
        manager.register_model("fr-FR", training_config)

        # Update state
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            eval_accuracy=0.8
        )
        manager.update_state("en-US", epoch=1, step=100, metrics=metrics)

        # Get state
        state = manager.get_state("en-US")
        self.assertIsNotNone(state)
        self.assertEqual(state.locale, "en-US")
        self.assertEqual(state.epoch, 1)
        self.assertEqual(state.step, 100)

        # Create checkpoint
        checkpoint_path = manager.create_checkpoint("en-US")
        self.assertTrue(os.path.exists(checkpoint_path))

        # Test state summary
        manager.save_state_summary()
        summary_path = os.path.join(base_dir, "state_summary.json")
        self.assertTrue(os.path.exists(summary_path))


class TestAdaptiveMerging(unittest.TestCase):
    """Test adaptive merging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_performance_tracker(self):
        """Test performance tracking."""
        tracker = PerformanceTracker(max_history=100)

        # Add some metrics
        for i in range(10):
            metrics = PerformanceMetrics(
                locale="en-US",
                timestamp=i * 100,
                epoch=i // 100,
                step=i,
                train_loss=1.0 - i * 0.05,  # Decreasing loss
                eval_accuracy=0.5 + i * 0.03,  # Increasing accuracy
                learning_rate=5e-5
            )
            tracker.add_metrics(metrics)

        # Get convergence status
        status = tracker.get_convergence_status("en-US")
        self.assertEqual(status['status'], 'improving')
        self.assertLess(status['convergence_rate'], 0)  # Negative rate = improving

        # Get plateau locales (should be empty with improving metrics)
        plateau_locales = tracker.get_plateau_locales(threshold=1e-6)
        self.assertEqual(len(plateau_locales), 0)

        # Test with plateau metrics
        for i in range(10, 20):
            metrics = PerformanceMetrics(
                locale="en-US",
                timestamp=i * 100,
                epoch=i // 100,
                step=i,
                train_loss=0.5,  # Constant loss
                eval_accuracy=0.8,  # Constant accuracy
                learning_rate=5e-5
            )
            tracker.add_metrics(metrics)

        # Now should detect plateau
        plateau_locales = tracker.get_plateau_locales(threshold=1e-6, min_steps=5)
        self.assertIn("en-US", plateau_locales)

    def test_adaptive_merge_scheduler(self):
        """Test adaptive merge scheduling."""
        scheduler = AdaptiveMergeScheduler(
            base_merge_frequency=3,
            convergence_threshold=1e-4
        )

        # Simulate some training history
        for locale in ["en-US", "fr-FR"]:
            for epoch in range(5):
                metrics = PerformanceMetrics(
                    locale=locale,
                    timestamp=epoch * 1000,
                    epoch=epoch,
                    step=epoch * 100,
                    train_loss=1.0 - epoch * 0.1,
                    eval_accuracy=0.5 + epoch * 0.05,
                    learning_rate=5e-5
                )
                scheduler.performance_tracker.add_metrics(metrics)

        # Test merge decision
        decision = scheduler.evaluate_merge_necessity(
            current_epoch=3,
            active_locales=["en-US", "fr-FR"]
        )

        self.assertIsInstance(decision, MergeDecision)
        self.assertIsInstance(decision.should_merge, bool)
        self.assertIsInstance(decision.confidence, float)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertIsInstance(decision.reason, str)
        self.assertIsInstance(decision.recommended_merge_frequency, int)

    def test_enhanced_monitor(self):
        """Test enhanced monitoring."""
        monitor = EnhancedMonitor(self.temp_dir)

        # Log some training steps
        for step in range(5):
            metrics = {
                "train_loss": 1.0 - step * 0.1,
                "eval_accuracy": 0.5 + step * 0.05,
                "learning_rate": 5e-5
            }
            system_metrics = {
                "memory_usage_percent": 50.0 + step
            }
            monitor.log_training_step(
                locale="en-US",
                epoch=1,
                step=step,
                metrics=metrics,
                system_metrics=system_metrics
            )

        # Get recent metrics
        recent = monitor.get_recent_metrics("en-US", last_n=3)
        self.assertEqual(len(recent), 3)

        # Trigger an alert
        alert_metrics = {
            "train_loss": float('nan')  # This should trigger an alert
        }
        monitor.log_training_step(
            locale="en-US",
            epoch=2,
            step=10,
            metrics=alert_metrics
        )

        # Check alerts
        alert_summary = monitor.get_alert_summary()
        self.assertGreater(alert_summary['total_alerts'], 0)

        # Save monitoring data
        monitor.save_monitoring_data()
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "training_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "training_alerts.json")))


class TestMergeCoordinator(unittest.TestCase):
    """Test merge coordination."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merge_coordinator_creation(self):
        """Test merge coordinator creation."""
        merge_config = IterativeMergeConfig(
            merge_frequency=3,
            merge_algorithm="linear",
            weight_calculation="similarity"
        )
        state_manager = TrainingStateManager(self.temp_dir)

        coordinator = MergeCoordinator(
            merge_config=merge_config,
            training_state_manager=state_manager,
            base_output_dir=self.temp_dir
        )

        self.assertIsNotNone(coordinator.merge_config)
        self.assertIsNotNone(coordinator.state_manager)
        self.assertFalse(coordinator.merge_in_progress)

    def test_merge_decision_timing(self):
        """Test merge timing decisions."""
        merge_config = IterativeMergeConfig(
            merge_frequency=3,
            merge_frequency_unit="epoch",
            performance_merge_trigger=True,
            convergence_threshold=1e-4
        )
        state_manager = TrainingStateManager(self.temp_dir)
        coordinator = MergeCoordinator(merge_config, state_manager, self.temp_dir)

        # Test epoch-based triggering
        self.assertFalse(coordinator.should_merge(
            current_epoch=1,  # Not a multiple of 3
            current_step=100,
            metrics_history=[]
        ))

        self.assertTrue(coordinator.should_merge(
            current_epoch=3,  # Multiple of 3
            current_step=300,
            metrics_history=[]
        ))

        # Test step-based triggering
        merge_config.merge_frequency_unit = "steps"
        merge_config.merge_steps = 500

        self.assertFalse(coordinator.should_merge(
            current_epoch=1,
            current_step=100,  # Not a multiple of 500
            metrics_history=[]
        ))

        self.assertTrue(coordinator.should_merge(
            current_epoch=1,
            current_step=500,  # Multiple of 500
            metrics_history=[]
        ))

    def test_merge_statistics(self):
        """Test merge statistics tracking."""
        merge_config = IterativeMergeConfig()
        state_manager = TrainingStateManager(self.temp_dir)
        coordinator = MergeCoordinator(merge_config, state_manager, self.temp_dir)

        # Initially should have no statistics
        stats = coordinator.get_merge_statistics()
        self.assertEqual(stats['total_merges'], 0)
        self.assertEqual(stats['successful_merges'], 0)
        self.assertEqual(stats['failed_merges'], 0)

        # Record a successful merge
        coordinator._record_merge_success(
            active_locales=["en-US", "fr-FR"],
            target_locales=["en-US"],
            metadata={"test": True}
        )

        stats = coordinator.get_merge_statistics()
        self.assertEqual(stats['total_merges'], 1)
        self.assertEqual(stats['successful_merges'], 1)
        self.assertEqual(stats['failed_merges'], 0)

        # Record a failed merge
        coordinator._record_merge_failure(
            active_locales=["en-US"],
            target_locales=["en-US"],
            error_message="Test error"
        )

        stats = coordinator.get_merge_statistics()
        self.assertEqual(stats['total_merges'], 2)
        self.assertEqual(stats['successful_merges'], 1)
        self.assertEqual(stats['failed_merges'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('datasets.load_dataset')
    def test_config_to_training_pipeline(self, mock_dataset, mock_tokenizer, mock_model):
        """Test integration from configuration to training pipeline setup."""
        # Mock the heavy dependencies
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_dataset.return_value = Mock()

        # Create configuration
        config = create_default_config(
            locales=["en-US", "fr-FR"],
            base_output_dir=self.temp_dir,
            merge_frequency=2
        )

        # Verify configuration structure
        self.assertEqual(len(config.training_configs), 2)
        self.assertEqual(config.merge_config.merge_frequency, 2)
        self.assertTrue(config.enable_auto_recovery)

        # Verify configuration serialization
        config_path = os.path.join(self.temp_dir, "integration_config.json")
        config.save_config(config_path)
        loaded_config = IterativeOrchestratorConfig.load_config(config_path)

        self.assertEqual(
            len(loaded_config.training_configs),
            len(config.training_configs)
        )
        self.assertEqual(
            loaded_config.merge_config.merge_frequency,
            config.merge_config.merge_frequency
        )

    def test_state_management_workflow(self):
        """Test complete state management workflow."""
        # Create state manager
        state_manager = TrainingStateManager(self.temp_dir)

        # Register models
        locales = ["en-US", "fr-FR", "de-DE"]
        for locale in locales:
            training_config = {
                "model_name_or_path": f"model_{locale}",
                "max_epochs": 10
            }
            state_manager.register_model(locale, training_config)

        # Simulate training progress
        for epoch in range(1, 4):
            for i, locale in enumerate(locales):
                step = epoch * 100 + i * 10
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    train_loss=1.0 - epoch * 0.1,
                    eval_accuracy=0.5 + epoch * 0.05,
                    learning_rate=5e-5
                )

                state_manager.update_state(
                    locale=locale,
                    epoch=epoch,
                    step=step,
                    metrics=metrics
                )

        # Create checkpoints for all models
        checkpoint_paths = {}
        for locale in locales:
            checkpoint_path = state_manager.create_checkpoint(locale)
            checkpoint_paths[locale] = checkpoint_path
            self.assertTrue(os.path.exists(checkpoint_path))

        # Verify all states are tracked
        all_states = state_manager.get_all_states()
        self.assertEqual(len(all_states), 3)

        for locale in locales:
            state = all_states[locale]
            self.assertEqual(state.epoch, 3)  # Last epoch
            self.assertIsNotNone(state.current_metrics)

        # Save state summary
        state_manager.save_state_summary()
        summary_path = os.path.join(self.temp_dir, "state_summary.json")
        self.assertTrue(os.path.exists(summary_path))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestIterativeConfig,
        TestTrainingState,
        TestAdaptiveMerging,
        TestMergeCoordinator,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)