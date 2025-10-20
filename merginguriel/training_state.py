"""
Training state management for iterative training and merging.

This module provides data structures and utilities for managing the state of
individual model trainers during iterative training, including checkpoint
management and state synchronization.
"""

import os
import json
import torch
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None
    learning_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelState:
    """Complete state of a model including weights, optimizer, and training info."""

    # Model identification
    locale: str
    model_name_or_path: str
    checkpoint_path: str

    # Training state
    epoch: int
    step: int
    total_steps: int

    # Model weights and optimizer state
    model_state_dict: Optional[Dict[str, torch.Tensor]] = None
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None

    # Metrics and performance
    current_metrics: Optional[TrainingMetrics] = None
    best_metrics: Optional[TrainingMetrics] = None
    history: List[TrainingMetrics] = field(default_factory=list)

    # Training configuration
    training_config: Optional[Dict[str, Any]] = None

    # Checkpoint metadata
    checkpoint_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checkpoint_version: str = "1.0"

    # Integrity validation
    checksum: Optional[str] = None

    def calculate_checksum(self) -> str:
        """Calculate checksum for state integrity validation."""
        import hashlib
        state_str = json.dumps({
            'locale': self.locale,
            'epoch': self.epoch,
            'step': self.step,
            'checkpoint_path': self.checkpoint_path,
            'timestamp': self.checkpoint_timestamp
        }, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def validate_integrity(self) -> bool:
        """Validate the integrity of the model state."""
        if not self.checksum:
            return True  # No checksum to validate against

        calculated_checksum = self.calculate_checksum()
        return calculated_checksum == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert tensors to CPU for serialization
        if self.model_state_dict:
            result['model_state_dict'] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.model_state_dict.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelState":
        """Create from dictionary."""
        # Convert TrainingMetrics objects
        if data.get('current_metrics'):
            data['current_metrics'] = TrainingMetrics.from_dict(data['current_metrics'])
        if data.get('best_metrics'):
            data['best_metrics'] = TrainingMetrics.from_dict(data['best_metrics'])
        if data.get('history'):
            data['history'] = [TrainingMetrics.from_dict(m) for m in data['history']]

        return cls(**data)


class CheckpointManager:
    """Manages checkpoint creation, loading, and cleanup for iterative training."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model_state: ModelState,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a complete checkpoint of the model state.

        Args:
            model_state: Complete model state to save
            include_optimizer: Whether to include optimizer state
            include_scheduler: Whether to include scheduler state
            metadata: Additional metadata to include

        Returns:
            Path to the saved checkpoint
        """
        # Create checkpoint directory
        checkpoint_name = f"checkpoint_epoch_{model_state.epoch}_step_{model_state.step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)

        # Calculate and set checksum
        model_state.checksum = model_state.calculate_checksum()

        # Prepare checkpoint data
        checkpoint_data = {
            'model_state': model_state.to_dict(),
            'metadata': metadata or {},
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Save model state
        if model_state.model_state_dict:
            torch.save(
                model_state.model_state_dict,
                checkpoint_path / "pytorch_model.bin"
            )

        # Save optimizer state if requested
        if include_optimizer and model_state.optimizer_state_dict:
            torch.save(
                model_state.optimizer_state_dict,
                checkpoint_path / "optimizer.bin"
            )

        # Save scheduler state if requested
        if include_scheduler and model_state.scheduler_state_dict:
            torch.save(
                model_state.scheduler_state_dict,
                checkpoint_path / "scheduler.bin"
            )

        # Save metadata
        with open(checkpoint_path / "checkpoint_metadata.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> ModelState:
        """
        Load a checkpoint and return the model state.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            Loaded model state
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load metadata
        with open(checkpoint_path / "checkpoint_metadata.json", 'r') as f:
            checkpoint_data = json.load(f)

        # Reconstruct model state
        model_state_dict = checkpoint_data['model_state']

        # Load model weights
        model_weights_path = checkpoint_path / "pytorch_model.bin"
        if model_weights_path.exists():
            model_state_dict['model_state_dict'] = torch.load(
                model_weights_path, map_location='cpu'
            )

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.bin"
        if optimizer_path.exists():
            model_state_dict['optimizer_state_dict'] = torch.load(
                optimizer_path, map_location='cpu'
            )

        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.bin"
        if scheduler_path.exists():
            model_state_dict['scheduler_state_dict'] = torch.load(
                scheduler_path, map_location='cpu'
            )

        model_state = ModelState.from_dict(model_state_dict)

        # Validate integrity
        if not model_state.validate_integrity():
            logger.warning(f"Checkpoint integrity check failed: {checkpoint_path}")

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return model_state

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []

        for checkpoint_dir in self.checkpoint_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                try:
                    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        checkpoints.append({
                            'path': str(checkpoint_dir),
                            'epoch': metadata['model_state']['epoch'],
                            'step': metadata['model_state']['step'],
                            'timestamp': metadata['timestamp'],
                            'locale': metadata['model_state']['locale']
                        })
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint metadata from {checkpoint_dir}: {e}")

        # Sort by epoch and step
        checkpoints.sort(key=lambda x: (x['epoch'], x['step']))
        return checkpoints

    def get_latest_checkpoint(self, locale: Optional[str] = None) -> Optional[str]:
        """Get the path to the latest checkpoint for a given locale."""
        checkpoints = self.list_checkpoints()

        if locale:
            checkpoints = [cp for cp in checkpoints if cp['locale'] == locale]

        if not checkpoints:
            return None

        return checkpoints[-1]['path']

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Remove oldest checkpoints
        to_remove = len(checkpoints) - self.max_checkpoints
        for i in range(to_remove):
            checkpoint_path = Path(checkpoints[i]['path'])
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint_path}: {e}")


class TrainingStateManager:
    """Manages training states for multiple models during iterative training."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.states: Dict[str, ModelState] = {}
        self.checkpoint_managers: Dict[str, CheckpointManager] = {}

    def register_model(self, locale: str, training_config: Dict[str, Any]):
        """Register a model for state tracking."""
        checkpoint_dir = self.base_dir / locale / "checkpoints"
        self.checkpoint_managers[locale] = CheckpointManager(str(checkpoint_dir))

        # Initialize state if not exists
        if locale not in self.states:
            self.states[locale] = ModelState(
                locale=locale,
                model_name_or_path=training_config.get('model_name_or_path', ''),
                checkpoint_path='',
                epoch=0,
                step=0,
                total_steps=0,
                training_config=training_config
            )

    def update_state(
        self,
        locale: str,
        epoch: int,
        step: int,
        model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[TrainingMetrics] = None
    ):
        """Update the training state for a model."""
        if locale not in self.states:
            raise ValueError(f"Model {locale} not registered")

        state = self.states[locale]
        state.epoch = epoch
        state.step = step
        state.current_metrics = metrics

        if model_state_dict:
            state.model_state_dict = model_state_dict
        if optimizer_state_dict:
            state.optimizer_state_dict = optimizer_state_dict
        if scheduler_state_dict:
            state.scheduler_state_dict = scheduler_state_dict

        if metrics:
            state.history.append(metrics)
            # Update best metrics
            if (state.best_metrics is None or
                (metrics.eval_accuracy and
                 metrics.eval_accuracy > (state.best_metrics.eval_accuracy or 0))):
                state.best_metrics = metrics

        # Calculate and set checksum for integrity validation
        state.checksum = state.calculate_checksum()

    def create_checkpoint(
        self,
        locale: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a checkpoint for the specified model."""
        if locale not in self.states:
            raise ValueError(f"Model {locale} not registered")

        state = self.states[locale]
        checkpoint_manager = self.checkpoint_managers[locale]

        checkpoint_path = checkpoint_manager.save_checkpoint(
            state, metadata=metadata
        )
        state.checkpoint_path = checkpoint_path

        return checkpoint_path

    def load_from_checkpoint(self, locale: str, checkpoint_path: str) -> ModelState:
        """Load model state from checkpoint."""
        checkpoint_manager = self.checkpoint_managers.get(locale)
        if not checkpoint_manager:
            checkpoint_manager = CheckpointManager(
                str(self.base_dir / locale / "checkpoints")
            )
            self.checkpoint_managers[locale] = checkpoint_manager

        state = checkpoint_manager.load_checkpoint(checkpoint_path)
        self.states[locale] = state

        return state

    def get_state(self, locale: str) -> Optional[ModelState]:
        """Get the current state for a model."""
        return self.states.get(locale)

    def get_all_states(self) -> Dict[str, ModelState]:
        """Get all current model states."""
        return self.states.copy()

    def save_state_summary(self):
        """Save a summary of all current states."""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'models': {}
        }

        for locale, state in self.states.items():
            summary['models'][locale] = {
                'locale': state.locale,
                'epoch': state.epoch,
                'step': state.step,
                'checkpoint_path': state.checkpoint_path,
                'current_metrics': state.current_metrics.to_dict() if state.current_metrics else None,
                'best_metrics': state.best_metrics.to_dict() if state.best_metrics else None
            }

        summary_path = self.base_dir / "state_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved state summary: {summary_path}")