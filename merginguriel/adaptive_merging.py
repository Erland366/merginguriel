"""
Adaptive merging and enhanced monitoring for iterative training.

This module provides advanced features for dynamic merge scheduling,
performance-based triggering, and comprehensive monitoring of the
iterative training process.
"""

from merginguriel import logger
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

# Loguru logger imported from merginguriel package


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single locale."""
    locale: str
    timestamp: float
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None
    learning_rate: float = 0.0
    convergence_rate: float = 0.0
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'locale': self.locale,
            'timestamp': self.timestamp,
            'epoch': self.epoch,
            'step': self.step,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'eval_accuracy': self.eval_accuracy,
            'eval_f1': self.eval_f1,
            'learning_rate': self.learning_rate,
            'convergence_rate': self.convergence_rate,
            'gradient_norm': self.gradient_norm,
            'memory_usage': self.memory_usage
        }


@dataclass
class MergeDecision:
    """Decision about whether to merge and the reasoning."""
    should_merge: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    metrics: Dict[str, float]
    recommended_merge_frequency: Optional[int] = None
    suggested_algorithm: Optional[str] = None


class PerformanceTracker:
    """Tracks and analyzes performance metrics across multiple models."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = {}
        self.convergence_rates: Dict[str, float] = {}
        self.performance_trends: Dict[str, List[float]] = {}

    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics for a locale."""
        locale = metrics.locale

        # Initialize history for new locale
        if locale not in self.metrics_history:
            self.metrics_history[locale] = deque(maxlen=self.max_history)
            self.convergence_rates[locale] = 0.0
            self.performance_trends[locale] = []

        # Add metrics to history
        self.metrics_history[locale].append(metrics)

        # Update convergence rate
        self._update_convergence_rate(locale)

        # Update performance trends
        self._update_performance_trends(locale)

    def _update_convergence_rate(self, locale: str):
        """Calculate convergence rate for a locale."""
        history = list(self.metrics_history[locale])
        if len(history) < 3:
            return

        # Calculate convergence rate based on loss improvement
        recent_losses = [m.train_loss for m in history[-5:]]
        if len(recent_losses) < 2:
            return

        # Simple convergence rate: negative of loss improvement rate
        loss_changes = np.diff(recent_losses)
        if len(loss_changes) > 0:
            convergence_rate = -np.mean(loss_changes)
            self.convergence_rates[locale] = convergence_rate

    def _update_performance_trends(self, locale: str):
        """Update performance trend indicators."""
        history = list(self.metrics_history[locale])
        if len(history) < 10:
            return

        # Calculate trend in accuracy (if available)
        accuracies = [m.eval_accuracy for m in history[-10:] if m.eval_accuracy is not None]
        if len(accuracies) >= 3:
            # Simple linear trend
            x = np.arange(len(accuracies))
            trend = np.polyfit(x, accuracies, 1)[0]
            self.performance_trends[locale] = [trend]  # Store as list for potential multiple indicators

    def get_convergence_status(self, locale: str) -> Dict[str, Any]:
        """Get convergence status for a locale."""
        if locale not in self.metrics_history:
            return {"status": "no_data"}

        history = list(self.metrics_history[locale])
        if len(history) < 3:
            return {"status": "insufficient_data"}

        convergence_rate = self.convergence_rates.get(locale, 0.0)
        recent_loss = history[-1].train_loss
        recent_accuracy = history[-1].eval_accuracy

        # Determine convergence status
        if abs(convergence_rate) < 1e-6:
            status = "converged"
        elif convergence_rate < 0:
            status = "improving"
        else:
            status = "diverging"

        return {
            "status": status,
            "convergence_rate": convergence_rate,
            "recent_loss": recent_loss,
            "recent_accuracy": recent_accuracy,
            "trend": self.performance_trends.get(locale, [0.0])[0]
        }

    def get_plateau_locales(self, threshold: float = 1e-4, min_steps: int = 100) -> List[str]:
        """Identify locales that have plateaued in performance."""
        plateau_locales = []

        for locale, history_deque in self.metrics_history.items():
            history = list(history_deque)
            if len(history) < min_steps:
                continue

            # Check if recent loss changes are below threshold
            recent_losses = [m.train_loss for m in history[-20:]]
            if len(recent_losses) < 10:
                continue

            loss_variance = np.var(recent_losses)
            if loss_variance < threshold:
                plateau_locales.append(locale)

        return plateau_locales


class AdaptiveMergeScheduler:
    """Adaptive scheduler for merge operations based on performance metrics."""

    def __init__(
        self,
        base_merge_frequency: int = 3,
        min_merge_frequency: int = 1,
        max_merge_frequency: int = 10,
        convergence_threshold: float = 1e-4,
        performance_window: int = 5
    ):
        self.base_merge_frequency = base_merge_frequency
        self.min_merge_frequency = min_merge_frequency
        self.max_merge_frequency = max_merge_frequency
        self.convergence_threshold = convergence_threshold
        self.performance_window = performance_window

        self.performance_tracker = PerformanceTracker()
        self.merge_history: List[Dict[str, Any]] = []
        self.last_merge_epoch: Dict[str, int] = {}

    def evaluate_merge_necessity(
        self,
        current_epoch: int,
        active_locales: List[str],
        global_metrics: Optional[Dict[str, float]] = None
    ) -> MergeDecision:
        """
        Evaluate whether a merge should be triggered based on current performance.

        Args:
            current_epoch: Current training epoch
            active_locales: List of currently active locales
            global_metrics: Global training metrics

        Returns:
            MergeDecision with recommendation and confidence
        """
        # Check if we have enough data
        if not self._has_sufficient_data(active_locales):
            return MergeDecision(
                should_merge=False,
                confidence=0.0,
                reason="insufficient_data",
                metrics={}
            )

        # Analyze convergence status
        convergence_status = self._analyze_convergence(active_locales)

        # Analyze performance plateaus
        plateau_locales = self.performance_tracker.get_plateau_locales(
            self.convergence_threshold
        )

        # Calculate merge necessity score
        merge_score, reasons = self._calculate_merge_score(
            convergence_status, plateau_locales, active_locales
        )

        # Determine if merge should be triggered
        should_merge = merge_score > 0.5

        # Adjust merge frequency if needed
        recommended_frequency = self._adjust_merge_frequency(
            merge_score, current_epoch, active_locales
        )

        # Suggest algorithm based on performance patterns
        suggested_algorithm = self._suggest_merge_algorithm(convergence_status)

        return MergeDecision(
            should_merge=should_merge,
            confidence=merge_score,
            reason="; ".join(reasons),
            metrics={
                "merge_score": merge_score,
                "convergence_status": convergence_status,
                "plateau_locales": plateau_locales,
                "recommended_frequency": recommended_frequency
            },
            recommended_merge_frequency=recommended_frequency,
            suggested_algorithm=suggested_algorithm
        )

    def _has_sufficient_data(self, active_locales: List[str]) -> bool:
        """Check if we have sufficient performance data for decision making."""
        for locale in active_locales:
            history = self.performance_tracker.metrics_history.get(locale)
            if not history or len(history) < 5:
                return False
        return True

    def _analyze_convergence(self, active_locales: List[str]) -> Dict[str, Any]:
        """Analyze convergence status across all active locales."""
        convergence_status = {}

        for locale in active_locales:
            status = self.performance_tracker.get_convergence_status(locale)
            convergence_status[locale] = status

        return convergence_status

    def _calculate_merge_score(
        self,
        convergence_status: Dict[str, Any],
        plateau_locales: List[str],
        active_locales: List[str]
    ) -> Tuple[float, List[str]]:
        """Calculate a score indicating merge necessity."""
        score = 0.0
        reasons = []

        # Factor 1: Plateau detection (high importance)
        plateau_ratio = len(plateau_locales) / len(active_locales)
        if plateau_ratio > 0.5:
            score += 0.4
            reasons.append(f"high_plateau_ratio ({plateau_ratio:.2f})")
        elif plateau_ratio > 0.25:
            score += 0.2
            reasons.append(f"moderate_plateau_ratio ({plateau_ratio:.2f})")

        # Factor 2: Convergence status
        converged_count = sum(
            1 for status in convergence_status.values()
            if status.get("status") == "converged"
        )
        converged_ratio = converged_count / len(active_locales)

        if converged_ratio > 0.7:
            score += 0.3
            reasons.append(f"high_convergence_ratio ({converged_ratio:.2f})")
        elif converged_ratio > 0.4:
            score += 0.15
            reasons.append(f"moderate_convergence_ratio ({converged_ratio:.2f})")

        # Factor 3: Negative trends
        negative_trend_count = sum(
            1 for status in convergence_status.values()
            if status.get("trend", 0) < -0.01
        )
        if negative_trend_count > 0:
            negative_ratio = negative_trend_count / len(active_locales)
            score += 0.3 * negative_ratio
            reasons.append(f"negative_performance_trends ({negative_ratio:.2f})")

        return min(score, 1.0), reasons

    def _adjust_merge_frequency(
        self,
        merge_score: float,
        current_epoch: int,
        active_locales: List[str]
    ) -> int:
        """Adjust merge frequency based on performance analysis."""
        if merge_score > 0.8:
            # High necessity - increase frequency
            new_frequency = max(self.min_merge_frequency, self.base_merge_frequency - 1)
        elif merge_score > 0.6:
            # Moderate necessity - keep base frequency
            new_frequency = self.base_merge_frequency
        elif merge_score > 0.3:
            # Low necessity - decrease frequency
            new_frequency = min(self.max_merge_frequency, self.base_merge_frequency + 1)
        else:
            # Very low necessity - significantly decrease frequency
            new_frequency = min(self.max_merge_frequency, self.base_merge_frequency + 2)

        return new_frequency

    def _suggest_merge_algorithm(self, convergence_status: Dict[str, Any]) -> str:
        """Suggest the most appropriate merge algorithm based on convergence patterns."""
        # Count different convergence statuses
        statuses = [status.get("status") for status in convergence_status.values()]

        converged_count = statuses.count("converged")
        improving_count = statuses.count("improving")
        diverging_count = statuses.count("diverging")

        total = len(statuses)

        # Suggest algorithms based on convergence patterns
        if diverging_count / total > 0.3:
            # Many models diverging - use conservative approach
            return "average"
        elif converged_count / total > 0.5:
            # Most models converged - can use more sophisticated merging
            return "linear"
        else:
            # Mixed convergence - use similarity-based approach
            return "similarity"

    def record_merge(
        self,
        epoch: int,
        locales: List[str],
        algorithm: str,
        outcome: str,
        performance_change: Optional[Dict[str, float]] = None
    ):
        """Record the outcome of a merge operation."""
        merge_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "epoch": epoch,
            "locales": locales,
            "algorithm": algorithm,
            "outcome": outcome,  # "success", "failed", "partial"
            "performance_change": performance_change or {}
        }

        self.merge_history.append(merge_record)

        # Update last merge epoch for locales
        for locale in locales:
            self.last_merge_epoch[locale] = epoch

        logger.info(f"Recorded merge: {merge_record}")

    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about merge operations."""
        if not self.merge_history:
            return {"total_merges": 0}

        total_merges = len(self.merge_history)
        successful_merges = sum(1 for m in self.merge_history if m["outcome"] == "success")
        failed_merges = sum(1 for m in self.merge_history if m["outcome"] == "failed")

        # Algorithm performance
        algorithm_stats = {}
        for merge in self.merge_history:
            algo = merge["algorithm"]
            if algo not in algorithm_stats:
                algorithm_stats[algo] = {"total": 0, "success": 0}
            algorithm_stats[algo]["total"] += 1
            if merge["outcome"] == "success":
                algorithm_stats[algo]["success"] += 1

        # Calculate success rates
        for algo, stats in algorithm_stats.items():
            stats["success_rate"] = stats["success"] / stats["total"]

        return {
            "total_merges": total_merges,
            "successful_merges": successful_merges,
            "failed_merges": failed_merges,
            "overall_success_rate": successful_merges / total_merges,
            "algorithm_statistics": algorithm_stats,
            "last_merge_epoch": self.last_merge_epoch,
            "base_merge_frequency": self.base_merge_frequency
        }


class EnhancedMonitor:
    """Enhanced monitoring system for iterative training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_log: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log_training_step(
        self,
        locale: str,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        system_metrics: Optional[Dict[str, float]] = None
    ):
        """Log a training step with comprehensive metrics."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "locale": locale,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "system_metrics": system_metrics or {}
        }

        self.metrics_log.append(log_entry)

        # Check for alerts
        self._check_alerts(log_entry)

    def _check_alerts(self, log_entry: Dict[str, Any]):
        """Check for any alert conditions."""
        locale = log_entry["locale"]
        metrics = log_entry["metrics"]

        # Alert for NaN or infinite loss
        train_loss = metrics.get("train_loss")
        if train_loss is not None and (np.isnan(train_loss) or np.isinf(train_loss)):
            self._create_alert(
                "critical",
                f"Invalid loss detected for {locale}",
                {"locale": locale, "loss": train_loss}
            )

        # Alert for very high loss
        if train_loss is not None and train_loss > 10.0:
            self._create_alert(
                "warning",
                f"High training loss for {locale}",
                {"locale": locale, "loss": train_loss}
            )

        # Alert for zero accuracy
        eval_accuracy = metrics.get("eval_accuracy")
        if eval_accuracy is not None and eval_accuracy == 0.0:
            self._create_alert(
                "warning",
                f"Zero evaluation accuracy for {locale}",
                {"locale": locale}
            )

        # Alert for memory usage
        system_metrics = log_entry.get("system_metrics", {})
        memory_usage = system_metrics.get("memory_usage_percent")
        if memory_usage is not None and memory_usage > 90:
            self._create_alert(
                "critical",
                f"High memory usage for {locale}",
                {"locale": locale, "memory_usage": memory_usage}
            )

    def _create_alert(self, severity: str, message: str, details: Dict[str, Any]):
        """Create an alert entry."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "message": message,
            "details": details
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {message} - {details}")

    def get_recent_metrics(
        self,
        locale: Optional[str] = None,
        last_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent metrics entries."""
        recent_metrics = self.metrics_log[-last_n:]

        if locale:
            recent_metrics = [m for m in recent_metrics if m["locale"] == locale]

        return recent_metrics

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of all alerts."""
        if not self.alerts:
            return {"total_alerts": 0}

        severity_counts = {}
        for alert in self.alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "severity_breakdown": severity_counts,
            "recent_alerts": self.alerts[-10:]  # Last 10 alerts
        }

    def save_monitoring_data(self):
        """Save all monitoring data to files."""
        import json
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        # Save metrics log
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

        # Save alerts
        alerts_path = os.path.join(self.output_dir, "training_alerts.json")
        with open(alerts_path, 'w') as f:
            json.dump(self.alerts, f, indent=2)

        logger.info(f"Monitoring data saved to {self.output_dir}")