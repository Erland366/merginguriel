# Iterative Training & Merging for MergingUriel

This document describes the new iterative training and merging functionality added to the MergingUriel project. This feature enables merging models *during* the training process itself, rather than only after training is complete.

## Overview

The iterative training system allows multiple language models to be trained simultaneously while periodically merging their checkpoints. This approach can lead to better knowledge transfer between models and potentially improved final performance.

### Key Benefits

1. **Enhanced Model Integration**: Models share knowledge throughout training, not just at the end
2. **Improved Convergence**: Regular merging can help escape local minima
3. **Performance-Based Merging**: Intelligent scheduling based on convergence patterns
4. **Robust State Management**: Comprehensive checkpointing and recovery mechanisms
5. **Advanced Monitoring**: Detailed performance tracking and alerting

## Architecture

The iterative training system consists of several key components:

### Core Components

- **`IterativeTrainingOrchestrator`**: Main coordinator that manages multiple trainers
- **`TrainingStateManager`**: Handles model states, checkpoints, and persistence
- **`MergeCoordinator`**: Manages merge operations during training
- **`AdaptiveMergeScheduler`**: Intelligent merge timing based on performance metrics
- **`EnhancedMonitor`**: Comprehensive monitoring and alerting system

### Configuration System

- **`IterativeOrchestratorConfig`**: Main configuration for the entire system
- **`IterativeTrainingConfig`**: Per-model training configuration
- **`IterativeMergeConfig`**: Merge operation configuration

## Quick Start

### Basic Usage

```bash
# Train 3 models with merging every 3 epochs
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR,de-DE \
  --max-epochs 15 \
  --merge-frequency 3 \
  --merge-algorithm linear \
  --weight-calculation similarity \
  --output-dir iterative_results
```

### Advanced Usage with Adaptive Features

```bash
# Enable adaptive merging and performance-based triggering
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR,de-DE,es-ES,it-IT \
  --max-epochs 20 \
  --merge-frequency 3 \
  --merge-algorithm linear \
  --adaptive-merge-frequency \
  --performance-merge-trigger \
  --convergence-threshold 1e-4 \
  --checkpoint-before-merge \
  --enable-wandb \
  --wandb-project "Iterative-Training-Experiment" \
  --save-config
```

### Configuration File Usage

```bash
# Save configuration and use it later
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR \
  --save-config \
  --output-dir experiment_1

# Later, use the saved configuration
python merginguriel/run_iterative_training.py \
  --config-file experiment_1/experiment_config.json
```

## Configuration Options

### Training Parameters

- `--locales`: Comma-separated list of locale codes to train
- `--max-epochs`: Maximum number of training epochs
- `--learning-rate`: Learning rate for training
- `--batch-size`: Training batch size
- `--fp16`: Enable mixed precision training

### Merge Parameters

- `--merge-frequency`: Number of epochs between merges (always per epoch)
- `--merge-algorithm`: Merging algorithm (linear/fisher_simple/fisher_dataset)
- `--weight-calculation`: Weight calculation strategy (similarity/average/manual)

**Note**: Merging always occurs per epoch for consistent training dynamics and stable convergence patterns.

### Advanced Features

- `--adaptive-merge-frequency`: Enable adaptive merge frequency adjustment
- `--performance-merge-trigger`: Enable performance-based merge triggering
- `--convergence-threshold`: Threshold for convergence detection
- `--checkpoint-before-merge`: Create checkpoints before each merge

### Important Note on Merge Timing
**Merging always occurs per epoch** to ensure consistent training dynamics. While performance-based triggers can still activate merges between scheduled epochs, the system fundamentally operates on epoch boundaries to maintain stable convergence patterns and predictable training behavior.

### Monitoring and Recovery

- `--enable-wandb`: Enable Weights & Biases logging
- `--enable-auto-recovery`: Enable automatic recovery from failures
- `--validate-merge-integrity`: Validate merge integrity
- `--max-merge-attempts`: Maximum attempts for each merge

## Examples

### Example 1: Basic Two-Model Training

```bash
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR \
  --max-epochs 10 \
  --merge-frequency 2 \
  --merge-algorithm linear \
  --output-dir basic_experiment
```

### Example 2: Fisher-Based Merging with Custom Parameters

```bash
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR,de-DE \
  --max-epochs 15 \
  --merge-frequency 3 \
  --merge-algorithm fisher_dataset \
  --fisher-data-mode target \
  --num-fisher-examples 200 \
  --fisher-batch-size 32
```

### Example 3: High-Performance Experimental Setup

```bash
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR,de-DE,es-ES,it-IT,pt-PT \
  --max-epochs 25 \
  --merge-frequency 2 \
  --merge-algorithm linear \
  --adaptive-merge-frequency \
  --performance-merge-trigger \
  --convergence-threshold 5e-5 \
  --fp16 \
  --gradient-accumulation-steps 2 \
  --batch-size 64 \
  --enable-wandb \
  --wandb-project "Large-Scale-Iterative" \
  --checkpoint-before-merge \
  --retain-merge-checkpoints 5 \
  --enable-auto-recovery \
  --validate-merge-integrity
```

## Output Structure

The iterative training system creates a structured output directory:

```
iterative_results/
├── experiment_config.json          # Configuration file
├── iterative_training.log          # Main log file
├── state_summary.json              # Training state summary
├── merge_history.json              # Complete merge history
├── final_statistics.json           # Final experiment statistics
├── en-US/                         # Per-locale directories
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs
│   └── ...                       # Training outputs
├── fr-FR/
│   ├── checkpoints/
│   └── ...
├── merged_models/                  # Merged model outputs
│   ├── iterative_merge_en-US_fr-FR_20231214_143022/
│   ├── iterative_merge_..._20231214_150015/
│   └── ...
└── training_metrics.json          # Detailed training metrics
```

## Monitoring and Analysis

### Real-time Monitoring

The system provides comprehensive monitoring through:

1. **Console Logging**: Real-time progress updates
2. **Log Files**: Detailed logs saved to files
3. **Wandb Integration**: Web-based monitoring (if enabled)
4. **Performance Metrics**: Convergence tracking and analysis

### Key Metrics Tracked

- Training and validation loss per locale
- Evaluation accuracy and F1 scores
- Convergence rates and trends
- Merge effectiveness and timing
- Resource utilization (memory, compute)
- System health and alerts

### Analyzing Results

After training completes, you can analyze:

1. **Final Statistics**: `final_statistics.json` contains comprehensive results
2. **Merge History**: `merge_history.json` shows all merge operations
3. **Training Metrics**: `training_metrics.json` contains detailed per-step metrics
4. **Model Performance**: Compare final models against baselines

## Advanced Usage

### Custom Weight Calculation

For custom weight calculation strategies:

```python
from merginguriel.run_merging_pipeline_refactored import IterativeWeightCalculator

# Create custom weight calculator
custom_weights = {
    "en-US": 0.4,
    "fr-FR": 0.3,
    "de-DE": 0.3
}

weight_calculator = IterativeWeightCalculator(
    active_model_states=your_model_states,
    target_locales=["en-US", "fr-FR"]
)
```

### Custom Merge Scheduling

```python
from merginguriel.adaptive_merging import AdaptiveMergeScheduler

# Create custom scheduler
scheduler = AdaptiveMergeScheduler(
    base_merge_frequency=3,
    convergence_threshold=1e-4,
    min_merge_frequency=1,
    max_merge_frequency=10
)

# Evaluate merge necessity
decision = scheduler.evaluate_merge_necessity(
    current_epoch=5,
    active_locales=["en-US", "fr-FR"]
)

if decision.should_merge:
    print(f"Merge recommended: {decision.reason}")
```

### Integration with Existing Pipeline

The iterative system integrates seamlessly with existing MergingUriel functionality:

```bash
# Traditional post-training merging (still supported)
python merginguriel/run_merging_pipeline_refactored.py \
  --mode similarity \
  --target-lang sq-AL \
  --num-languages 5

# New iterative training
python merginguriel/run_iterative_training.py \
  --locales sq-AL,en-US,fr-FR \
  --merge-frequency 3 \
  --target-languages sq-AL
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 32`
   - Enable gradient accumulation: `--gradient-accumulation-steps 4`
   - Use mixed precision: `--fp16`

2. **Merge Failures**:
   - Enable auto-recovery: `--enable-auto-recovery`
   - Increase merge timeout: `--merge-timeout 1200`
   - Check merge integrity: `--validate-merge-integrity`

3. **Slow Convergence**:
   - Enable adaptive merging: `--adaptive-merge-frequency`
   - Adjust learning rate: `--learning-rate 1e-4`
   - Check convergence threshold: `--convergence-threshold 1e-5`

### Debug Mode

For debugging, use higher verbosity:

```bash
python merginguriel/run_iterative_training.py \
  --locales en-US,fr-FR \
  --log-level DEBUG \
  --max-epochs 3 \
  --merge-frequency 1
```

## Testing

### Running Tests

```bash
# Run all tests
python tests/test_iterative_training.py

# Run with pytest (if available)
pytest tests/test_iterative_training.py -v
```

### Example Demonstration

```bash
# Run the example demonstration
python examples/iterative_training_example.py
```

## Performance Considerations

### Resource Requirements

- **GPU Memory**: Recommend at least 8GB per model
- **Storage**: Additional space for checkpoints (~2-3x model size)
- **Compute**: Slightly longer training time due to merge operations

### Optimization Tips

1. **Batch Size**: Use the largest batch size that fits in memory
2. **Merge Frequency**: Balance between frequency and overhead
3. **Checkpoint Management**: Limit retained checkpoints to save space
4. **Mixed Precision**: Use `--fp16` for memory efficiency
5. **Gradient Accumulation**: Use for effective larger batch sizes

## Future Development

### Planned Enhancements

1. **Distributed Training**: Multi-GPU and multi-node support
2. **Advanced Algorithms**: Support for more merging algorithms (TIES, DARE, etc.)
3. **Ensemble Methods**: Alternative to parameter merging
4. **Hyperparameter Optimization**: Automatic tuning of merge parameters
5. **Visualization**: Training and merge visualization tools

### Extending the System

The system is designed to be extensible:

- Add new merging algorithms in `MergingStrategyFactory`
- Implement custom weight calculators
- Create new merge scheduling strategies
- Add additional monitoring metrics

## Contributing

When contributing to the iterative training system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure backward compatibility with existing functionality

## References

- [MergingUriel Project Documentation](./CLAUDE.md)
- [Original Merging Pipeline](./merginguriel/run_merging_pipeline_refactored.py)
- [Training Script](./merginguriel/training_bert.py)
- [Configuration System](./merginguriel/iterative_config.py)

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test cases for usage examples
3. Examine the log files for detailed error information
4. Create an issue in the project repository