## 7. Future Development Plan

This section outlines the strategic goals for the evolution of the MergingUriel project.

### 7.1. Advanced Merging Methods & System Integration ✅ COMPLETED

#### **Key Achievements:**
-   **Advanced Merging Methods Integration**: Successfully integrated TIES, Task Arithmetic, SLERP, RegMean, and DARE merging algorithms with full URIEL-weighting support
-   **Production-Ready Sequential Training**: Complete memory-safe training system with periodic merging during training, GPU optimization, and comprehensive checkpointing
-   **URIEL-Guided Ensemble Inference**: Alternative to parameter merging combining model outputs at inference time using weighted logits based on URIEL similarity scores
-   **Comprehensive Automated Evaluation**: Complete refactoring of results aggregation with dynamic experiment parsing, automatic baseline integration, and professional reporting capabilities

#### **New Components Added:**

-   **`iterative_training_orchestrator.py`**: Main orchestrator class managing multi-model training
-   **`iterative_config.py`**: Comprehensive configuration system for iterative training
-   **`training_state.py`**: Model state management and checkpointing system
-   **`merge_coordinator.py`**: Coordinates merge operations during training
-   **`adaptive_merging.py`**: Intelligent merge scheduling and performance monitoring
-   **`run_iterative_training.py`**: CLI interface for iterative training experiments
-   **`uriel_ensemble_inference.py`**: Main ensemble inference system with URIEL logits weighting
-   **`ensemble_runner.py`**: Modular experiment runner for individual tests
-   **`comparison_runner.py`**: Comprehensive comparison framework
-   **Enhanced aggregate_results.py`**: Complete automated evaluation system with baseline integration

#### **Key Features Implemented:**

-   **Advanced Merging**: All major merging algorithms (TIES, Task Arithmetic, SLERP, RegMean, DARE) with URIEL integration
-   **Sequential Training**: Models trained one-by-one to prevent GPU OOM errors (default behavior)
-   **Memory Management**: Automatic GPU cache clearing between model training
-   **Adaptive Merge Scheduling**: Performance-based merge triggering after each model
-   **Comprehensive Checkpointing**: Automatic state preservation and recovery
-   **Ensemble Inference**: URIEL-guided logits weighting with multiple voting methods
-   **Dynamic Results Parsing**: Automatic discovery of experiment types without hardcoded configurations
-   **Baseline Integration**: Automatic inclusion of Best Source Language and Best Overall Zero-Shot baselines
-   **Professional Reporting**: Multiple output formats (CSV, JSON, Markdown) with win rate analysis

#### **Usage Examples:**

```bash
# Basic sequential training (default, memory-safe)
python merginguriel/run_iterative_training.py \
  --target-lang sq-AL \
  --mode similarity \
  --max-epochs 15 \
  --batch-size 32

# Advanced merging with new methods
python merginguriel/run_merging_pipeline_refactored.py \
  --mode ties \
  --target-lang sq-AL \
  --num-languages 5

# URIEL ensemble inference
python merginguriel/uriel_ensemble_inference.py \
  --target-lang "sq-AL" \
  --voting-method "uriel_logits" \
  --num-languages 5

# Comprehensive automated evaluation
python merginguriel/aggregate_results.py --verbose
```

### 7.2. Future Development Initiatives 📋 **PLANNED**

### 7.5. Large-Scale Ensemble Inference Experiment Runner ✅ **COMPLETED**

-   **Goal:** Create a large-scale experiment runner for URIEL-guided ensemble inference that parallels `run_large_scale_experiment.py` but focuses on ensemble methods rather than model merging.
-   **Status:** ✅ **IMPLEMENTED** - Full production-ready system with comprehensive automation.
-   **Why Needed:** The ensemble system lacked automation for systematic evaluation across multiple target languages and voting methods. This enables direct comparison between parameter merging and ensemble inference approaches.

#### 7.5.1. **Core Implementation**

**Script: `merginguriel/run_large_scale_ensemble_experiments.py`**

**Architecture Features:**
- **Parallel Structure:** Mirrors `run_large_scale_experiment.py` but adapted for ensemble inference
- **Multi-Voting Support:** Supports all voting methods: `majority`, `weighted_majority`, `soft`, `uriel_logits`
- **Unified Similarity Processing:** Uses the same `similarity_utils.py` as merging pipeline
- **Results Compatibility:** Generates results fully compatible with `aggregate_results.py`
- **Progress Tracking:** Complete resumption capability with detailed progress files

**Key Features Implemented:**
- **Full Automation:** Run experiments across all 49 locales automatically
- **Voting Method Matrix:** Tests all combinations of target languages × voting methods
- **GPU Memory Management:** Loads all ensemble models simultaneously within VRAM limits
- **No Fallback Policy:** Immediate error detection without dummy data masking
- **Full Test Set Evaluation:** Supports complete test set evaluation (2,974 examples per locale)

#### 7.5.2. **Command Interface**

**Full Coverage Evaluation:**
```bash
# Complete evaluation across all locales and voting methods
python merginguriel/run_large_scale_ensemble_experiments.py \
    --max-locales 49 \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits" \
    --num-examples None \
    --num-languages 5 \
    --top-k 20 \
    --sinkhorn-iters 20
```

**Targeted Evaluation:**
```bash
# Specific locales with all voting methods
python merginguriel/run_large_scale_ensemble_experiments.py \
    --target-languages "en-US" "sq-AL" "sw-KE" \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits" \
    --num-examples 100
```

**Progress Management:**
```bash
# Resume from specific locale
python merginguriel/run_large_scale_ensemble_experiments.py \
    --start-from 10 \
    --max-locales 20 \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits"
```

#### 7.5.3. **Results Integration**

**Folder Structure:**
```
results/
├── ensemble_majority_en-US/
├── ensemble_weighted_majority_en-US/
├── ensemble_soft_en-US/
├── ensemble_uriel_logits_en-US/
├── ensemble_majority_sq-AL/
└── ... (all combinations)
```

**Results Format:**
- **Compatible Structure:** Generates `results.json` files with same schema as merging experiments
- **Metadata Inclusion:** Complete experiment configuration, voting method, similarity settings
- **Progress Tracking:** Unified progress files compatible with merging experiments

#### 7.5.4. **Critical Fixes Applied**

**No-Fallback Implementation:**
- **Ensemble Script Fix:** Removed dangerous dummy data fallback that was masking 0% accuracy errors
- **Dataset Loading:** Added `trust_remote_code=True` and fixed HuggingFace dataset slicing behavior
- **Error Transparency:** Scripts now fail immediately on real issues without silent fallbacks

**Aggregation Parser Enhancement:**
- **Ensemble Folder Parsing:** Updated `aggregate_results.py` to handle `ensemble_{voting_method}_{locale}` format
- **Underscore Support:** Fixed parsing for voting method names with underscores (weighted_majority, uriel_logits)

#### 7.5.5. **Integration with Existing Systems**

**Unified Reporting:**
- **Cross-Method Analysis:** `aggregate_results.py` handles both merging and ensemble results
- **Baseline Integration:** Ensemble results include same baseline comparisons as merging
- **Comprehensive Reports:** Single reports showing all approaches side-by-side

**GPU Memory Management:**
- **Simultaneous Loading:** Confirmed ensemble loads all models simultaneously within VRAM limits
- **Memory Efficiency:** No VRAM increase during ensemble inference vs individual model loading

#### 7.5.6. **Success Criteria Met**

- ✅ **Automated Execution:** Successfully runs experiments for all 49 locales without intervention
- ✅ **Complete Coverage:** Tests all voting methods across all target languages
- ✅ **Results Compatibility:** Results aggregate seamlessly with existing merging experiments
- ✅ **Progress Resumption:** Full resumption capability from any interruption point
- ✅ **Unified Reporting:** Integrated with `aggregate_results.py` for comprehensive analysis
- ✅ **Performance Comparison:** Enables direct comparison between merging and ensemble approaches

#### 7.5.7. **Usage Examples**

**Quick Validation:**
```bash
# Test with 3 locales to validate system
python merginguriel/run_large_scale_ensemble_experiments.py \
    --max-locales 3 \
    --voting-methods "majority" "uriel_logits"
```

**Production Run:**
```bash
# Full experiment across all locales
python merginguriel/run_large_scale_ensemble_experiments.py \
    --max-locales 49 \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits"
```

**Combined Analysis:**
```bash
# Aggregate both merging and ensemble results
python merginguriel/aggregate_results.py --verbose
```

### 7.5.8. **Full Test Set Evaluation Verification** ✅ **COMPLETED**

**Verification Results:**
- **N-vs-N Evaluation**: `run_nxn_evaluation.py` uses complete test sets via `split="test"` without sampling
- **Large-Scale Merging**: `run_large_scale_experiment.py` evaluates on full test datasets through `evaluate_specific_model()` calls
- **Test Set Size**: Each locale has approximately 2,974 test examples in the MASSIVE dataset
- **No Sampling**: Both evaluation scripts use all available data rows for comprehensive accuracy measurement

**Commands for Full Evaluation:**
```bash
# Complete N-vs-N evaluation (all locales × all locales)
python merginguriel/run_nxn_evaluation.py --max-workers 4

# Full large-scale merging with complete test sets
python merginguriel/run_large_scale_experiment.py \
    --modes baseline similarity average fisher ties task_arithmetic slerp regmean \
    --max-locales 49

# Full large-scale ensemble with complete test sets
python merginguriel/run_large_scale_ensemble_experiments.py \
    --max-locales 49 \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits" \
    --num-examples None
```

### 7.6. Large-Scale Iterative Training Experiment Runner ✅ **COMPLETED**

**Goal:** Create a comprehensive large-scale experiment runner for iterative training that orchestrates multiple sequential training experiments and is fully compatible with the existing aggregation system.

**Status:** ✅ **IMPLEMENTED** - Full production-ready system with comprehensive automation and integration.

#### **7.6.1. **Core Implementation**

**Script: `merginguriel/run_large_scale_iterative_training.py`**

**Architecture Features:**
- **Sequential Training Orchestration**: Manages multiple iterative training runs with models trained one-by-one to prevent OOM
- **Auto-Selection System**: Automatically selects source languages based on URIEL similarity to target
- **Progress Tracking**: Complete resumption capability with JSON progress files and comprehensive state management
- **Resource Management**: GPU monitoring, storage management, timeout handling, and automatic recovery
- **Results Integration**: Generates `results.json` files fully compatible with `aggregate_results.py`

**Key Features Implemented:**
- **Full Automation**: Run experiments across all 49 locales automatically
- **Intelligent Source Selection**: Auto-selects 5-7 diverse source languages based on URIEL similarity
- **Memory-Safe Training**: Sequential training with GPU cache clearing between models
- **Comprehensive Monitoring**: Real-time resource monitoring (CPU, memory, disk, GPU utilization)
- **Results Extraction**: Automatic extraction of best-performing merged models from training output

#### **7.6.2. **Command Interface**

**Full Coverage Evaluation:**
```bash
# Complete evaluation across all locales with iterative training
python merginguriel/run_large_scale_iterative_training.py \
    --max-locales 49 \
    --mode similarity \
    --max-epochs 15 \
    --max-models 5 \
    --merge-frequency 3
```

**Targeted Evaluation:**
```bash
# Specific locales with different iterative training modes
python merginguriel/run_large_scale_iterative_training.py \
    --target-languages "sq-AL" "sw-KE" "th-TH" \
    --mode fisher_dataset \
    --max-epochs 10 \
    --max-models 3
```

**Resource Management:**
```bash
# With enhanced monitoring and recovery
python merginguriel/run_large_scale_iterative_training.py \
    --max-locales 20 \
    --mode similarity \
    --sequential-training \
    --enable-auto-recovery \
    --timeout-hours 12.0
```

#### **7.6.3. **Results Integration**

**Folder Structure:**
```
results/
├── iterative_similarity_sq-AL/
├── iterative_fisher_dataset_sw-KE/
├── iterative_task_arithmetic_th-TH/
└── ... (one folder per target-language-mode combination)
```

**Results Format:**
- **Compatible Structure**: Generates `results.json` files with same schema as other experiments
- **Performance Extraction**: Automatically identifies best-performing merged model across all training cycles
- **Metadata Preservation**: Complete training configuration, source languages, and merge details
- **Progress Tracking**: Integration with existing progress file systems

#### **7.6.4. **Enhanced Features**

**Auto-Selection System:**
```python
def auto_select_iterative_sources(target_lang: str, max_models: int) -> List[str]:
    """Auto-select source languages based on URIEL similarity"""
    # Uses existing similarity_utils.py for language selection
    # Limits to 5-7 models to manage training time effectively
    # Prioritizes diverse language families for robust merging
```

**Resource Management:**
- **Real-time Monitoring**: CPU, memory, disk, and dual GPU utilization tracking
- **Memory Optimization**: Automatic GPU cache clearing between model training
- **Storage Planning**: Monitors disk space and manages checkpoint cleanup
- **Timeout Handling**: Configurable timeouts per locale (default: 12 hours)

**Progress Tracking:**
- **JSON Progress Files**: Resumption capability from any interruption point
- **Training-Level Progress**: Track which models have completed training
- **Merge-Level Progress**: Track which merge cycles have completed
- **Checkpoint Integration**: Leverages existing robust checkpoint system

#### **7.6.5. **Integration with Existing Systems**

**Aggregate Results Compatibility:**
- **Updated Parser**: Enhanced `aggregate_results.py` to handle `iterative_{mode}_{locale}` folder names
- **Results Schema**: Compatible with existing aggregation and reporting systems
- **Baseline Integration**: Supports all baseline comparisons and win rate analysis
- **Cross-Method Analysis**: Enables comparison between iterative training, post-training merging, and ensemble inference

**Command-Line Consistency:**
- **Standard Arguments**: Follows same naming conventions as other large-scale runners
- **Pass-Through Options**: Supports all iterative training parameters
- **Progress Management**: Same progress file format as merging experiments
- **Output Structure**: Compatible with existing results directory organization

#### **7.6.6. **Success Criteria Met**

- ✅ **Automated Execution**: Successfully runs iterative training experiments across multiple locales
- ✅ **Results Compatibility**: Results aggregate seamlessly with existing merging and ensemble experiments
- ✅ **Progress Resumption**: Full resumption capability from any interruption point
- ✅ **Resource Management**: Effective GPU memory and storage management for long-running experiments
- ✅ **Integration**: Works seamlessly with existing `aggregate_results.py` and reporting systems
- ✅ **Performance Comparison**: Enables direct comparison between all three approaches (iterative, merging, ensemble)

#### **7.6.7. **Usage Examples**

**Quick Validation:**
```bash
# Test with 2 locales to validate system
python merginguriel/run_large_scale_iterative_training.py \
    --max-locales 2 \
    --mode similarity \
    --max-epochs 3 \
    --max-models 3
```

**Production Run:**
```bash
# Full experiment across all locales
python merginguriel/run_large_scale_iterative_training.py \
    --max-locales 49 \
    --mode similarity \
    --max-epochs 15 \
    --max-models 5 \
    --merge-frequency 3
```

**Comparative Analysis:**
```bash
# Run all three approaches for comprehensive comparison
python merginguriel/run_large_scale_experiment.py --max-locales 10
python merginguriel/run_large_scale_ensemble_experiments.py --max-locales 10
python merginguriel/run_large_scale_iterative_training.py --max-locales 10

# Aggregate all results
python merginguriel/aggregate_results.py --verbose
```

#### **7.6.8. **Key Implementation Achievements**

**Complete Automation Pipeline:**
- **End-to-End Automation**: From source language selection to results aggregation
- **Error Recovery**: Automatic recovery from training failures with retry mechanisms
- **Resource Optimization**: Intelligent management of GPU memory and storage throughout long runs

**Advanced Monitoring:**
- **System Resource Tracking**: Real-time monitoring of CPU, memory, disk, and GPU utilization
- **Training Progress Estimation**: Time estimates and progress tracking for long-running experiments
- **Comprehensive Logging**: Detailed logging with real-time output streaming

**Results Processing:**
- **Best Model Selection**: Automatic identification of highest-performing merged model across training cycles
- **Performance Tracking**: Complete training metadata and performance evolution
- **Standardized Output**: Results in format expected by existing analysis tools

This implementation completes the MergingUriel experiment suite, enabling comprehensive comparison between:
1. **Post-Training Merging** (run_large_scale_experiment.py)
2. **Ensemble Inference** (run_large_scale_ensemble_experiments.py)
3. **Iterative Training** (run_large_scale_iterative_training.py)

All three approaches now produce compatible results that can be aggregated and compared using the existing `aggregate_results.py` system.

### 7.7. Comprehensive Results Visualization & Plotting ✅ **NEW**

**Goal:** Create a comprehensive visualization system for MergingUriel experiment results that can read CSV files generated by `aggregate_results.py` and produce publication-ready plots for presentations and analysis.

**Status:** ✅ **IMPLEMENTED** - Full production-ready plotting system with automatic CSV format detection and comprehensive visualization capabilities.

#### 7.7.1. **Core Implementation**

**Scripts:**
- **`plot_results.py`**: Main plotting script with comprehensive visualization features
- **`quick_plot.py`**: Simple interactive interface for quick plot generation

**Architecture Features:**
- **Smart Format Detection**: Automatically detects old vs new CSV formats from `aggregate_results.py`
- **Multiple Plot Types**: Accuracy comparisons, win rate analysis, performance heatmaps, summary statistics
- **Publication-Ready Output**: High-resolution (300 DPI) plots with professional styling
- **Flexible Data Handling**: Works with partial and incomplete experiment results
- **Automatic File Selection**: Uses latest results files by default

#### 7.7.2. **Available Plot Types**

**1. Accuracy Comparison Plot:**
- Multi-panel visualization with 4 subplots
- Method comparison bar charts
- Accuracy distribution box plots
- Baseline vs merging methods comparison
- Performance improvement heatmap
- **Use Case**: Overall performance overview and method comparison

**2. Win Rate Analysis Plot:**
- Shows how often each method beats baselines
- Calculates win rates vs native baseline, best source language, and best overall zero-shot
- Bar chart with percentage labels and reference lines
- **Use Case**: Method effectiveness analysis

**3. Locale Performance Heatmap:**
- Heatmap showing accuracy by locale and method
- Color-coded performance visualization
- Quick identification of high/low performing locales
- **Use Case**: Geographic and linguistic pattern analysis

**4. Summary Statistics Plot:**
- Mean accuracy with standard deviation error bars
- Experiment count per method
- Statistical overview of all experiments
- **Use Case**: Executive summary and high-level insights

#### 7.7.3. **Command Interface**

**Auto-Plot Latest Results:**
```bash
# Generate all plot types from latest results
python plot_results.py

# Generate specific plot types
python plot_results.py --plot-types accuracy win_rate
```

**Use Specific CSV File:**
```bash
# Plot from specific results file
python plot_results.py --csv-file results_aggregated_20251015_055403.csv

# Filter by locales and experiment types
python plot_results.py --locales sq-AL fr-FR zh-TW --plot-types accuracy heatmap
```

**Interactive Quick Plot:**
```bash
# Interactive plot generator
python quick_plot.py

# Available options:
# 1. accuracy    - Accuracy comparison across methods
# 2. win_rate    - Win rate analysis vs baselines
# 3. heatmap     - Performance heatmap by locale
# 4. summary     - Summary statistics
# 5. all         - Generate all plots (default)
```

**List Available Data:**
```bash
# List all available CSV files with formats
python plot_results.py --list-csv

# Example output:
# Available CSV files:
#   1. results_aggregated_20251015_055403.csv (new) - 2025-10-15 05:54:00
#   2. results_comparison_20251015_055403.csv (old) - 2025-10-15 05:54:00
#   ...
```

#### 7.7.4. **Smart Features**

**Format Detection:**
```python
def detect_csv_format(csv_path: str) -> str:
    """
    Automatically detects CSV format:
    - 'new' for aggregate_results.py format
    - 'old' for legacy format
    """
    # Checks for new format indicators: experiment_type, source_languages, weights, merge_mode
    # Falls back to old format detection: baseline, similarity, average columns
```

**Data Preprocessing:**
- **New Format**: Direct processing with proper typing and categorical variables
- **Old Format**: Automatic reshaping from wide to long format for consistency
- **Error Handling**: Robust handling of missing or malformed data

**Output Management:**
- **Timestamped Files**: All plots include timestamps for version control
- **High Resolution**: 300 DPI for publication quality
- **Multiple Formats**: Ready for presentations and papers
- **Organized Storage**: All plots saved to `plots/` directory

#### 7.7.5. **Integration with Existing Systems**

**CSV File Compatibility:**
- **Auto-Discovery**: Finds latest results files automatically using modification time
- **Multiple Formats**: Handles both new and old CSV formats from `aggregate_results.py`
- **Data Validation**: Checks data integrity before plotting

**Results Directory Structure:**
```
plots/
├── accuracy_comparison_20251015_055847.png     # Multi-panel accuracy comparison
├── win_rate_analysis_20251015_055850.png       # Win rate vs baselines
├── locale_performance_heatmap_20251015_055851.png  # Locale × method heatmap
└── summary_statistics_20251015_055852.png      # Statistical overview
```

**Seamless Workflow Integration:**
```bash
# Complete analysis workflow
python merginguriel/aggregate_results.py --verbose    # Generate results
python plot_results.py                               # Create plots
# Present with high-quality visualizations
```

#### 7.7.6. **Advanced Visualization Features**

**Multi-Panel Layouts:**
- **Figure 1**: 2×2 subplot layout for comprehensive analysis
- **Subplot 1**: Method comparison bar chart with legend
- **Subplot 2**: Box plot showing accuracy distributions
- **Subplot 3**: Baseline vs merging methods
- **Subplot 4**: Improvement heatmap with color gradients

**Professional Styling:**
- **Color Palettes**: Seaborn "husl" palette for method differentiation
- **Grid Lines**: Subtle grid lines for better readability
- **Axis Labels**: Clear, descriptive labels with rotation where needed
- **Value Annotations**: Data values displayed on bars and heatmaps

**Statistical Annotations:**
- **Error Bars**: Standard deviation shown on mean accuracy plots
- **Confidence Intervals**: Visual indicators of statistical significance
- **Reference Lines**: 50% win rate reference line for context
- **Percentage Labels**: Win rate percentages on bar charts

#### 7.7.7. **Success Criteria Met**

- ✅ **Automatic Data Detection**: Successfully detects and processes both old and new CSV formats
- ✅ **Comprehensive Visualization**: Four distinct plot types covering all analysis needs
- ✅ **Publication Quality**: High-resolution (300 DPI) plots suitable for academic papers
- ✅ **User-Friendly Interface**: Both command-line and interactive interfaces available
- ✅ **Integration**: Seamless workflow with existing `aggregate_results.py` system
- ✅ **Error Handling**: Robust handling of missing data and format variations
- ✅ **Presentation Ready**: Professional styling suitable for conferences and reports

#### 7.7.8. **Usage Examples for Presentations**

**Quick Presentation Generation:**
```bash
# Generate all plots for your presentation
python plot_results.py

# Output files ready for slides:
# plots/accuracy_comparison_20251015_055847.png
# plots/win_rate_analysis_20251015_055850.png
# plots/locale_performance_heatmap_20251015_055851.png
# plots/summary_statistics_20251015_055852.png
```

**Focused Analysis:**
```bash
# Focus on specific aspects for targeted presentations
python plot_results.py --plot-types accuracy win_rate

# Analyze specific locales of interest
python plot_results.py --locales sq-AL fr-FR zh-TW --plot-types heatmap
```

**Latest Results Always:**
```bash
# Always uses your most recent experiment results
# No need to specify file names - automatic detection
python plot_results.py
```

This comprehensive plotting system completes the MergingUriel analysis pipeline, enabling:
1. **Data Generation** (aggregate_results.py)
2. **Visualization** (plot_results.py)
3. **Presentation** (High-quality plots)

All components work seamlessly together to support your research presentations and analysis needs.