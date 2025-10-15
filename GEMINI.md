# MergingUriel: The Living Modular Blueprint

## 0. Document Control

**Document Status: LIVING DOCUMENT - Last Updated: 2025-10-15 06:00 UTC**

**The Golden Rule:** This document, `CLAUDE.md`, is the single source of truth for this project. Any developer, human or AI, who modifies the codebase, adds a feature, or changes a workflow **must** update the relevant sections of this document in the same commit/change. This ensures the documentation remains synchronized with the code.

**A Note to Future Developers (Human or AI):** Before committing any changes, you are required to read this document to understand the project's standards and architecture. After implementing your changes, you **must** update the relevant sections of this `CLAUDE.md` file and modify the `Last Updated` date and time above. This is a mandatory step for all contributions.

## 0.1. Development Principles

### **🚫 NO FALLBACKS - FAIL FAST PRINCIPLE**

**Critical Rule:** NEVER implement fallback mechanisms (dummy data, default values, graceful degradation) in production code during development.

**Why:**
- **Immediate Error Detection:** Fallbacks mask real issues and delay debugging
- **Clear Failure Signals:** Scripts should fail immediately when something is wrong
- **Development Transparency:** No silent failures or misleading "success" messages
- **Research Integrity:** Results must reflect real data, not synthetic substitutes

**Examples of FORBIDDEN Fallbacks:**
```python
# ❌ NEVER DO THIS
try:
    dataset = load_dataset("AmazonScience/massive", locale, split="test")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Fallback to dummy data - FORBIDDEN!
    dataset = create_dummy_data()  # 🚫 FORBIDDEN
```

**Required Approach:**
```python
# ✅ ALWAYS DO THIS
dataset = load_dataset("AmazonScience/massive", locale, split="test", trust_remote_code=True)
# Let the script fail if data loading fails - this reveals the real issue immediately
```

**Enforcement:**
- All PRs must be checked for fallback patterns
- Any fallback mechanism must be justified and documented
- Development phase requires immediate failure on any error
- Fallbacks only allowed in production with explicit monitoring

## 1. Introduction & Goal

**MergingUriel** is a research project focused on exploring the efficacy of merging pre-trained language models to create a new, effective model for a target language without requiring training from scratch.

The core hypothesis is that by intelligently combining models from languages that are typologically similar to a target language, we can create a high-performing model that leverages the knowledge embedded in existing artifacts. This approach aims to significantly reduce the computational cost and time required to support new languages.

-   **Base Model:** `xlm-roberta-base`
-   **Primary Dataset:** `AmazonScience/massive` for intent classification tasks.
-   **Guiding Principle:** [URIEL](http://www.cs.cmu.edu/~dmortens/uriel.html) typological vectors serve as the "knowledge base" to determine language similarity and guide the merging process.

## 2. System Architecture

The project follows a modular, multi-stage architecture. The end-to-end workflow is orchestrated to allow for experimentation with different weighting schemes and merging algorithms.

### Data Flow

1.  **Source Models:** The process begins with a collection of individual models, each fine-tuned on the `massive` dataset for a specific language (e.g., `en-US`, `fr-FR`). These models are trained using the `merginguriel/training_bert.py` script.
2.  **Similarity Matrix Processing:** The system uses a pre-computed cosine similarity matrix (`language_similarity_matrix_unified.csv`) containing URIEL typological similarity scores between all language pairs. This matrix is processed on-demand using the `similarity_utils.py` module:
    - **Top-K Filtering**: For a given target language, only the top-K most similar languages are retained (configurable via `--top-k` parameter)
    - **Sinkhorn Normalization**: The filtered similarity scores are normalized using the Sinkhorn algorithm to create a doubly stochastic distribution (rows sum to 1)
    - **Dynamic Processing**: This allows experimentation with different top-K values without regenerating the entire similarity matrix
3.  **Model Selection & Weighting:** The pipeline selects source models based on availability in `haryoaw_k_models.csv` and processes similarity weights as follows:
    - **Direct Locale Matching**: Uses locale codes (en-US, fr-FR, etc.) directly without complex mapping
    - **Weight Assignment**: Each selected model receives a weight proportional to its normalized URIEL similarity to the target language
    - **Model Availability**: Only models that exist in the available models list are considered, regardless of similarity ranking
4.  **Model Merging:** The main pipeline script, `merginguriel/run_merging_pipeline_refactored.py`, combines the parameters of the selected source models using weights from the processed similarity matrix and a merging algorithm from the `auto-merge-llm` library.
5.  **Ensemble Inference:** As an alternative to parameter merging, the `merginguriel/uriel_ensemble_inference.py` system combines model outputs (logits) at inference time using URIEL similarity scores as weights, without creating a new merged model.
6.  **Evaluation:** Both merged models and ensembled predictions are evaluated on benchmark tasks to measure their performance, using scripts like `merginguriel/evaluate_base_encoder.py` and comprehensive comparison frameworks.

## 3. Environment Setup

Before running any training or merging workflows, ensure the project environment is correctly set up.

### 3.1. Activate Virtual Environment

All dependencies are managed within a Python virtual environment. Activate it using:

```bash
source .venv/bin/activate
```

### 3.1.1. Command-Line Argument Conventions

All scripts in the MergingUriel project follow standardized command-line argument naming conventions to ensure consistency and usability. See [`CMD_ARGUMENT_CONVENTIONS.md`](CMD_ARGUMENT_CONVENTIONS.md) for complete documentation.

**Key Naming Rules:**
- **Singular form** (`--target-lang`, `--voting-method`): Used for single-target operations
- **Plural form** (`--target-languages`, `--voting-methods`): Used for batch/comparison operations
- **Kebab-case**: All arguments use lowercase with hyphens
- **Consistency**: Core arguments (`--num-languages`, `--top-k`, `--sinkhorn-iters`) are identical across all scripts

### 3.3. Testing & Quality Assurance

The project includes a comprehensive test suite organized by test type to ensure code quality and system reliability.

#### **Test Directory Structure**
```
tests/
├── __init__.py                    # Test package metadata
├── unit/                          # Unit tests for individual components
│   ├── __init__.py
│   └── test_similarity_utils.py  # Similarity processing unit tests
├── integration/                   # Integration tests for complete workflows
│   ├── __init__.py
│   ├── test_ensemble_inference.py  # Full ensemble pipeline tests
│   └── test_merging_pipeline.py    # Merging pipeline integration tests
└── verification/                  # Verification scripts for system validation
    ├── __init__.py
    └── test_weight_normalization.py  # Weight normalization verification
```

#### **Running Tests**

The project provides a convenient test runner script:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type verification

# Run specific test file
python run_tests.py --file tests/unit/test_similarity_utils.py

# List available test files
python run_tests.py --list
```

#### **Test Categories**
- **Unit Tests**: Test individual components and functions in isolation
- **Integration Tests**: Test complete workflows and system interactions
- **Verification Scripts**: Validate system properties and performance requirements

### 3.4. Hardware & Monitoring

-   **GPU:** The primary development environment is equipped with two NVIDIA RTX 4090 GPUs. However, most individual training and merging tasks are designed to run on a single GPU.
-   **VRAM Management:** It is crucial to monitor GPU memory usage to prevent Out-Of-Memory (OOM) errors, especially when working with large models or batch sizes. Use the following command to check GPU status:

    ```bash
    nvidia-smi
    ```

### 3.3. Testing Framework

The project includes a comprehensive testing framework located in the `tests/` directory. Tests are organized by functionality and should be run before making any significant changes.

#### Available Tests:

-   **`tests/test_new_methods_simple.py`** - Validates that all new merging strategies can be created
-   **`tests/demo_new_merging_methods.py`** - User-friendly demo showing usage examples for all advanced merging methods
-   **`tests/test_all_merging_methods.py`** - Comprehensive test suite for all merging methods
-   **`tests/test_breadcrumbs_merging.py`** - Test framework for future breadcrumbs implementation

#### Running Tests:

```bash
# Quick validation of new methods
python tests/test_new_methods_simple.py

# Full demo with usage examples
python tests/demo_new_merging_methods.py

# Comprehensive testing (may take longer)
python tests/test_all_merging_methods.py
```

**Note:** All tests are designed as dry-run validation that doesn't require actual model files, making them safe to run in any environment.

## 4. Component Breakdown

This section provides a deeper look into the main components of the MergingUriel project.

### `merginguriel/` - The Core Logic

This directory is the heart of the project, containing the Python scripts that define and orchestrate the merging pipeline.

-   **`run_merging_pipeline_refactored.py`**: This is the main entry point for executing a merge. It uses a class-based, strategy pattern design to provide a flexible and extensible workflow. The `--mode` command-line argument is critical as it determines the entire behavior of the pipeline.
    -   **`MergingPipeline`**: The main orchestrator class that coordinates the steps: weight calculation, model merging, and saving the output.
    -   **`WeightCalculatorFactory`**: A factory that, based on the `--mode` argument, instantiates the correct weighting strategy. For example:
        -   `--mode similarity`: Uses `SimilarityWeightCalculator` to load weights from the pre-computed URIEL similarity matrix.
        -   `--mode average`: Uses `AverageWeightCalculator` to assign an equal weight to all source models, establishing a crucial baseline.
    -   **`MergingStrategyFactory`**: A factory that selects the underlying parameter-merging algorithm (e.g., `linear`, `fisher`).
    -   **`ModelMerger`**: The class that takes the models, weights, and merging strategy and invokes the `auto-merge-llm` library to perform the merge.
    -   **`OutputManager`**: Saves the final merged model and a `merge_details.txt` file that records the configuration used for the run.

-   **`training_bert.py`**: A standard Hugging Face `Trainer` script used to fine-tune the `xlm-roberta-base` model on individual language subsets of the `AmazonScience/massive` dataset. The models produced by this script are the raw inputs for the merging process.

-   **`similarity_utils.py`**: **NEW** Reusable module for processing URIEL similarity matrices. This is the core similarity processing system used by both merging pipeline and ensemble inference:
    - **`load_and_process_similarity()`**: Complete pipeline for loading and processing similarity matrices
    - **`process_similarity_matrix()`**: Applies top-K filtering and Sinkhorn normalization
    - **`get_similarity_weights()`**: Extracts normalized weights for target language
    - **Dynamic Top-K**: Allows runtime adjustment of how many similar languages to consider
    - **No Mapping Required**: Works directly with locale codes without complex transformations

-   **`evaluate_base_encoder.py`**: A script for evaluating a model's semantic representation capabilities on the STS-B benchmark. It calculates the Spearman correlation between the cosine similarity of sentence embeddings and the human-annotated similarity scores.

-   **`uriel_ensemble_inference.py`**: Main URIEL-guided ensemble inference system that combines model outputs at inference time using URIEL similarity scores as weights. Implements multiple voting methods including the core `uriel_logits` approach.
    -   **`uriel_weighted_logits()`**: Core algorithm implementing the section 7.3 approach - multiplies each model's logits by its URIEL similarity score and combines them.
    -   **Error Handling**: No fallback system - catches all errors to ensure complete transparency in model loading failures.
    -   **Comprehensive Evaluation**: Supports multiple voting methods for baseline comparison.

-   **`ensemble_runner.py`**: Modular experiment runner for single ensemble tests. Provides reusable functions for running individual experiments and collecting detailed results.

-   **`comparison_runner.py`**: Comprehensive comparison framework that runs experiments across multiple target languages and voting methods, providing statistical analysis and automated reporting.

-   **Files Used**:
    - **`language_similarity_matrix_unified.csv`**: Pre-computed URIEL cosine similarity matrix (50×50) containing similarity scores between all language pairs
    - **`haryos_model/`**: Consolidated model directory containing `xlm-roberta-base_massive_k_{locale}` models
    - **`similarity_utils.py`**: Shared processing module (replaces deprecated similarity_matrix.py)

### Model Directory Structure

The project now uses a consolidated model directory structure for better organization:

```
haryos_model/
├── xlm-roberta-base_massive_k_en-US/
├── xlm-roberta-base_massive_k_fr-FR/
└── ... (50+ locales total)
```

Each model directory contains a complete fine-tuned model with all necessary configuration files for direct loading and merging.

### `auto-merge-llm` Integration

The project leverages the `auto-merge-llm` library (located in `submodules/`) as the powerful backend for performing the parameter-level model merges. This library provides a framework and a collection of established merging algorithms.

-   **High-Level Abstraction:** `MergingUriel` uses `auto-merge-llm` as a library, abstracting away the low-level tensor manipulations. The `run_merging_pipeline_refactored.py` script acts as a high-level orchestrator that prepares models and parameters before feeding them to the selected `auto-merge-llm` method.
-   **Weighting Strategy:** The core innovation of this project is the application of a URIEL-based weighting scheme *on top of* the `auto-merge-llm` methods. The pipeline calculates typologically-informed weights and passes them as parameters (e.g., the `weights` parameter for `linear` merge, or `fisher_scaling_coefficients` for `fisher` merge) to the `auto-merge-llm` functions.

### Similarity Matrix Processing & Top-K Mechanism

The core innovation of MergingUriel is the dynamic processing of URIEL similarity matrices with configurable top-K filtering:

#### **Processing Pipeline**
1. **Load Pre-computed Matrix**: Uses `language_similarity_matrix_unified.csv` containing cosine similarity scores between 50 languages
2. **Top-K Filtering**: For target language T, retains only the K most similar languages (configurable via `--top-k`)
3. **Sinkhorn Normalization**: Applies Sinkhorn algorithm to make the filtered weights doubly stochastic (rows sum to 1.0)
4. **Model Matching**: Direct locale matching with available models in `haryoaw_k_models.csv`

#### **Key Features**
- **Dynamic Top-K**: Experiment with different numbers of source languages without regenerating matrices
- **Flexible Filtering**: Try different top-K values (5, 10, 20) to find optimal ensemble size
- **Automatic Normalization**: Ensures weights always sum to 1.0 regardless of top-K setting
- **Model Availability**: Only considers models that actually exist in the model repository

#### **Example Processing Flow**
```python
# Target: fr-FR, Top-K: 5
# 1. Get top 5 most similar languages: [ar-SA, af-ZA, cy-GB, lv-LV, nl-NL]
# 2. Apply Sinkhorn normalization: [0.094, 0.084, 0.062, 0.052, 0.049]
# 3. Match with available models: [ar-SA, en-US, de-DE, sq-AL, it-IT]
# 4. Final weights: [0.094, 0.046, 0.046, 0.045, 0.043]
```

#### **Configuration Parameters**
- `--top-k 20`: Number of similar languages to consider (default: 20)
- `--sinkhorn-iters 20`: Sinkhorn normalization iterations (default: 20)
- `--num-languages 5`: Maximum number of models to actually use (default: 5)

#### **Enhanced Similarity Processing**
The system supports both sparse and dense similarity computation:

- **Sparse Mode (Default)**: Uses pre-computed similarity matrix with top-K filtering
- **Dense Mode**: Computes similarities on-the-fly using URIEL features with configurable top-K and Sinkhorn normalization
- **Dynamic Top-K**: Experiment with different numbers of source languages without regenerating matrices
- **Flexible Filtering**: Try different top-K values (5, 10, 20) to find optimal ensemble size
- **Automatic Normalization**: Ensures weights always sum to 1.0 regardless of top-K setting

## 5. Supported Merging Methods & Baselines

To scientifically validate the effectiveness of URIEL-guided merging, every experiment is compared against a strong baseline. The `--mode` flag in `run_merging_pipeline_refactored.py` controls which approach is used.

-   **URIEL-Weighted (`--mode similarity`):** Merges the top-N models using weights derived from the URIEL similarity matrix.
-   **Average Baseline (`--mode average`):** Merges the *same* top-N models but assigns each an equal weight (1/N). This isolates the impact of the URIEL weighting scheme itself.

Below is the status of the merging algorithms from `auto-merge-llm` that are currently integrated into this pipeline.

| Merging Method (from auto-merge-llm) | Description | Status in MergingUriel | URIEL Weighting Support | Baseline Comparison |
| :--- | :--- | :--- | :--- | :--- |
| `linear` | A simple weighted average of model parameters. | **Implemented** | Yes | Yes, via `--mode average` |
| `fisher` | Estimates parameter importance using gradients from a sample dataset (primary Fisher method). | **Implemented** | Yes | Yes, via `--preweight equal` |
| `fisher_simple` | Weights parameters by their magnitude as a proxy for importance (class retained, option removed). | **Implemented** | Yes | Yes, via `--mode average` |
| `fisher_dataset` | Fisher dataset-based method (now mapped to `--mode fisher`). | **Implemented** | Yes | Yes, via `--preweight equal` |
| `ties` | Merges models by resolving sign disagreements and pruning low-magnitude weights. | **Implemented** | Yes | Yes, via `--mode ties` |
| `slerp` | Spherical Linear Interpolation, useful for interpolating between models. | **Implemented** | Yes | Yes, via `--mode serp` |
| `task_arithmetic` | Adds or subtracts task vectors representing fine-tuning changes. | **Implemented** | Yes | Yes, via `--mode task_arithmetic` |
| `dare` | A state-of-the-art method that prunes and rescales task vectors before merging. | **Implemented*** | Yes | Yes, via `--mode task_arithmetic --param_value_mask_rate` |
| `regmean` | A method that uses regression to find optimal merging coefficients. | **Implemented** | Yes | Yes, via `--mode regmean` |
| `breadcrumbs` | A method for merging models by analyzing their training trajectories. | Not Implemented | No | No |

*DARE functionality is available through `task_arithmetic` with pruning parameters (`param_value_mask_rate`).

### 5.1. Additional Evaluation Baselines

Beyond the "Average Merge" baseline, the performance of merged models is contextualized by comparing them against strong single-model baselines. These are derived from a comprehensive N-vs-N evaluation performed by the `merginguriel/run_nxn_evaluation.py` script.

-   **Best Source Language Baseline:** For a given merge of N models, this is the accuracy of the single best-performing model *from that specific set of N source models* when evaluated on the target language's test set. This tells us if the merge is better than simply picking the best available component.
-   **Best Overall Zero-Shot Baseline:** This is the highest accuracy achieved by *any* single language model (excluding the target language itself) on the target language's test set. This represents the state-of-the-art for zero-shot transfer without merging.

*Note: The automatic inclusion of these baselines in the final report is a key objective of the refactoring effort detailed in Section 7.4.*

## 6. Key Workflows

### Workflow 1: Training a Base Model

1.  **Objective:** Fine-tune `xlm-roberta-base` on a single language from the `massive` dataset.
2.  **Script:** `merginguriel/training_bert.py`
3.  **Command Example:**
    ```bash
    python merginguriel/training_bert.py --dataset_config_name sw-KE
    ```

### Workflow 2: Running a Single Merge Experiment

1.  **Objective:** Merge several pre-trained models into a new model for a target language.
2.  **Prerequisites:** Fine-tuned source models, `language_similarity_matrix_unified.csv`, and `haryoaw_k_models.csv`.
3.  **Script:** `merginguriel/run_merging_pipeline_refactored.py`
4.  **Command Example (URIEL-Weighted with Top-K):**
    ```bash
    python merginguriel/run_merging_pipeline_refactored.py \
        --mode similarity \
        --target-lang sq-AL \
        --num-languages 5 \
        --top-k 20 \
        --sinkhorn-iters 20
    ```
5.  **Command Example (Average Baseline):**
    ```bash
    python merginguriel/run_merging_pipeline_refactored.py \
        --mode average \
        --target-lang sq-AL \
        --num-languages 5
    ```

**Key Parameters:**
- `--top-k`: Number of similar languages to consider from similarity matrix
- `--num-languages`: Maximum models to actually use (limited by availability)
- `--sinkhorn-iters`: Number of Sinkhorn normalization iterations

### Workflow 3: Running Large-Scale Experiments

This workflow automates the process of merging and evaluating models for multiple target languages and aggregates the results.

1.  **Step 1: Generate N-vs-N Baselines (Prerequisite)**
    -   **Objective:** Evaluate every available source model on every other target language's test set to establish the crucial "Best Source" and "Best Overall Zero-Shot" baselines.
    -   **Script:** `merginguriel/run_nxn_evaluation.py`
    -   **Command:** `python merginguriel/run_nxn_evaluation.py --max-workers 4`
    -   **Output:** Creates a timestamped results directory in `nxn_results/` containing an `evaluation_matrix.csv`. This matrix is a critical input for the final analysis.

2.  **Step 2: Run Large-Scale Merging**
    -   **Objective:** Automatically run the merging and evaluation pipeline for a list of specified target languages and merging modes.
    -   **Script:** `merginguriel/run_large_scale_experiment.py`
    -   **Command:** `python merginguriel/run_large_scale_experiment.py --modes baseline similarity average --max-locales 10`
    -   **Functionality:** This script iterates through the specified locales. For each one, it runs the baseline evaluation (evaluating the target model on its own language) and then invokes `run_merging_pipeline_refactored.py` for the `similarity` and `average` modes.

3.  **Step 3: Aggregate and Analyze Results**
    -   **Objective:** Collect results from all individual experiment folders and the N-vs-N evaluation to produce a single, comprehensive summary report.
    -   **Script:** `merginguriel/aggregate_results.py`
    -   **Command:** `python merginguriel/aggregate_results.py`
    -   **Output:** Generates comprehensive CSV files and Markdown reports with full baseline integration, win rate analysis, and statistical comparisons. **✅ FULLY IMPLEMENTED** - See section 7.4.

## Workflow 5: Testing & Quality Assurance

This workflow provides comprehensive testing frameworks to ensure system reliability and validate all components.

### 5.1. **Prerequisites**
- **Required Files:** `language_similarity_matrix_unified.csv`, `haryos_model/` directory
- **Processing Module:** `merginguriel/similarity_utils.py` for unified similarity processing
- **Test Framework:** `test_data/` directory with testing utilities

### 5.2. **Generate Test Data**
```bash
python test_data/generate_test_data.py
```
Creates realistic test experiments with:
- 3 locales (sq-AL, th-TH, hr-H) with baseline, similarity, average experiments
- Complete `merge_details.txt` files with source languages and weights
- N-x-N evaluation matrix for baseline integration

### 5.3. **Run Compatibility Tests**
```bash
# Full system test with baseline integration
python merginguriel/aggregate_results.py --verbose

# Test individual components
python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang sq-AL --num-languages 3
python merginguriel/uriel_ensemble_inference.py --target-lang sq-AL --voting-method uriel_logits --num-languages 3

# Test filtering and options
python merginguriel/aggregate_results.py --locales sq-AL th-TH --experiment-types similarity baseline
```

### 5.4. **Test Results Documentation**
- **Comprehensive Report:** `test_data/COMPATIBILITY_TEST.md`
- **Testing Guide:** `test_data/TESTING_GUIDE.md`
- **Status:** ✅ **ALL SYSTEMS COMPATIBLE** - Latest validation confirms full integration

### 5.5. **Current System Status (2025-10-14)**

| Component | Status | Details |
|-----------|--------|---------|
| **Similarity Processing** | ✅ WORKING | Dynamic top-K + Sinkhorn normalization with 50×50 matrix |
| **Model Repository** | ✅ AVAILABLE | 49 models in `/home/coder/Python_project/MergingUriel/haryos_model/` |
| **Merging Pipeline** | ✅ FUNCTIONAL | Successfully merges with URIEL weights, generates proper `merge_details.txt` |
| **Ensemble Inference** | ✅ FUNCTIONAL | URIEL logits weighting with same similarity processing |
| **Automated Evaluation** | ✅ IMPLEMENTED | Comprehensive baseline integration with N-vs-N data |
| **Testing Framework** | ✅ COMPLETE | Full test suite with compatibility validation |

### 5.6. **Key Integration Features**
- **Unified Similarity Processing**: Both merging and ensemble use `similarity_utils.py`
- **Dynamic Top-K Configuration**: Runtime adjustment without matrix regeneration
- **Enhanced Merge Details**: Detailed format with source languages and weights
- **Comprehensive Reporting**: Automated evaluation with win rate analysis and statistical comparison
- **Flexible CLI Interface**: Filtering, verbose logging, and multiple output formats

### Workflow 4: Iterative Training & Merging ✅ **NEW**

1.  **Objective:** Train multiple language models simultaneously with periodic merging during training to foster deeper integration.
2.  **Script:** `merginguriel/run_iterative_training.py`
3.  **Prerequisites:** Sufficient GPU memory for multiple models, locales for training
4.  **Command Example (Auto-select source languages):**
    ```bash
    python merginguriel/run_iterative_training.py \
      --target-lang sw-KE \
      --max-epochs 15 \
      --merge-frequency 3 \
      --merge-algorithm linear \
      --output-dir iterative_results
    ```
5.  **Command Example (Manual source languages):**
    ```bash
    python merginguriel/run_iterative_training.py \
      --target-lang sw-KE \
      --locales en-US,fr-FR,de-DE \
      --max-epochs 15 \
      --merge-frequency 3 \
      --merge-algorithm linear
    ```
6.  **Command Example (Advanced with Adaptive Features):**
    ```bash
    python merginguriel/run_iterative_training.py \
      --target-lang sw-KE \
      --locales en-US,fr-FR,de-DE,es-ES,it-IT \
      --max-epochs 20 \
      --merge-frequency 3 \
      --adaptive-merge-frequency \
      --performance-merge-trigger \
      --convergence-threshold 1e-4 \
      --checkpoint-before-merge \
      --enable-wandb \
      --save-config
    ```
7.  **Key Features:**
    -   Simultaneous multi-locale training with orchestrated merges
    -   Adaptive merge scheduling based on performance convergence
    -   Comprehensive state management and checkpointing
    -   Real-time monitoring and automatic error recovery
    -   Integration with existing MergingUriel pipeline
8.  **Output:** Creates structured output with training logs, merged models, performance metrics, and comprehensive experiment statistics. 

### Workflow 5: URIEL-Guided Ensemble Inference

This workflow provides an alternative to parameter merging by combining model outputs at inference time using the same top-K similarity processing.

1.  **Prerequisites:**
    -   **Required Files:** `language_similarity_matrix_unified.csv` and `haryoaw_k_models.csv`
    -   **Processing Module:** `similarity_utils.py` handles all similarity matrix processing
    -   **No Matrix Generation Needed:** Uses pre-computed similarity matrix with dynamic top-K processing

2.  **Run Single Ensemble Experiment**
    -   **Objective:** Test ensemble inference with a specific target language and voting method.
    -   **Script:** `merginguriel/uriel_ensemble_inference.py`
    -   **Command Example (URIEL logits with Top-K):**
        ```bash
        python merginguriel/uriel_ensemble_inference.py \
            --target-lang "sq-AL" \
            --voting-method "uriel_logits" \
            --num-languages 5 \
            --top-k 20 \
            --sinkhorn-iters 20
        ```
    -   **Command Example (Traditional baseline):**
        ```bash
        python merginguriel/uriel_ensemble_inference.py \
            --target-lang "sq-AL" \
            --voting-method "majority" \
            --num-languages 5
        ```

3.  **Run Comprehensive Ensemble Comparison**
    -   **Objective:** Compare different voting methods across multiple target languages.
    -   **Script:** `merginguriel/comparison_runner.py`
    -   **Command Example:**
        ```bash
        python merginguriel/comparison_runner.py \
            --target-languages "en-US" "sq-AL" "sw-KE" \
            --voting-methods "majority" "weighted_majority" "soft" "uriel_logits" \
            --num-languages 5 \
            --top-k 20
        ```
    -   **Output:** Generates comprehensive comparison report with statistical analysis and method ranking. 

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
