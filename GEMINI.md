# MergingUriel: The Living Modular Blueprint

## 0. Document Control

**Document Status: LIVING DOCUMENT - Last Updated: 2025-10-14 19:00 UTC**

**The Golden Rule:** This document, `GEMINI.md`, is the single source of truth for this project. Any developer, human or AI, who modifies the codebase, adds a feature, or changes a workflow **must** update the relevant sections of this document in the same commit/change. This ensures the documentation remains synchronized with the code.

**A Note to Future Developers (Human or AI):** Before committing any changes, you are required to read this document to understand the project's standards and architecture. After implementing your changes, you **must** update the relevant sections of this `GEMINI.md` file and modify the `Last Updated` date and time above. This is a mandatory step for all contributions.

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

### 3.2. Testing & Quality Assurance

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

### 3.3. Hardware & Monitoring

-   **GPU:** The primary development environment is equipped with two NVIDIA RTX 4090 GPUs. However, most individual training and merging tasks are designed to run on a single GPU.
-   **VRAM Management:** It is crucial to monitor GPU memory usage to prevent Out-Of-Memory (OOM) errors, especially when working with large models or batch sizes. Use the following command to check GPU status:

    ```bash
    nvidia-smi
    ```

## 4. Component Breakdown

This section provides a deeper look into the main components of the MergingUriel project.

### `merginguriel/` - The Core Logic

This directory is the heart of the project, containing the Python scripts that define and orchestrate the merging pipeline.

-   **`run_merging_pipeline_refactored.py`**: This is the main entry point for executing a merge. It uses a class-based, strategy pattern design to provide a flexible and extensible workflow. The `--mode` command-line argument is critical as it determines the entire behavior of the pipeline.
    -   **`MergingPipeline`**: The main orchestrator class that coordinates the steps: weight calculation, model merging, and saving the output.
    -   **`WeightCalculatorFactory`**: A factory that, based on the `--mode` argument, instantiates the correct weighting strategy. For example:
        -   `--mode similarity`: Uses `SimilarityWeightCalculator` to load weights from the pre-computed URIEL similarity matrix.
        -   `--mode average`: Uses `AverageWeightCalculator` to assign an equal weight to all source models, establishing a crucial baseline.
    -   **`MergingStrategyFactory`**: A factory that selects the underlying parameter-merging algorithm (e.g., `linear`, `fisher_simple`).
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
    - **`haryoaw_k_models.csv`**: List of available models with locale codes and model paths
    - **`similarity_utils.py`**: Shared processing module (replaces deprecated similarity_matrix.py)

### `auto-merge-llm` Integration

The project leverages the `auto-merge-llm` library (located in `submodules/`) as the powerful backend for performing the parameter-level model merges. This library provides a framework and a collection of established merging algorithms.

-   **High-Level Abstraction:** `MergingUriel` uses `auto-merge-llm` as a library, abstracting away the low-level tensor manipulations. The `run_merging_pipeline_refactored.py` script acts as a high-level orchestrator that prepares models and parameters before feeding them to the selected `auto-merge-llm` method.
-   **Weighting Strategy:** The core innovation of this project is the application of a URIEL-based weighting scheme *on top of* the `auto-merge-llm` methods. The pipeline calculates typologically-informed weights and passes them as parameters (e.g., the `weights` parameter for `linear` merge, or `fisher_scaling_coefficients` for `fisher_dataset` merge) to the `auto-merge-llm` functions.

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

## 5. Supported Merging Methods & Baselines

To scientifically validate the effectiveness of URIEL-guided merging, every experiment is compared against a strong baseline. The `--mode` flag in `run_merging_pipeline_refactored.py` controls which approach is used.

-   **URIEL-Weighted (`--mode similarity`):** Merges the top-N models using weights derived from the URIEL similarity matrix.
-   **Average Baseline (`--mode average`):** Merges the *same* top-N models but assigns each an equal weight (1/N). This isolates the impact of the URIEL weighting scheme itself.

Below is the status of the merging algorithms from `auto-merge-llm` that are currently integrated into this pipeline.

| Merging Method (from auto-merge-llm) | Description | Status in MergingUriel | URIEL Weighting Support | Baseline Comparison |
| :--- | :--- | :--- | :--- | :--- |
| `linear` | A simple weighted average of model parameters. | **Implemented** | Yes | Yes, via `--mode average` |
| `fisher_simple` | Weights parameters by their magnitude as a proxy for importance. | **Implemented** | Yes | Yes, via `--mode average` |
| `fisher_dataset` | Estimates parameter importance using gradients from a sample dataset. | **Implemented** | Yes | Yes, via `--preweight equal` |
| `ties` | Merges models by resolving sign disagreements and pruning low-magnitude weights. | Not Implemented | No | No |
| `slerp` | Spherical Linear Interpolation, useful for interpolating between two models. | Not Implemented | No | No |
| `task_arithmetic` | Adds or subtracts task vectors representing fine-tuning changes. | Not Implemented | No | No |
| `dare` | A state-of-the-art method that prunes and rescales task vectors before merging. | Not Implemented | No | No |
| `regmean` | A method that uses regression to find optimal merging coefficients. | Not Implemented | No | No |
| `breadcrumbs` | A method for merging models by analyzing their training trajectories. | Not Implemented | No | No |

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
    -   **Output:** Generates a CSV file and a Markdown report (`results_report_[timestamp].md`) comparing the performance of the `baseline`, `similarity`, and `average` methods across all tested locales. *Note: This script currently needs to be refactored to include the N-vs-N baselines (see Future Development Plan).*

### Workflow 4: URIEL-Guided Ensemble Inference

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

### 7.1. Support for Advanced Merging Methods

-   **Goal:** Expand the pipeline to support all major merging algorithms available in the `auto-merge-llm` library, such as TIES, DARE, and SLERP.
-   **Implementation Strategy:**
    1.  Extend the `MergingStrategyFactory` in `run_merging_pipeline_refactored.py` to recognize new modes (e.g., `--mode ties`).
    2.  Implement corresponding `MergingStrategy` classes (e.g., `TiesStrategy`) that correctly format the parameters (`scaling_coefficient`, `param_value_mask_rate`, etc.) for the `auto-merge-llm` backend.
    3.  Investigate how URIEL similarity scores can be adapted to serve as inputs for these more complex methods, which may go beyond a simple linear weighting.

### 7.2. Iterative Training & Merging

-   **Goal:** Move beyond a single, post-training merge. This new approach involves merging models *during* the training process itself to foster a more deeply integrated final model.
-   **Implementation Strategy:**
    1.  Develop a new, sophisticated training orchestrator that can manage multiple `Trainer` instances for different languages simultaneously.
    2.  At the end of each epoch (or a set number of steps), the orchestrator will:
        a. Pause training for all models.
        b. Trigger the merging pipeline to combine the current checkpoints of the source models.
        c. Reload the weights of each source model with the weights from the newly merged model.
        d. Resume training for each model on its respective language-specific dataset.
    3.  This will require careful management of model states, checkpoints, and the `Trainer` lifecycle.

### 7.3. URIEL-Guided Ensemble Inference ✅ COMPLETED

-   **Goal:** Explore ensembling as an alternative to parameter merging. Instead of creating a new model, this approach combines the *outputs* (logits) of multiple source models at inference time.
-   **Status:** ✅ **FULLY IMPLEMENTED** - The URIEL-guided ensemble inference system is complete and functional.
-   **Implementation Strategy:**
    1.  **Core Algorithm Implemented:** The system applies URIEL similarity scores as weights to scale logits from source models.
    2.  **Workflow:**
        a. For a given input, get the logit outputs from each of the K source models.
        b. For each source model, multiply its logit tensor by its URIEL similarity score to the target language.
        c. Sum the weighted logits from all source models to produce a final, ensembled logit distribution.
        d. The final prediction is the argmax of this combined distribution.
    3.  **Custom Inference Pipeline:** Built a real-time ensemble system that loads multiple models and performs weighted combination.

#### 7.3.1. Implementation Components

-   **`merginguriel/uriel_ensemble_inference.py`**: Main ensemble inference system with URIEL logits weighting
-   **`merginguriel/ensemble_runner.py`**: Modular experiment runner for individual tests
-   **`merginguriel/comparison_runner.py`**: Comprehensive comparison framework
-   **`merginguriel/similarity_utils.py`**: **NEW** Shared similarity processing module (replaces deprecated similarity_matrix.py)
-   **Files Used**:
    - **`language_similarity_matrix_unified.csv`**: Pre-computed URIEL cosine similarity matrix
    - **`haryoaw_k_models.csv`**: Available models list with direct locale matching

#### 7.3.2. Supported Voting Methods

| Method | Description | URIEL Integration | Status |
|--------|-------------|-------------------|--------|
| `majority` | Simple majority voting | ❌ Traditional baseline | ✅ Implemented |
| `weighted_majority` | Weighted majority using URIEL scores | ✅ URIEL-weighted voting | ✅ Implemented |
| `soft` | Probability averaging with URIEL weights | ✅ URIEL-weighted probabilities | ✅ Implemented |
| `uriel_logits` | **NEW** Direct logits weighting | ✅ **Core implementation** | ✅ Implemented |

#### 7.3.3. Usage Examples

**Basic URIEL Ensemble Inference:**
```bash
python merginguriel/uriel_ensemble_inference.py \
    --target-lang "en-US" \
    --voting-method "uriel_logits" \
    --num-languages 5 \
    --num-examples 100
```

**Comprehensive Comparison:**
```bash
python merginguriel/comparison_runner.py \
    --target-languages "en-US" "sq-AL" "sw-KE" \
    --voting-methods "majority" "weighted_majority" "soft" "uriel_logits" \
    --num-examples 100 \
    --num-languages 5
```

#### 7.3.4. Key Features

- **No Fallback System**: The system catches all errors to ensure complete transparency in model loading failures
- **Dynamic Top-K Processing**: Shared similarity processing module allows flexible configuration
- **Direct Locale Matching**: Simplified architecture using locale codes without complex mapping
- **Comprehensive Evaluation**: Automated comparison framework with statistical analysis
- **Cross-Language Validation**: Demonstrated with multiple target languages (en-US, fr-FR, es-ES, it-IT, de-DE)
- **Detailed Logging**: Complete experiment tracking and result persistence

#### 7.3.5. Real-World Performance Examples

Based on testing with the current model repository:

| Target Language | Available Models Found | Top Weights |
|----------------|----------------------|-------------|
| **es-ES** | 6 models (ar-SA, fr-FR, en-US, it-IT, sq-AL, de-DE) | 0.095, 0.050, 0.047, 0.047, 0.046, 0.046 |
| **it-IT** | 6 models (ar-SA, fr-FR, en-US, sq-AL, de-DE, es-ES) | 0.092, 0.047, 0.046, 0.045, 0.045, 0.044 |
| **fr-FR** | 5 models (ar-SA, en-US, de-DE, sq-AL, it-IT) | 0.094, 0.046, 0.046, 0.045, 0.043 |
| **de-DE** | 4 models (en-US, fr-FR, sq-AL, it-IT) | 0.047, 0.047, 0.044, 0.042 |
| **en-US** | 4 models (fr-FR, de-DE, sq-AL, it-IT) | 0.044, 0.043, 0.040, 0.039 |

### 7.4. Create a Comprehensive, Automated Evaluation Report

-   **Goal:** Fully automate the comprehensive evaluation process by refactoring `aggregate_results.py` to include the "Best Source Language" and "Best Overall Zero-Shot" baselines (as defined in Section 5.1) in the final reports.
-   **Problem:** The current `aggregate_results.py` script is not designed to handle results from new merging methods and does not incorporate the crucial baselines from the N-vs-N evaluation.
-   **Implementation Strategy:**
    1.  **Refactor `aggregate_results.py`:** Modify the script to dynamically parse experiment results without hardcoded assumptions about the merge types (`baseline`, `similarity`, etc.). This could involve reading metadata from a `merge_details.txt` file in each result folder.
    2.  **Integrate N-vs-N Baselines:** The script must be updated to read the `evaluation_matrix.csv` generated by `run_nxn_evaluation.py`.
    3.  **Automated Lookup:** For each merged model result, the script should identify the source models used and automatically look up the corresponding "Best Source Language" and "Best Overall Zero-Shot" scores from the N-vs-N matrix.
    4.  **Unified Reporting:** The final report must include columns for the merged model's performance alongside all key baselines: its own native performance, the average merge baseline, the best source baseline, and the best overall zero-shot baseline.
    5.  **CSV Output:** In addition to the Markdown report, the script must generate a comprehensive CSV file containing all aggregated data for easier analysis and plotting.
