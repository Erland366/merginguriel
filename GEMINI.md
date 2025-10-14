# MergingUriel: The Living Modular Blueprint

## 0. Document Control

**Document Status: LIVING DOCUMENT - Last Updated: 2025-10-14 18:00 UTC**

**The Golden Rule:** This document, `CLAUDE.md`, is the single source of truth for this project. Any developer, human or AI, who modifies the codebase, adds a feature, or changes a workflow **must** update the relevant sections of this document in the same commit/change. This ensures the documentation remains synchronized with the code.

**A Note to Future Developers (Human or AI):** Before committing any changes, you are required to read this document to understand the project's standards and architecture. After implementing your changes, you **must** update the relevant sections of this `CLAUDE.md` file and modify the `Last Updated` date and time above. This is a mandatory step for all contributions.

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
2.  **URIEL Vector Extraction:** For a given target language, we identify a set of source languages. The typological feature vectors for these languages are extracted using the `lang2vec` library.
3.  **Similarity Matrix Generation:** The extracted URIEL vectors are used to compute a cosine similarity matrix. This matrix is then sparsified (keeping only the top-K most similar languages for each target) and normalized using the Sinkhorn algorithm to become doubly stochastic. This process is handled by `merginguriel/generate_sparse_similarity_unified.py`. The output is a weight matrix where each entry represents the "importance" of a source language to a target language.
4.  **Model Merging:** The main pipeline script, `merginguriel/run_merging_pipeline_refactored.py`, reads the similarity matrix to select source models and their corresponding weights. It then uses a merging algorithm provided by the `auto-merge-llm` library to combine the parameters of the source models into a new, single merged model.
5.  **Evaluation:** The newly created merged model is evaluated on a benchmark task (e.g., STS-B) to measure its performance, using scripts like `merginguriel/evaluate_base_encoder.py`.

## 3. Environment Setup

Before running any training or merging workflows, ensure the project environment is correctly set up.

### 3.1. Activate Virtual Environment

All dependencies are managed within a Python virtual environment. Activate it using:

```bash
source .venv/bin/activate
```

### 3.2. Hardware & Monitoring

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
    -   **`MergingStrategyFactory`**: A factory that selects the underlying parameter-merging algorithm (e.g., `linear`, `fisher_simple`).
    -   **`ModelMerger`**: The class that takes the models, weights, and merging strategy and invokes the `auto-merge-llm` library to perform the merge.
    -   **`OutputManager`**: Saves the final merged model and a `merge_details.txt` file that records the configuration used for the run.

-   **`training_bert.py`**: A standard Hugging Face `Trainer` script used to fine-tune the `xlm-roberta-base` model on individual language subsets of the `AmazonScience/massive` dataset. The models produced by this script are the raw inputs for the merging process.

-   **`generate_sparse_similarity_unified.py`**: This utility creates the core guidance mechanism for the merge. It uses the `lang2vec` library to fetch URIEL vectors for all languages, computes a dense cosine similarity matrix, sparsifies it by keeping only the top-K most similar languages for each language, and finally applies Sinkhorn normalization to make the matrix doubly stochastic. The result is a matrix where each value can be interpreted as the influence of a source language on a target language.

-   **`evaluate_base_encoder.py`**: A script for evaluating a model's semantic representation capabilities on the STS-B benchmark. It calculates the Spearman correlation between the cosine similarity of sentence embeddings and the human-annotated similarity scores.

### `auto-merge-llm` Integration

The project leverages the `auto-merge-llm` library (located in `submodules/`) as the powerful backend for performing the parameter-level model merges. This library provides a framework and a collection of established merging algorithms.

-   **High-Level Abstraction:** `MergingUriel` uses `auto-merge-llm` as a library, abstracting away the low-level tensor manipulations. The `run_merging_pipeline_refactored.py` script acts as a high-level orchestrator that prepares models and parameters before feeding them to the selected `auto-merge-llm` method.
-   **Weighting Strategy:** The core innovation of this project is the application of a URIEL-based weighting scheme *on top of* the `auto-merge-llm` methods. The pipeline calculates typologically-informed weights and passes them as parameters (e.g., the `weights` parameter for `linear` merge, or `fisher_scaling_coefficients` for `fisher_dataset` merge) to the `auto-merge-llm` functions.

### `lang2vec` & URIEL Vectors

The `lang2vec` library (located in `submodules/`) is the interface to the URIEL typological database.

-   **Usage:** It is primarily used within `generate_sparse_similarity_unified.py` to fetch feature vectors for different languages (e.g., `l2v.get_features(languages, "syntax_knn")`). These vectors are the foundation for the similarity scores that guide the merge.

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
2.  **Prerequisites:** Fine-tuned source models and a `sparsed_language_similarity_matrix_unified.csv`.
3.  **Script:** `merginguriel/run_merging_pipeline_refactored.py`
4.  **Command Examples:**
    ```bash
    # URIEL-Weighted (recommended for production)
    python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang sq-AL --num-languages 5

    # Average Baseline (for comparison)
    python merginguriel/run_merging_pipeline_refactored.py --mode average --target-lang sq-AL --num-languages 5

    # Advanced Merging Methods
    python merginguriel/run_merging_pipeline_refactored.py --mode ties --target-lang sq-AL --num-languages 5
    python merginguriel/run_merging_pipeline_refactored.py --mode task_arithmetic --target-lang sq-AL --num-languages 5
    python merginguriel/run_merging_pipeline_refactored.py --mode slerp --target-lang sq-AL --num-languages 5
    python merginguriel/run_merging_pipeline_refactored.py --mode regmean --target-lang sq-AL --num-languages 5
    ```

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

## 7. Future Development Plan

This section outlines the strategic goals for the evolution of the MergingUriel project.

### 7.1. Support for Advanced Merging Methods ✅ COMPLETED

-   **Status:** ✅ **COMPLETED** - All major advanced merging methods have been successfully integrated into the pipeline.
-   **Implemented Methods:**
    -   **TIES**: Resolves sign disagreements and prunes low-magnitude weights (--mode ties)
    -   **Task Arithmetic**: Adds/subtracts task vectors representing fine-tuning changes (--mode task_arithmetic)
    -   **SLERP**: Spherical Linear Interpolation for models (--mode slerp)
    -   **RegMean**: Uses regression to find optimal merging coefficients (--mode regmean)
    -   **DARE**: Implemented through task_arithmetic with pruning parameters
-   **Implementation Details:**
    1.  ✅ Extended the `MergingStrategyFactory` to recognize new modes: `ties`, `task_arithmetic`, `slerp`, `regmean`
    2.  ✅ Implemented corresponding `MergingStrategy` classes with proper parameter formatting for the `auto-merge-llm` backend
    3.  ✅ Adapted URIEL similarity scores to influence each merging method:
        -   **TIES**: Uses similarity scores to adjust scaling coefficients
        -   **Task Arithmetic**: Directly scales task vectors by URIEL weights
        -   **SLERP**: Uses average of similarity weights for interpolation ratios
        -   **RegMean**: Uses similarity weights as priors for regression coefficients
-   **Usage:** All new methods support both URIEL-weighted (`--mode similarity`) and average baseline (`--mode average`) comparisons for scientific evaluation.

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

### 7.3. URIEL-Guided Ensemble Inference

-   **Goal:** Explore ensembling as an alternative to parameter merging. Instead of creating a new model, this approach combines the *outputs* (logits) of multiple source models at inference time.
-   **Implementation Strategy (TODO):**
    1.  The primary research question is how to best utilize URIEL vectors in this context.
    2.  A potential approach is to use the URIEL similarity score between a source language and the target language to **scale the logits** produced by that source model.
    3.  The proposed workflow would be:
        a. For a given input, get the logit outputs from each of the K source models.
        b. For each source model, multiply its logit tensor by its URIEL similarity score to the target language.
        c. Sum the weighted logits from all source models to produce a final, ensembled logit distribution.
        d. The final prediction is the argmax of this combined distribution.
    4.  This requires building a custom inference pipeline that can load multiple models and perform this weighted combination in real-time.

### 7.4. Implement Breadcrumbs Merging Method

-   **Goal:** Implement the `breadcrumbs` merging method from `auto-merge-llm` to enable trajectory-based model merging.
-   **Status:** 🔄 **FUTURE IMPLEMENTATION** - Requires training pipeline modifications
-   **Why Breadcrumbs?**
    -   Analyzes training trajectories rather than just final model states
    -   Could provide better merging for models with different learning patterns
    -   Offers novel approach for cross-lingual transfer considering language-specific learning trajectories
-   **Implementation Challenges:**
    -   Requires significant changes to training pipeline to save intermediate checkpoints
    -   Needs substantial storage for trajectory data (loss, gradients, parameter changes)
    -   Complex trajectory comparison and similarity algorithms
    -   Higher computational overhead during merging process
-   **Implementation Strategy:**
    1.  **Modify Training Pipeline:** Update `merginguriel/training_bert.py` to save intermediate checkpoints and training metrics
    2.  **Design Trajectory Storage:** Create efficient format for storing and compressing training trajectories
    3.  **Implement Trajectory Analysis:** Develop algorithms to compare and weight different learning paths
    4.  **Create BreadcrumbsStrategy:** Implement strategy class with URIEL integration for trajectory weighting
    5.  **Update Pipeline:** Integrate breadcrumbs mode into existing merging infrastructure
    6.  **Testing:** Comprehensive validation using `tests/test_breadcrumbs_merging.py`
-   **Test Coverage:** Test framework already prepared in `tests/test_breadcrumbs_merging.py`

### 7.5. Create a Comprehensive, Automated Evaluation Report

-   **Goal:** Fully automate the comprehensive evaluation process by refactoring `aggregate_results.py` to include the "Best Source Language" and "Best Overall Zero-Shot" baselines (as defined in Section 5.1) in the final reports.
-   **Problem:** The current `aggregate_results.py` script is not designed to handle results from new merging methods and does not incorporate the crucial baselines from the N-vs-N evaluation.
-   **Implementation Strategy:**
    1.  **Refactor `aggregate_results.py`:** Modify the script to dynamically parse experiment results without hardcoded assumptions about the merge types (`baseline`, `similarity`, etc.). This could involve reading metadata from a `merge_details.txt` file in each result folder.
    2.  **Integrate N-vs-N Baselines:** The script must be updated to read the `evaluation_matrix.csv` generated by `run_nxn_evaluation.py`.
    3.  **Automated Lookup:** For each merged model result, the script should identify the source models used and automatically look up the corresponding "Best Source Language" and "Best Overall Zero-Shot" scores from the N-vs-N matrix.
    4.  **Unified Reporting:** The final report must include columns for the merged model's performance alongside all key baselines: its own native performance, the average merge baseline, the best source baseline, and the best overall zero-shot baseline.
    5.  **CSV Output:** In addition to the Markdown report, the script must generate a comprehensive CSV file containing all aggregated data for easier analysis and plotting.
