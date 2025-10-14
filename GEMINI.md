# MergingUriel: The Living Modular Blueprint

## 0. Document Control

**Document Status: LIVING DOCUMENT - Last Updated: 2025-10-14 17:20 UTC**

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
2.  **Prerequisites:** Fine-tuned source models and a `sparsed_language_similarity_matrix_unified.csv`.
3.  **Script:** `merginguriel/run_merging_pipeline_refactored.py`
4.  **Command Example (URIEL-Weighted):**
    ```bash
    python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang sq-AL --num-languages 5
    ```
5.  **Command Example (Average Baseline):**
    ```bash
    python merginguriel/run_merging_pipeline_refactored.py --mode average --target-lang sq-AL --num-languages 5
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

### 7.4. ✅ COMPLETED: Comprehensive, Automated Evaluation Report

-   **Goal:** Fully automate the comprehensive evaluation process by refactoring `aggregate_results.py` to include the "Best Source Language" and "Best Overall Zero-Shot" baselines (as defined in Section 5.1) in the final reports.
-   **Status:** ✅ **COMPLETED** - Implemented comprehensive automated evaluation system with full baseline integration.
-   **Solution:** Complete refactoring of `aggregate_results.py` with the following key enhancements:

#### 7.4.1. Dynamic Experiment Parsing
-   **`ExperimentMetadata` Class:** Parses experiment details from `merge_details.txt` files
-   **No Hardcoded Types:** Supports arbitrary merging methods without code changes
-   **Automatic Detection:** Dynamically discovers experiment types and source languages
-   **Metadata Extraction:** Extracts weights, merge modes, and source languages automatically

#### 7.4.2. N-vs-N Baseline Integration
-   **Automatic Discovery:** Finds and loads the most recent `evaluation_matrix.csv` from N-x-N results
-   **Best Source Language Baseline:** Automatically identifies the best performing source model for each target
-   **Best Overall Zero-Shot Baseline:** Finds the best performing any language model (excluding target)
-   **Graceful Fallback:** Handles missing baseline data with informative warnings

#### 7.4.3. Enhanced Reporting System
-   **Dynamic Table Generation:** Creates comparison tables based on discovered experiment types
-   **Multiple Improvement Metrics:** Calculates improvements against all baseline types
-   **Win Rate Analysis:** Determines how often each method outperforms baselines
-   **Statistical Analysis:** Provides comprehensive statistics for all experiment types

#### 7.4.4. Comprehensive Output Formats
-   **Raw Aggregated Data:** Complete dataset with all metadata and baseline information
-   **Comparison Tables:** Dynamic tables with all experiment types and baselines
-   **Comprehensive CSV:** Single file with all metrics for easy analysis and plotting
-   **Markdown Reports:** Professional reports with executive summary, detailed analysis, and methodology
-   **Win Rate Analysis:** JSON export of win rate statistics and performance comparisons

#### 7.4.5. Advanced CLI Interface
```bash
# Basic usage with automatic baseline integration
python merginguriel/aggregate_results.py

# Use specific evaluation matrix
python merginguriel/aggregate_results.py --evaluation-matrix path/to/matrix.csv

# Analyze specific locales and experiment types
python merginguriel/aggregate_results.py --locales sq-AL th-TH --experiment-types similarity average

# Skip baselines for faster processing
python merginguriel/aggregate_results.py --no-baselines --verbose

# Show detailed options
python merginguriel/aggregate_results.py --help
```

#### 7.4.6. Key Features
-   **Extensible:** Supports unlimited merging methods without code modification
-   **Robust Error Handling:** Graceful handling of missing or corrupted data files
-   **Flexible Filtering:** Filter by locales, experiment types, or specific evaluation matrices
-   **Logging System:** Structured logging with multiple verbosity levels
-   **Backward Compatibility:** All existing functionality preserved and enhanced

#### 7.4.7. Example Output Structure
```
results_comprehensive_20251014_172045.csv      # All metrics in one file
results_report_20251014_172045.md               # Professional report
results_aggregated_20251014_172045.csv          # Raw data with metadata
results_comparison_20251014_172045.csv          # Comparison tables
results_summary_20251014_172045.json            # Summary statistics
results_win_analysis_20251014_172045.json       # Win rate analysis
```

The enhanced system now provides a complete, automated evaluation pipeline that fully integrates N-vs-N baselines, supports arbitrary merging methods, and generates comprehensive reports with statistical analysis and win rate analysis.
