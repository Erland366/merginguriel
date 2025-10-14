# Tests Directory

This directory contains comprehensive test scripts for the MergingUriel project.

## Directory Structure

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

## Test Files

### ✅ Implemented Methods Tests

- **`test_new_methods_simple.py`** - Quick validation that all new merging strategies can be created correctly
- **`demo_new_merging_methods.py`** - User-friendly demo with usage examples and scientific workflow guidance
- **`test_all_merging_methods.py`** - Comprehensive test suite for all merging methods (longer runtime)

### 🔄 Future Implementation Tests

- **`test_breadcrumbs_merging.py`** - Test framework for the future breadcrumbs merging method implementation

### 📊 Other Project Tests

- **`test_batch_sizes.py`** - Tests for different batch size configurations
- **`test_early_stopping.py`** - Tests for early stopping mechanisms
- **`test_massive_configs.py`** - Tests for MASSIVE dataset configurations
- **`test_run_name_generation.py`** - Tests for experiment run name generation
- **`test_single_evaluation.py`** - Tests for single model evaluation
- **`test_wandb_integration.py`** - Tests for Weights & Biases integration

## Running Tests

### Quick Tests (Recommended for development)

```bash
# Validate all new merging methods work
python tests/test_new_methods_simple.py

# See usage examples and scientific workflow
python tests/demo_new_merging_methods.py
```

### Comprehensive Tests

```bash
# Full test suite (may take longer)
python tests/test_all_merging_methods.py
```

### Future Method Testing

```bash
# See breadcrumbs implementation plan and requirements
python tests/test_breadcrumbs_merging.py
```

## Test Design Philosophy

All tests in this directory are designed as **dry-run validation** that:

- ✅ Don't require actual model files
- ✅ Safe to run in any environment
- ✅ Validate integration and pipeline flow
- ✅ Provide clear success/failure feedback
- ✅ Include helpful error messages for debugging

## Adding New Tests

When adding new merging methods or features:

1. Create a new test file following the naming convention `test_<feature>.py`
2. Include both unit tests and integration tests
3. Follow the dry-run validation pattern
4. Update this README.md with the new test description
5. Add the test to the main documentation if it's a core feature test

## Test Coverage

- ✅ All 4 new advanced merging methods (TIES, Task Arithmetic, SLERP, RegMean)
- ✅ Strategy pattern implementation
- ✅ Argument parser integration
- ✅ Weight calculation integration
- ✅ URIEL similarity score adaptation
- 🔄 Breadcrumbs method (future implementation)
- 📊 Training and evaluation pipeline components


## Running Tests

The project provides a convenient test runner script at the project root:

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

## Test Categories

### Unit Tests
- Test individual components and functions in isolation
- Focus on testing the `similarity_utils.py` module
- Validate core functionality without external dependencies

### Integration Tests
- Test complete workflows and system interactions
- Validate ensemble inference pipeline with real models
- Test merging pipeline integration with similarity processing
- Require actual model files and data to run

### Verification Scripts
- Validate system properties and performance requirements
- Check weight normalization across different systems
- Verify system compliance with specifications
- Used for system validation and quality assurance

## Dependencies

Some tests may require additional dependencies:
- **pytest**: For running unit and integration tests (when available)
- **torch**: Required for model-related tests
- **transformers**: Required for model loading tests
- **pandas**: Required for data processing tests

## Test Data

Some integration tests require:
- `language_similarity_matrix_unified.csv`: Similarity matrix
- `haryoaw_k_models.csv`: Model availability list
- Actual trained models in the model directory

## Notes

- Integration tests may be skipped if required files or models are not available
- Verification scripts are designed to run independently and provide clear pass/fail results
- The test runner automatically falls back to direct Python execution if pytest is unavailable
