# MergingUriel Test Suite

This directory contains the comprehensive test suite for the MergingUriel project, organized by test type to ensure code quality and system reliability.

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