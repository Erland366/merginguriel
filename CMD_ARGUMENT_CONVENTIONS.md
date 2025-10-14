# Command-Line Argument Naming Conventions

This document outlines the standardized naming conventions for command-line arguments across all MergingUriel scripts to ensure consistency and usability.

## Core Arguments (Consistent Across All Scripts)

| Argument | Type | Default | Description | Used In |
|----------|------|---------|-------------|----------|
| `--target-lang` | str | Required (singular) | Target language/locale (e.g., 'en-US', 'sq-AL') | All single-target scripts |
| `--target-languages` | str[] | Multiple values | Target languages for batch processing | Batch/comparison scripts |
| `--num-languages` | int | 5 | Number of source models to include | All scripts |
| `--top-k` | int | 20 | Top-K similar languages to consider | All scripts |
| `--sinkhorn-iters` | int | 20 | Sinkhorn normalization iterations | All scripts |
| `--subfolder-pattern` | str | "alpha_0.5_{locale}_epoch-9" | Model subfolder pattern | Merging scripts |

## Ensemble-Specific Arguments

| Argument | Type | Default | Description | Used In |
|----------|------|---------|-------------|----------|
| `--voting-method` | str | "uriel_logits" | Single voting method to test | `uriel_ensemble_inference.py` |
| `--voting-methods` | str[] | Multiple values | Multiple voting methods for comparison | `comparison_runner.py` |
| `--num-examples` | int | 100 | Number of test examples | Ensemble scripts |
| `--output-dir` | str | "urie_ensemble_results" | Output directory | Ensemble scripts |

## Merging-Specific Arguments

| Argument | Type | Default | Description | Used In |
|----------|------|---------|-------------|----------|
| `--mode` | str | Required | Merging mode (similarity, average, etc.) | `run_merging_pipeline_refactored.py` |
| `--modes` | str[] | Multiple values | Multiple modes for batch processing | `run_large_scale_experiment.py` |
| `--dataset-name` | str | None | HuggingFace dataset name | Fisher merging scripts |
| `--dataset-split` | str | "train" | Dataset split to use | Fisher merging scripts |
| `--text-column` | str | "utt" | Text column name | Fisher merging scripts |
| `--label-column` | str | "label" | Label column name | Fisher merging scripts |
| `--num-fisher-examples` | int | 100 | Examples for Fisher computation | Fisher merging scripts |
| `--fisher-data-mode` | str | "target" | Fisher data distribution mode | Fisher merging scripts |
| `--preweight` | str | "equal" | Pre-weighting strategy | Fisher merging scripts |
| `--batch-size` | int | 16 | Batch size for Fisher computation | Fisher merging scripts |
| `--max-seq-length` | int | 128 | Max sequence length | Fisher merging scripts |

## Naming Convention Rules

### 1. Singular vs Plural Usage
- **Singular form** (`--target-lang`, `--voting-method`, `--mode`): Used for single-target/single-method scripts
- **Plural form** (`--target-languages`, `--voting-methods`, `--modes`): Used for batch/comparison scripts
- **Mutually exclusive**: Scripts support either singular OR plural, not both simultaneously

### 2. Consistency Standards
- **Kebab-case**: All arguments use kebab-case (lowercase with hyphens)
- **Descriptive names**: Argument names clearly indicate their purpose
- **Default values**: All arguments have sensible defaults where appropriate
- **Help text**: All arguments include clear, concise help descriptions

### 3. Argument Validation
- **Required arguments**: Clearly marked and validated
- **Choice validation**: Enumerated choices where applicable
- **Type validation**: Strong type checking for all arguments

## Usage Examples

### Single Target (Singular)
```bash
# Ensemble inference for single target
python merginguriel/uriel_ensemble_inference.py \
    --target-lang "en-US" \
    --voting-method "uriel_logits" \
    --num-languages 5

# Merging for single target
python merginguriel/run_merging_pipeline_refactored.py \
    --mode similarity \
    --target-lang "en-US" \
    --num-languages 5
```

### Batch Processing (Plural)
```bash
# Comparison across multiple targets and methods
python merginguriel/comparison_runner.py \
    --target-languages "en-US" "sq-AL" "sw-KE" \
    --voting-methods "majority" "soft" "uriel_logits"

# Large-scale experiments
python merginguriel/run_large_scale_experiment.py \
    --modes similarity average \
    --max-locales 10
```

## Implementation Notes

### Flexible Argument Handling
Scripts that support both singular and plural forms should implement mutually exclusive argument groups:

```python
target_lang_group = parser.add_mutually_exclusive_group(required=False)
target_lang_group.add_argument("--target-lang", type=str, help="Single target language")
target_lang_group.add_argument("--target-languages", type=str, nargs="+", help="Multiple target languages")
```

### Argument Processing
Always validate and normalize arguments:

```python
# Handle both singular and plural forms
target_languages = args.target_languages if args.target_languages else [args.target_lang]
voting_methods = args.voting_methods if args.voting_methods else [args.voting_method]

# Validate required arguments
if not target_languages or not target_languages[0]:
    parser.error("Either --target-lang or --target-languages must be specified")
```

## Future Consistency

When adding new scripts or arguments:
1. Follow the established naming conventions
2. Use singular form for single-target operations
3. Use plural form for batch/comparison operations
4. Provide clear help text and sensible defaults
5. Update this document when adding new arguments