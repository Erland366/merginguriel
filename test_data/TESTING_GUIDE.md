# Testing Guide for Comprehensive Automated Evaluation Report

## Quick Start Testing

### 1. Generate Test Data
```bash
python test_data/generate_test_data.py
```

This creates:
- 3 locales (sq-AL, th-TH, hr-HR) with baseline, similarity, and average experiments
- Complete `merge_details.txt` files with source languages and weights
- N-x-N evaluation matrix for baseline integration

### 2. Basic Testing
```bash
# Full test with baseline integration
python merginguriel/aggregate_results.py --verbose

# Test filtering capabilities
python merginguriel/aggregate_results.py --locales sq-AL th-TH --verbose
python merginguriel/aggregate_results.py --experiment-types similarity baseline --verbose

# Test without baselines (faster)
python merginguriel/aggregate_results.py --no-baselines --verbose

# Test with specific evaluation matrix
python merginguriel/aggregate_results.py --evaluation-matrix test_data/sample_evaluation_matrix.csv
```

### 3. Test Output Files
The system generates these files with timestamps:
- `results_aggregated_*.csv` - Raw data with all metadata
- `results_comprehensive_*.csv` - All metrics in one file
- `results_comparison_*.csv` - Comparison tables
- `results_summary_*.json` - Summary statistics
- `results_win_analysis_*.json` - Win rate analysis
- `results_report_*.md` - Professional markdown report

## Advanced Testing Scenarios

### Scenario 1: Test with Custom Evaluation Matrix
```bash
# Use your own evaluation matrix
python merginguriel/aggregate_results.py --evaluation-matrix /path/to/your/matrix.csv
```

### Scenario 2: Test with Real Experiment Data
1. Run actual experiments:
   ```bash
   python merginguriel/run_merging_pipeline_refactored.py --mode similarity --target-lang sq-AL --num-languages 5
   ```

2. Aggregate results:
   ```bash
   python merginguriel/aggregate_results.py --verbose
   ```

### Scenario 3: Test with Missing Data
```bash
# Test show-missing functionality
python merginguriel/aggregate_results.py --show-missing --experiment-types baseline similarity fisher_dataset ties dare
```

### Scenario 4: Performance Testing
```bash
# Test with no baselines for faster processing
python merginguriel/aggregate_results.py --no-baselines

# Test with specific locales for focused analysis
python merginguriel/aggregate_results.py --locales sq-AL th-TH --verbose
```

## Expected Test Results

### Test Data Structure
```
results/
├── baseline_sq-AL/
│   ├── results.json
│   └── merge_details.txt (not created for baseline)
├── similarity_sq-AL/
│   ├── results.json
│   └── merge_details.txt
├── average_sq-AL/
│   ├── results.json
│   └── merge_details.txt
└── ... (same for th-TH and hr-HR)

nxn_results/nxn_eval_test/
└── evaluation_matrix.csv
```

### Key Features to Verify

1. **Dynamic Experiment Parsing**
   - System should detect similarity, average, baseline experiment types
   - Should read merge_details.txt correctly for merging experiments
   - Should extract source languages and weights

2. **Baseline Integration**
   - Should automatically load evaluation_matrix.csv
   - Should calculate Best Source Language baseline
   - Should calculate Best Overall Zero-Shot baseline

3. **Comprehensive Reporting**
   - Executive summary with experiment counts
   - Summary statistics for all experiment types
   - Win rate analysis showing how often methods beat baselines
   - Detailed comparison tables with all metrics

4. **Output Files**
   - All 6 output files should be generated
   - CSV files should contain complete data
   - Markdown report should be professional and readable
   - JSON files should contain structured data

### Example Expected Output

From our test run, you should see:
```
2025-10-14 17:28:22,689 - INFO - Summary Statistics:
2025-10-14 17:28:22,689 - INFO -   average: 3 experiments, mean accuracy = 0.8301
2025-10-14 17:28:22,689 - INFO -   baseline: 3 experiments, mean accuracy = 0.8230
2025-10-14 17:28:22,689 - INFO -   similarity: 3 experiments, mean accuracy = 0.8456
2025-10-14 17:28:22,690 - INFO - Win Rate Analysis Summary:
2025-10-14 17:28:22,690 - INFO -   similarity: 100.0% win rate vs baseline (3/3)
```

## Troubleshooting

### Common Issues

1. **"No results found in the results directory"**
   - Solution: Run `python test_data/generate_test_data.py` first

2. **"No evaluation matrix found"**
   - Solution: Ensure `nxn_results/nxn_eval_test/evaluation_matrix.csv` exists
   - Or use `--evaluation-matrix` flag to specify path

3. **Merge details parsing errors**
   - Solution: Check that merge_details.txt files are properly formatted
   - Run test data generation again

### Debug Options

```bash
# Enable verbose logging
python merginguriel/aggregate_results.py --verbose

# Show missing experiments
python merginguriel/aggregate_results.py --show-missing

# Skip problematic components
python merginguriel/aggregate_results.py --no-baselines
```

## Custom Test Data

### Creating Custom Test Data

1. **Modify test_data/generate_test_data.py**:
   - Add new locales in the `test_cases` list
   - Adjust accuracy values and source languages
   - Add new experiment types

2. **Create Manual Test Data**:
   ```bash
   mkdir -p results/custom_test
   # Create results.json and merge_details.txt files
   ```

3. **Create Custom Evaluation Matrix**:
   ```bash
   cp test_data/sample_evaluation_matrix.csv custom_matrix.csv
   # Edit custom_matrix.csv with your data
   ```

## Performance Considerations

- **With baselines**: Slower but comprehensive analysis
- **Without baselines**: 10x faster for quick testing
- **Filtering**: Reduces processing time significantly
- **Large datasets**: Use `--locales` and `--experiment-types` for focused analysis

This testing framework provides comprehensive validation of all features in the automated evaluation report system!