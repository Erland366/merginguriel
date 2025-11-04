#!/bin/bash

echo "Running model sanitization tests..."

# Command 1: af-ZA with large model
echo "=== af-ZA with large model ==="
python merginguriel/run_large_scale_experiment.py \
    --locale af-ZA \
    --modes ties \
    --similarity-type REAL \
    --num-languages 3 \
    --models-root haryos_model_large \
    --results-dir testing_chamber/results_afZA_large \
    --merged-models-dir testing_chamber/merged_afZA_large \
    --cleanup-after-eval

# Command 2: af-ZA with base model
echo "=== af-ZA with base model ==="
python merginguriel/run_large_scale_experiment.py \
    --locale af-ZA \
    --modes ties \
    --similarity-type REAL \
    --num-languages 3 \
    --models-root haryos_model \
    --results-dir testing_chamber/results_afZA_base \
    --merged-models-dir testing_chamber/merged_afZA_base \
    --cleanup-after-eval

# Command 3: am-ET with large model
echo "=== am-ET with large model ==="
python merginguriel/run_large_scale_experiment.py \
    --locale am-ET \
    --modes ties \
    --similarity-type REAL \
    --num-languages 3 \
    --models-root haryos_model_large \
    --results-dir testing_chamber/results_amET_large \
    --merged-models-dir testing_chamber/merged_amET_large \
    --cleanup-after-eval

# Command 4: am-ET with base model
echo "=== am-ET with base model ==="
python merginguriel/run_large_scale_experiment.py \
    --locale am-ET \
    --modes ties \
    --similarity-type REAL \
    --num-languages 3 \
    --models-root haryos_model \
    --results-dir testing_chamber/results_amET_base \
    --merged-models-dir testing_chamber/merged_amET_base \
    --cleanup-after-eval

# Command 5: Aggregate all results
echo "=== Aggregating results ==="
# Check which directories exist before aggregating
DIRS=("testing_chamber/results_afZA_large" "testing_chamber/results_afZA_base" "testing_chamber/results_amET_large" "testing_chamber/results_amET_base")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Aggregating results in $dir..."
        python merginguriel/aggregate_results.py --results-dir "$dir" --verbose
    else
        echo "Warning: Directory $dir not found, skipping aggregation"
    fi
done

# Command 6: Generate plots
echo "=== Generating plots ==="
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        plot_dir="testing_chamber/plots_${dir#testing_chamber/results_}"
        echo "Generating plots for $dir..."
        python merginguriel/plot_results.py --results-dir "$dir" --plots-dir "$plot_dir"
    else
        echo "Warning: Directory $dir not found, skipping plots"
    fi
done

echo "All tests completed!"