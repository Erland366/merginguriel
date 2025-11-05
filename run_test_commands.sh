#!/bin/bash

echo "Running model sanitization tests with IT/ET functionality..."

# Command 1: af-ZA with large model (ET - Exclude Target)
echo "=== af-ZA with large model (ET) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales af-ZA \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --models-root haryos_model_large \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 2: af-ZA with large model (IT - Include Target)
# echo "=== af-ZA with large model (IT) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales af-ZA \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --include-target \
#     --models-root haryos_model_large \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 3: af-ZA with base model (ET - Exclude Target)
# echo "=== af-ZA with base model (ET) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales af-ZA \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --models-root haryos_model \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 4: af-ZA with base model (IT - Include Target)
# echo "=== af-ZA with base model (IT) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales af-ZA \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --include-target \
#     --models-root haryos_model \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 5: am-ET with large model (ET - Exclude Target)
# echo "=== am-ET with large model (ET) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales am-ET \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --models-root haryos_model_large \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 6: am-ET with large model (IT - Include Target)
# echo "=== am-ET with large model (IT) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales am-ET \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --include-target \
#     --models-root haryos_model_large \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 7: am-ET with base model (ET - Exclude Target)
# echo "=== am-ET with base model (ET) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales am-ET \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --models-root haryos_model \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

# # Command 8: am-ET with base model (IT - Include Target)
# echo "=== am-ET with base model (IT) ==="
# python merginguriel/run_large_scale_experiment.py \
#     --locales am-ET \
#     --modes ties \
#     --similarity-type REAL \
#     --num-languages 3 \
#     --include-target \
#     --models-root haryos_model \
#     --results-dir testing_chamber/results \
#     --merged-models-dir testing_chamber/merged_models \
#     --cleanup-after-eval

echo "=== Aggregating results ==="
python -m merginguriel.aggregate_results --results-dir testing_chamber/results --output-prefix final_results

echo "=== Generating plots ==="
python -m merginguriel.plot_results --results-dir testing_chamber/results --plots-dir testing_chamber/plots

echo "All tests completed!"