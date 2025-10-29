# Merge Weighting Concern

- **Date:** 2025-10-29
- **Area:** `merginguriel/run_merging_pipeline_refactored.py`
- **Description:** Current merging flow removes the first selected locale from `models_and_weights` and treats it as the `base_model`. Only the remaining K-1 models are passed to `ModelMerger.merge_models`. Inside each strategy (e.g., linear, average, similarity) the base checkpoint serves solely as a parameter sink; its stored weight is not used in the weighted sum. When requesting a merge with `num_languages = K`, the resulting checkpoint is therefore a weighted combination of only K-1 source models, overwriting the base parameters instead of blending them. This diverges from the expected “baseline + sources” aggregation and can bias comparisons.
- **Impact:** Reported weights, metadata, and any downstream analysis that assumes participation of the baseline model are misleading. Parameter deltas and activation analyses effectively compare against a checkpoint that never received the baseline’s intended contribution.

## Proposed Fix

1. Change the merge orchestration so the baseline model participates in the weighted combination:
   - When building `models_to_merge_paths` in `ModelMerger._perform_standard_merge`, include the base model path at the front of the list instead of removing it earlier.
   - Pass matching weights where the baseline’s weight is the first entry (e.g., `weights = [base_model_info.weight] + [info.weight for info in models_and_weights.values()]`).
   - Ensure the merge methods (linear, average, similarity, etc.) receive the augmented weight vector so they compute the full K-model weighted sum.
2. Update `merge_metadata` to reflect the baseline weight explicitly so notebook diagnostics display all contributors.
3. Re-run a small validation merge to confirm that the new logic yields identical results when the baseline weight is zero and produces the expected blended checkpoint when the weight is non-zero.

- **Status:** Pending implementation.
