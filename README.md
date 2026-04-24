# MergingUriel

Cross-lingual model merging for NLU. We merge LoRA adapters from multiple source languages using TIES sign election to improve performance on target languages without any target-language evaluation.

## What This Does

Given LoRA adapters trained on the same task (intent classification) in different languages, we:
1. Select source languages by typological distance (URIEL+)
2. Expand LoRA factors to full parameter space (merge-and-unload)
3. Merge via TIES sign election

This produces a single model that outperforms zero-shot transfer on 32/49 MASSIVE locales, using zero target evaluations and zero model access.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Requires trained LoRA adapters (not included). See `merginguriel/run_all_lora_training.py` to train them.

## Key Scripts

| Script | What it does |
|--------|-------------|
| `merginguriel/run_merging_pipeline_refactored.py` | Main merging pipeline |
| `merginguriel/run_lora_merging_experiment.py` | LoRA-TIES merging experiments |
| `merginguriel/run_large_scale_experiment.py` | Run merging across all locales |
| `merginguriel/run_all_lora_training.py` | Train LoRA adapters for all 49 locales |
| `merginguriel/evaluate_specific_model.py` | Evaluate a model on MASSIVE intent classification |
| `merginguriel/run_nxn_evaluation.py` | Build the 49x49 cross-lingual evaluation matrix |

## Package Structure

```
merginguriel/
    config/             # YAML-based experiment configuration
    experiments/        # Experiment runners
    plotting/           # Result visualization
    aggregation/        # Result aggregation
    selective_layer/    # Layer-selective merging
    synergy_predictor/  # Pre-merge success prediction
    training_lora.py    # LoRA adapter training
    similarity.py       # Language similarity (URIEL+, NxN)
    merge_coordinator.py # TIES/task arithmetic merging
```

## Dataset

[AmazonScience/MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) -- 49 locales, 60 intent classes, ~11.5k train / ~3k test per locale.

## Base Model

XLM-RoBERTa-base (279M) with rank-16 LoRA adapters (~1.2M trainable params each). Also tested at XLM-RoBERTa-large (560M).
