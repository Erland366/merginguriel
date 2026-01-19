# AGENTS.md — Project Operating Contract

> **Domain:** research
> **Created:** 2026-01-14

This document defines the operational contract for AI agents working in this project.

---

## 1. Project Context

**Purpose:** Research project

**Domain:** ML research and experimentation

---

## 2. Directory Structure

```
./
├── .codex/skills/          # Local skills (readable by Codex)
│   ├── registry.json       # Domain metadata and skill index
│   ├── domain-advisor/     # Planning assistance
│   └── domain-retrospective/  # Knowledge capture
├── .claude/
│   └── skills/             # Symlink to .codex/skills/ (for Claude Code)
├── .claude-plugin/
│   └── plugin.json         # Claude Code plugin manifest
├── references/
│   ├── experiment-log.md   # Chronological log
│   └── troubleshooting.md  # Error patterns and fixes
├── templates/              # Document templates
├── training_reports/          # Experiment/benchmark reports
└── AGENTS.md               # This file
```

---

## 3. Skill Commands

### `<advise>`

Invoke when planning new experiments or development tasks.

**Behavior:**
1. Reads `registry.json` to determine domain
2. Scans existing skills and reports for relevant context
3. Proposes 2-5 concrete experiments/tasks
4. Outputs markdown table with key differences

### `<retrospective>`

Invoke after completing experiments to capture learnings.

**Behavior:**
1. Reads specified reports from `training_reports/`
2. Summarizes: what worked, what failed, key insights
3. Proposes new result skills or troubleshooting entries
4. Only writes files with user approval

---

## 4. Documentation Rules

### Reports

- Store in `training_reports/`
- Use template: `templates/reports/report-template.md`
- Name format: `{description}-{YYYY-MM-DD}.md`

### Experiment Log

- Location: `references/experiment-log.md`
- Append entries chronologically
- Include: date, type, general description, details

### Troubleshooting

- Location: `references/troubleshooting.md`
- Add new error patterns as discovered
- Include: symptom, cause, solution

### Result Skills

- Location: `.codex/skills/{skill-name}/SKILL.md`
- Use template: `templates/skills/result-skill-template.md`
- Must include: description, when to apply, failure modes

---

## 5. Workflow

1. **Start session**: Type `<advise>` to get planning suggestions
2. **Execute**: Run experiments/development tasks
3. **Document**: Create reports in `training_reports/`
4. **Capture**: Type `<retrospective>` to distill learnings
5. **Iterate**: Use new skills in next `<advise>` cycle

---

## 6. MergingUriel Research Context

### Research Goals

MergingUriel investigates **cross-lingual model merging** for NLU tasks:

1. **Beat zero-shot cross-lingual performance**: A merged model should outperform individual source language models evaluated directly on the target language.

2. **Significantly improve over baseline**: The merged model should substantially beat the pretrained multilingual model (XLM-RoBERTa) without task-specific fine-tuning.

### The Task: MASSIVE Intent Classification

- **Dataset**: AmazonScience/massive (49 locales, 60 intent classes)
- **Task**: Sequence classification (intent prediction from utterance)
- **Metric**: Accuracy = correct_predictions / total_predictions

### Evaluation Framework

**Correct evaluation**: `evaluate_specific_model.py` (MASSIVE intent classification)
**Proxy metric**: `evaluate_base_encoder.py` (STS-B correlation) — NOT the final metric

**Baselines from NxN matrix** (`nxn_results/*/evaluation_matrix.csv`):
- Zero-shot: Source model tested directly on target language
- Best source: Highest-performing source model for target
- Best overall: Best zero-shot across all source models

### Success Criteria

```
merged_accuracy > max(source_accuracies_on_target)  # Beat best zero-shot
merged_accuracy > pretrained_xlm_roberta_baseline   # Beat raw pretrained
```

### Key Entry Points

| Script | Purpose |
|--------|---------|
| `run_merging_pipeline_refactored.py` | Merge models and evaluate |
| `run_large_scale_experiment.py` | Run merging across all locales |
| `run_large_scale_ensemble_experiments.py` | Ensemble inference experiments |
| `run_large_scale_iterative_training.py` | Iterative training with merging |

---

## 7. Domain-Specific Notes

- Focus on reproducibility: log all hyperparameters
- Track dataset versions and preprocessing
- Document negative results - they prevent repeated mistakes

---

## 8. Conventions

- **File naming:** lowercase with hyphens (e.g., `my-experiment.md`)
- **Skill naming:** `{topic}-{finding}` (e.g., `lora-rank-optimal`)
- **Dates:** YYYY-MM-DD format
- **Configs:** YAML or JSON, copy-paste ready
