# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-01-19 – Retrospective: Code Simplification

**Type:** Retrospective
**General description:** Migrated MergingUriel from CLI arguments to YAML config system and refactored large modules into packages.

### What we tried

1. **YAML Configuration System**
   - Created `merginguriel/config/` package with hierarchical dataclass configs
   - Implemented `TrackProvidedArgsAction` for CLI argument tracking
   - Added `--config` support to all 4 entry points
   - Built deprecation warning system for CLI-to-config migration

2. **Large Module Refactoring**
   - `plot_results.py`: 2,398 → 853 lines (extracted to `plotting/` package)
   - `aggregate_results.py`: 1,160 → 226 lines (extracted to `aggregation/` package)
   - `naming_config.py`: 526 → 381 lines (dropped legacy patterns)

### Key findings

- Dataclass-based configs provide IDE autocomplete and type hints
- `TrackProvidedArgsAction` cleanly detects explicitly-provided CLI args
- Dropping legacy naming patterns (15 → 3) significantly simplified maintenance
- Verified identical results between original and refactored code (Spearman 0.5779)

### What failed

- Monkeypatching lazy imports: patch at definition site, not usage site
- Hyphen in dataclass field name caused syntax error
- `--resume` flag comparison gave different results due to cached models

### Outcome

- Net reduction of ~3,300 lines through modularization
- 110 tests passing, 38 new config tests
- Created skill: `python-config-refactoring`

---

<!-- New entries go above this line -->
