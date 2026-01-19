---
name: python-config-refactoring
description: >
  Patterns for migrating CLI-heavy Python projects to YAML configuration systems.
  Use when: refactoring scattered argparse arguments into unified config files.
metadata:
  short-description: "CLI-to-YAML config migration patterns"
  tags:
    - refactoring
    - configuration
    - python
    - argparse
  domain: research
  created: 2026-01-19
  author: Claude Code
---

# Python Config Refactoring

## General Description

This skill captures patterns for migrating Python projects from scattered CLI arguments to unified YAML configuration systems while maintaining backward compatibility. The key insight is using dataclass hierarchies for type-safe config access and a custom argparse action to detect explicitly-provided arguments.

## When to Apply

Use this knowledge when:
- A project has 20+ CLI arguments spread across multiple entry points
- You need backward compatibility during migration (CLI args still work)
- You want IDE autocomplete and type checking for configuration
- Multiple scripts share overlapping configuration parameters

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| CLI args consolidated | 50+ → 1 `--config` | Plus all original args for backward compat |
| Code reduction | ~3,300 lines net | Through modularization |
| Test coverage | 38 new unit tests | For config loading/validation |
| Migration effort | ~2 sessions | Foundation + entry point updates |

## Recommended Practice

### 1. TrackProvidedArgsAction Pattern

Track which CLI arguments were explicitly provided (vs defaults):

```python
class TrackProvidedArgsAction(argparse.Action):
    """Custom argparse action that tracks which arguments were explicitly provided."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if not hasattr(namespace, "_provided_args"):
            namespace._provided_args = set()
        namespace._provided_args.add(self.dest)

# Usage
parser.add_argument("--batch-size", type=int, default=16, action=TrackProvidedArgsAction)
args = parser.parse_args()

# Check if explicitly provided
if "batch_size" in getattr(args, "_provided_args", set()):
    print("User explicitly set batch_size")
```

### 2. Dataclass Config Hierarchy

Use nested dataclasses for hierarchical config access:

```python
@dataclass
class ModelConfig:
    base_model: str = "xlm-roberta-base"
    num_languages: int = 5

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
        )

# Clean attribute access
config = ExperimentConfig.from_yaml("config.yaml")
print(config.model.base_model)  # IDE autocomplete works
```

### 3. CLI Override with Deprecation Warnings

```python
def merge_cli_args(config, args, emit_warnings: bool = True):
    """Override config values with explicitly-provided CLI args."""
    provided = getattr(args, "_provided_args", set())

    for arg_name in provided:
        if arg_name == "config":
            continue
        if emit_warnings:
            warnings.warn(
                f"CLI argument --{arg_name.replace('_', '-')} overrides config file. "
                "Consider moving this to your config file.",
                DeprecationWarning
            )
        # Map CLI arg to config attribute and override
        _apply_override(config, arg_name, getattr(args, arg_name))

    return config
```

### 4. Package Extraction Checklist

When to extract a module into a package:
- [ ] File exceeds ~500 lines
- [ ] Clear separation of concerns exists (data loading vs processing vs output)
- [ ] Multiple classes/functions could be tested independently
- [ ] Other modules only need a subset of the functionality

Structure pattern:
```
module.py (500+ lines) →
package/
├── __init__.py      # Public API exports
├── data_loader.py   # Data loading/parsing
├── processors.py    # Core logic
└── utils.py         # Helpers
```

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Hyphens in dataclass fields | `max-gpu-memory` is invalid Python | Always use underscores: `max_gpu_memory` |
| Patching lazy imports | `naming_manager` imported inside function | Patch at definition site: `naming_config.naming_manager` |
| Dict-style config access | `config["model"]["base"]` verbose | Use dataclasses for `config.model.base` |
| Keeping all legacy patterns | 15 regex patterns for backward compat | Drop legacy if migration is complete |

## Configuration

Example YAML structure:

```yaml
# config.yaml
model:
  base_model: "xlm-roberta-base"
  models_root: "haryos_model"
  num_languages: 5

target:
  locale: "sq-AL"
  inclusion: "ExcTar"

similarity:
  type: "URIEL"
  top_k: 20

output:
  results_dir: "results"
  cleanup_after_eval: false
```

## Related Files

- `merginguriel/config/loader.py` - TrackProvidedArgsAction implementation
- `merginguriel/config/base.py` - Nested dataclass configs
- `merginguriel/config/experiment.py` - Top-level experiment configs
- `configs/examples/` - Example YAML configurations
