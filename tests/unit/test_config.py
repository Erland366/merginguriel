"""
Unit tests for the config package.

Tests cover:
- YAML loading and parsing
- Nested dataclass instantiation
- Default values
- Validation
- CLI argument override with deprecation warnings
"""

import pytest
import warnings
import tempfile
from pathlib import Path
from argparse import Namespace

from merginguriel.config import (
    # Base configs
    SimilarityConfig,
    TargetConfig,
    ModelConfig,
    DatasetConfig,
    FisherConfig,
    OutputConfig,
    TrainingConfig,
    MergeConfig,
    # Experiment configs
    MergingExperimentConfig,
    EnsembleExperimentConfig,
    IterativeExperimentConfig,
    PipelineConfig,
    # Loader
    ConfigLoader,
    ConfigDeprecationWarning,
    MERGING_EXPERIMENT_ARG_MAP,
    # Validation
    validate_config,
    validate_and_raise,
    validate_locale,
    ConfigValidationError,
)


class TestBaseConfigs:
    """Tests for base configuration classes."""

    def test_similarity_config_defaults(self):
        """Test SimilarityConfig default values."""
        config = SimilarityConfig()
        assert config.type == "URIEL"
        assert config.source == "dense"
        assert config.top_k == 20
        assert config.sinkhorn_iters == 20

    def test_similarity_config_custom_values(self):
        """Test SimilarityConfig with custom values."""
        config = SimilarityConfig(type="REAL", source="sparse", top_k=10, sinkhorn_iters=5)
        assert config.type == "REAL"
        assert config.source == "sparse"
        assert config.top_k == 10
        assert config.sinkhorn_iters == 5

    def test_similarity_config_invalid_type(self):
        """Test SimilarityConfig rejects invalid type."""
        with pytest.raises(ValueError, match="similarity.type must be"):
            SimilarityConfig(type="INVALID")

    def test_similarity_config_invalid_source(self):
        """Test SimilarityConfig rejects invalid source."""
        with pytest.raises(ValueError, match="similarity.source must be"):
            SimilarityConfig(source="invalid")

    def test_target_config_defaults(self):
        """Test TargetConfig default values."""
        config = TargetConfig()
        assert config.locale == ""
        assert config.inclusion == "ExcTar"
        assert config.include_target is False

    def test_target_config_include_target_property(self):
        """Test TargetConfig include_target property."""
        config_exctar = TargetConfig(inclusion="ExcTar")
        assert config_exctar.include_target is False

        config_inctar = TargetConfig(inclusion="IncTar")
        assert config_inctar.include_target is True

    def test_target_config_invalid_inclusion(self):
        """Test TargetConfig rejects invalid inclusion."""
        with pytest.raises(ValueError, match="target.inclusion must be"):
            TargetConfig(inclusion="invalid")

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.base_model == "xlm-roberta-base"
        assert config.models_root == "haryos_model"
        assert config.num_languages == 5

    def test_model_config_invalid_num_languages(self):
        """Test ModelConfig rejects invalid num_languages."""
        with pytest.raises(ValueError, match="model.num_languages must be"):
            ModelConfig(num_languages=0)

    def test_fisher_config_defaults(self):
        """Test FisherConfig default values."""
        config = FisherConfig()
        assert config.data_mode == "target"
        assert config.preweight == "equal"
        assert config.num_examples == 1000
        assert config.batch_size == 16
        assert config.max_seq_length == 128

    def test_fisher_config_invalid_data_mode(self):
        """Test FisherConfig rejects invalid data_mode."""
        with pytest.raises(ValueError, match="fisher.data_mode must be"):
            FisherConfig(data_mode="invalid")


class TestExperimentConfigs:
    """Tests for experiment configuration classes."""

    def test_merging_experiment_config_defaults(self):
        """Test MergingExperimentConfig default values."""
        config = MergingExperimentConfig()
        assert config.locales is None
        assert config.preset == "none"
        assert config.resume is True
        assert "baseline" in config.modes
        assert config.model.base_model == "xlm-roberta-base"
        assert config.similarity.type == "URIEL"

    def test_merging_experiment_config_invalid_mode(self):
        """Test MergingExperimentConfig rejects invalid modes."""
        with pytest.raises(ValueError, match="Invalid mode"):
            MergingExperimentConfig(modes=["invalid_mode"])

    def test_merging_experiment_config_invalid_preset(self):
        """Test MergingExperimentConfig rejects invalid preset."""
        with pytest.raises(ValueError, match="Invalid preset"):
            MergingExperimentConfig(preset="invalid")

    def test_ensemble_experiment_config_defaults(self):
        """Test EnsembleExperimentConfig default values."""
        config = EnsembleExperimentConfig()
        assert config.target_languages is None
        assert "majority" in config.voting_methods
        assert config.num_examples is None
        assert config.resume is True

    def test_ensemble_experiment_config_invalid_voting_method(self):
        """Test EnsembleExperimentConfig rejects invalid voting method."""
        with pytest.raises(ValueError, match="Invalid voting method"):
            EnsembleExperimentConfig(voting_methods=["invalid"])

    def test_iterative_experiment_config_defaults(self):
        """Test IterativeExperimentConfig default values."""
        config = IterativeExperimentConfig()
        assert config.target_languages is None
        assert config.mode == "similarity"
        assert config.sequential_training is True
        assert config.enable_wandb is False
        assert config.wandb_mode == "disabled"
        assert config.training.epochs == 15
        assert config.merge.frequency == 3

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()
        assert config.mode == "similarity"
        assert config.target.locale == ""
        assert config.similarity.type == "URIEL"


class TestConfigLoader:
    """Tests for ConfigLoader utilities."""

    def test_from_dict_simple(self):
        """Test ConfigLoader.from_dict with simple config."""
        data = {
            "type": "REAL",
            "source": "sparse",
            "top_k": 10,
        }
        config = ConfigLoader.from_dict(SimilarityConfig, data)
        assert config.type == "REAL"
        assert config.source == "sparse"
        assert config.top_k == 10
        assert config.sinkhorn_iters == 20  # Default

    def test_from_dict_nested(self):
        """Test ConfigLoader.from_dict with nested dataclasses."""
        data = {
            "mode": "fisher",
            "similarity": {
                "type": "REAL",
                "top_k": 15,
            },
            "model": {
                "num_languages": 3,
            },
        }
        config = ConfigLoader.from_dict(PipelineConfig, data)
        assert config.mode == "fisher"
        assert config.similarity.type == "REAL"
        assert config.similarity.top_k == 15
        assert config.similarity.source == "dense"  # Default
        assert config.model.num_languages == 3
        assert config.model.base_model == "xlm-roberta-base"  # Default

    def test_to_dict(self):
        """Test ConfigLoader.to_dict."""
        config = PipelineConfig(
            mode="fisher",
            target=TargetConfig(locale="sq-AL", inclusion="IncTar"),
        )
        data = ConfigLoader.to_dict(config)
        assert data["mode"] == "fisher"
        assert data["target"]["locale"] == "sq-AL"
        assert data["target"]["inclusion"] == "IncTar"

    def test_load_yaml(self):
        """Test ConfigLoader.load_yaml."""
        yaml_content = """
mode: fisher
target:
  locale: sq-AL
  inclusion: IncTar
similarity:
  type: REAL
  top_k: 15
model:
  num_languages: 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            data = ConfigLoader.load_yaml(path)
            assert data["mode"] == "fisher"
            assert data["target"]["locale"] == "sq-AL"
            assert data["similarity"]["type"] == "REAL"
        finally:
            path.unlink()

    def test_from_yaml(self):
        """Test ConfigLoader.from_yaml end-to-end."""
        yaml_content = """
mode: fisher
target:
  locale: sq-AL
  inclusion: IncTar
similarity:
  type: REAL
  top_k: 15
model:
  num_languages: 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = ConfigLoader.from_yaml(PipelineConfig, path)
            assert config.mode == "fisher"
            assert config.target.locale == "sq-AL"
            assert config.target.inclusion == "IncTar"
            assert config.similarity.type == "REAL"
            assert config.similarity.top_k == 15
            assert config.model.num_languages == 3
        finally:
            path.unlink()

    def test_load_yaml_file_not_found(self):
        """Test ConfigLoader.load_yaml raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_yaml(Path("/nonexistent/path.yaml"))

    def test_get_nested_attr(self):
        """Test ConfigLoader.get_nested_attr."""
        config = PipelineConfig(
            similarity=SimilarityConfig(type="REAL", top_k=15),
        )
        assert ConfigLoader.get_nested_attr(config, "similarity.type") == "REAL"
        assert ConfigLoader.get_nested_attr(config, "similarity.top_k") == 15
        assert ConfigLoader.get_nested_attr(config, "mode") == "similarity"
        assert ConfigLoader.get_nested_attr(config, "nonexistent", "default") == "default"

    def test_set_nested_attr(self):
        """Test ConfigLoader._set_nested_attr."""
        config = PipelineConfig()
        ConfigLoader._set_nested_attr(config, "similarity.type", "REAL")
        assert config.similarity.type == "REAL"

        ConfigLoader._set_nested_attr(config, "mode", "fisher")
        assert config.mode == "fisher"


class TestValidation:
    """Tests for config validation."""

    def test_validate_locale_valid(self):
        """Test validate_locale with valid locales."""
        assert validate_locale("sq-AL") is None
        assert validate_locale("af-ZA") is None
        assert validate_locale("en-US") is None
        assert validate_locale("th-TH") is None

    def test_validate_locale_invalid(self):
        """Test validate_locale with invalid locales."""
        assert validate_locale("") is not None
        assert validate_locale("sq") is not None
        assert validate_locale("sq_AL") is not None
        assert validate_locale("SQ-AL") is not None
        assert validate_locale("sq-al") is not None

    def test_validate_config_valid(self):
        """Test validate_config with valid config."""
        config = MergingExperimentConfig(
            target=TargetConfig(locale="sq-AL"),
            modes=["baseline", "similarity"],
        )
        errors = validate_config(config)
        assert len(errors) == 0

    def test_validate_config_invalid_mode(self):
        """Test validate_config catches invalid mode."""
        config = MergingExperimentConfig()
        # Manually set invalid mode to bypass __post_init__
        config.modes = ["invalid_mode"]
        errors = validate_config(config)
        assert any("Invalid mode" in e for e in errors)

    def test_validate_config_invalid_locale(self):
        """Test validate_config catches invalid locale."""
        config = MergingExperimentConfig(locales=["invalid"])
        errors = validate_config(config)
        assert any("Invalid locale format" in e for e in errors)

    def test_validate_and_raise_valid(self):
        """Test validate_and_raise with valid config."""
        config = MergingExperimentConfig(
            target=TargetConfig(locale="sq-AL"),
        )
        validate_and_raise(config)  # Should not raise

    def test_validate_and_raise_invalid(self):
        """Test validate_and_raise with invalid config."""
        config = MergingExperimentConfig()
        config.modes = ["invalid"]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_raise(config)
        assert "invalid" in str(exc_info.value).lower()


class TestCLIDeprecationWarnings:
    """Tests for CLI deprecation warning system."""

    def test_merge_cli_args_with_warnings(self):
        """Test that CLI arg override emits deprecation warnings."""
        config = PipelineConfig()

        # Create args namespace with explicitly provided args
        args = Namespace(mode="fisher")
        args._provided_args = {"mode"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ConfigLoader.merge_cli_args(
                config, args, emit_warnings=True, arg_to_config_map={"mode": "mode"}
            )

            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, ConfigDeprecationWarning)
            assert "mode" in str(w[0].message)

        assert config.mode == "fisher"

    def test_merge_cli_args_without_warnings(self):
        """Test CLI arg override without deprecation warnings."""
        config = PipelineConfig()

        args = Namespace(mode="fisher")
        args._provided_args = {"mode"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ConfigLoader.merge_cli_args(
                config, args, emit_warnings=False, arg_to_config_map={"mode": "mode"}
            )

            # No warnings should be emitted
            assert len(w) == 0

        assert config.mode == "fisher"

    def test_merge_cli_args_nested(self):
        """Test CLI arg override for nested config attributes."""
        config = PipelineConfig()

        args = Namespace(similarity_type="REAL", top_k=10)
        args._provided_args = {"similarity_type", "top_k"}

        arg_map = {
            "similarity_type": "similarity.type",
            "top_k": "similarity.top_k",
        }

        config = ConfigLoader.merge_cli_args(
            config, args, emit_warnings=False, arg_to_config_map=arg_map
        )

        assert config.similarity.type == "REAL"
        assert config.similarity.top_k == 10


class TestHierarchicalAccess:
    """Tests verifying hierarchical attribute access pattern."""

    def test_attribute_access_chain(self):
        """Test that hierarchical access works as documented."""
        config = MergingExperimentConfig(
            model=ModelConfig(base_model="xlm-roberta-large", num_languages=10),
            similarity=SimilarityConfig(type="REAL", top_k=15),
            target=TargetConfig(locale="sq-AL", inclusion="IncTar"),
            fisher=FisherConfig(num_examples=500),
        )

        # Verify hierarchical attribute access
        assert config.model.base_model == "xlm-roberta-large"
        assert config.model.num_languages == 10
        assert config.similarity.type == "REAL"
        assert config.similarity.top_k == 15
        assert config.target.locale == "sq-AL"
        assert config.target.inclusion == "IncTar"
        assert config.target.include_target is True  # Property
        assert config.fisher.num_examples == 500

    def test_yaml_to_hierarchical_access(self):
        """Test that YAML maps correctly to hierarchical access."""
        yaml_content = """
model:
  base_model: xlm-roberta-large
  num_languages: 10
similarity:
  type: REAL
  top_k: 15
target:
  locale: sq-AL
  inclusion: IncTar
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = MergingExperimentConfig.from_yaml(path)

            # Verify hierarchical access from YAML
            assert config.model.base_model == "xlm-roberta-large"
            assert config.model.num_languages == 10
            assert config.similarity.type == "REAL"
            assert config.similarity.top_k == 15
            assert config.target.locale == "sq-AL"
            assert config.target.inclusion == "IncTar"
        finally:
            path.unlink()
