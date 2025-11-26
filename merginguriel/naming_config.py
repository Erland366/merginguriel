"""
Centralized naming configuration for all experiment-related naming schemes.
This ensures consistency across merging, evaluation, plotting, ensemble, and iterative training components.

No fallbacks - strict validation with early error detection.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import re
import os
from datetime import datetime

@dataclass
class NamingConfig:
    """Centralized configuration for all naming schemes."""

    # Core naming patterns
    results_dir_pattern: str = "{experiment_type}_{method}_{similarity_type}_{locale}_{model_family}_{num_languages}lang_{include_target}"
    merged_model_pattern: str = "{method}_{similarity_type}_merge_{locale}_{model_family}_{num_languages}merged_{include_target}"
    plot_filename_pattern: str = "{method}_{model_family}_{similarity_type}_{num_languages}lang_{include_target}"

    # Simple patterns for baseline
    baseline_results_pattern: str = "baseline_{locale}_{model_family}"
    baseline_plot_pattern: str = "baseline_{model_family}"

    # Model directory pattern (unchanged - source models)
    model_pattern: str = "{models_root}/{model_family}_massive_k_{locale}"

    # Valid similarity types
    similarity_types: list = None

    def __post_init__(self):
        if self.similarity_types is None:
            self.similarity_types = ['URIEL', 'REAL']

class NamingManager:
    """Manages all naming operations using centralized configuration."""

    def __init__(self, config: Optional[NamingConfig] = None):
        self.config = config or NamingConfig()

    def get_model_path(self, models_root: str, model_size: str, locale: str) -> str:
        """Generate model directory path."""
        return self.config.model_pattern.format(
            models_root=models_root,
            model_size=model_size,
            locale=locale
        )

    def get_results_dir_name(self, experiment_type: str, method: str, similarity_type: str,
                            locale: str, model_family: str, num_languages: Optional[int] = None,
                            include_target: bool = False, timestamp: Optional[str] = None) -> str:
        """Generate results directory name."""
        if experiment_type == 'baseline':
            return self.config.baseline_results_pattern.format(
                locale=locale,
                model_family=model_family
            )
        else:
            if num_languages is None:
                raise ValueError(f"num_languages is required for {experiment_type} experiments")

            include_target_suffix = "IncTar" if include_target else "ExcTar"
            return self.config.results_dir_pattern.format(
                experiment_type=experiment_type,
                method=method,
                similarity_type=similarity_type,
                locale=locale,
                model_family=model_family,
                num_languages=num_languages,
                include_target=include_target_suffix
            )

    def get_merged_model_dir_name(self, experiment_type: str, method: str, similarity_type: str,
                                  locale: str, model_family: str, num_languages: int,
                                  include_target: bool = False, timestamp: Optional[str] = None) -> str:
        """Generate merged model directory name."""
        if experiment_type == 'baseline':
            raise ValueError("Baseline experiments don't create merged models")

        include_target_suffix = "IncTar" if include_target else "ExcTar"
        return self.config.merged_model_pattern.format(
            method=method,
            similarity_type=similarity_type,
            locale=locale,
            model_family=model_family,
            num_languages=num_languages,
            include_target=include_target_suffix
        )

    def get_plot_filename(self, method: str, model_family: str, similarity_type: str,
                         num_languages: Optional[int] = None, include_target: bool = False,
                         timestamp: Optional[str] = None) -> str:
        """Generate plot filename."""
        if num_languages is None:
            # For baseline plots
            return self.config.baseline_plot_pattern.format(
                model_family=model_family
            )
        else:
            # For method plots
            include_target_suffix = "IncTar" if include_target else "ExcTar"
            return self.config.plot_filename_pattern.format(
                method=method,
                model_family=model_family,
                similarity_type=similarity_type,
                num_languages=num_languages,
                include_target=include_target_suffix
            )

    def detect_model_size_from_root(self, models_root: str) -> str:
        """Detect model size from models root directory."""
        if "large" in models_root.lower():
            return "large"
        elif "base" in models_root.lower():
            return "base"
        raise ValueError(f"Cannot determine model size from models_root: {models_root}")

    def detect_model_family_from_path(self, model_path: str) -> str:
        """Detect model family from HuggingFace model path."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)

            # Extract model family from various sources
            if hasattr(config, 'name_or_path'):
                model_name = config.name_or_path
            elif hasattr(config, '_name_or_path'):
                model_name = config._name_or_path
            elif hasattr(config, 'model_type'):
                model_name = config.model_type
            else:
                raise ValueError(f"Cannot determine model name from config for {model_path}")

            # Clean up the model name
            if '/' in model_name:
                model_name = model_name.split('/')[-1]

            # Remove common suffixes
            for suffix in ['-uncased', '-cased', '-v1', '-v2']:
                model_name = model_name.replace(suffix, '')

            if not model_name:
                raise ValueError(f"Empty model name detected for {model_path}")

            return model_name

        except Exception as e:
            raise ValueError(f"Failed to detect model family from {model_path}: {e}")

    def extract_model_family(self, model_path_or_name: str) -> str:
        """Extract model family from any model path, directory name, or model identifier.

        This function expects models to follow the pattern: {model_family}_massive_k_{locale}
        For example: "xlm-roberta-base_massive_k_af-ZA" -> "xlm-roberta-base"

        Args:
            model_path_or_name: Can be a full path, directory name, or model identifier

        Returns:
            The model family (e.g., "xlm-roberta-base", "bert-large-uncased")

        Raises:
            ValueError: If model family cannot be determined
        """
        if not model_path_or_name:
            raise ValueError("model_path_or_name cannot be empty")

        # Extract just the directory/file name if it's a full path
        name = model_path_or_name
        if '/' in name:
            name = name.split('/')[-1]

        # Remove common prefixes/suffixes that might interfere
        name = name.strip()

        # Pattern 1: Extract from directory names with massive_k pattern
        # e.g., "xlm-roberta-base_massive_k_af-ZA" -> "xlm-roberta-base"
        if "_massive_k_" in name:
            model_family = name.split("_massive_k_")[0]
            if model_family:
                return model_family

        # Pattern 2: Handle merged model directories
        # e.g., "ties_REAL_merge_af-ZA_xlm-roberta-base_4merged" -> "xlm-roberta-base"
        merged_pattern = r'^[a-zA-Z0-9_]+_[A-Z]+_merge_[a-z]{2}-[A-Z]{2}_([a-zA-Z0-9\-]+)_\d+merged$'
        match = re.match(merged_pattern, name)
        if match:
            return match.group(1)

        # Pattern 3: Fallback - just return the name as-is if it doesn't contain massive_k
        # This handles cases where the input is already a model family name
        if len(name) > 3:
            return name

        raise ValueError(f"Cannot determine model family from: {model_path_or_name}")

    def _strip_massive_suffix(self, model_family: Optional[str]) -> Optional[str]:
        """Remove trailing _massive_k suffixes (with optional locale) from model family names."""
        if not model_family:
            return model_family
        return re.sub(r'_massive_k(?:_[a-z]{2}-[A-Z]{2})?$', '', model_family)

    def parse_results_dir_name(self, dir_name: str) -> Dict[str, Any]:
        """Parse results directory name into components."""
        # Baseline pattern: baseline_sq-AL_xlm-roberta-base-uncased
        baseline_pattern = r'^baseline_(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)$'
        match = re.match(baseline_pattern, dir_name)
        if match:
            result = match.groupdict()
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            result['experiment_type'] = 'baseline'
            result['method'] = 'baseline'
            result['similarity_type'] = None
            result['num_languages'] = None
            result['timestamp'] = None
            return result

        # General pattern: merging_similarity_REAL_sq-AL_xlm-roberta-base-uncased_5lang_IT/ET
        # Pattern: experiment_type_method_similarity_type_locale_model_family_num_languages_include_target
        # Handle both model_family with and without _massive_k suffix
        massive_suffix_pattern = r'(?:_massive_k(?:_[a-z]{2}-[A-Z]{2})?)'
        general_pattern = (
            r'^(?P<experiment_type>[^_]+)_(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)lang_'
            r'(?P<include_target>IncTar|ExcTar|IT|ET)$'
        )
        match = re.match(general_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            result['timestamp'] = None
            # Remove the optional _massive_k group from the result
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            # Normalize include_target values
            result['include_target'] = self.normalize_include_target(result['include_target'])
            return result

        # Alternative pattern: ensemble_method_locale_model_family_num_languages_include_target
        ensemble_pattern = (
            r'^(?P<experiment_type>ensemble)_(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)lang_'
            r'(?P<include_target>IncTar|ExcTar|IT|ET)$'
        )
        match = re.match(ensemble_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            result['timestamp'] = None
            # Remove the optional _massive_k group from the result
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            # Normalize include_target values
            result['include_target'] = self.normalize_include_target(result['include_target'])
            return result

        # Alternative pattern: iterative_method_locale_model_family_num_languages_include_target
        iterative_pattern = (
            r'^(?P<experiment_type>iterative)_(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)lang_'
            r'(?P<include_target>IncTar|ExcTar|IT|ET)$'
        )
        match = re.match(iterative_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            result['timestamp'] = None
            # Remove the optional _massive_k group from the result
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            # Normalize include_target values
            result['include_target'] = self.normalize_include_target(result['include_target'])
            return result

        # Legacy pattern without IT/ET: method_num_languages_locale (no similarity type, no timestamp)
        legacy_pattern = r'^(?P<method>[^_]+)_(?P<num_languages>\d+)lang_(?P<locale>[a-z]{2}-[A-Z]{2})$'
        match = re.match(legacy_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['experiment_type'] = 'merging'
            result['similarity_type'] = 'URIEL'  # Default for legacy
            result['model_family'] = 'unknown'
            result['num_languages'] = int(result['num_languages'])
            result['include_target'] = 'ET'  # Default to ET for backward compatibility
            result['timestamp'] = None
            return result

        # New pattern without IT/ET for backward compatibility: experiment_type_method_similarity_type_locale_model_family_num_languages
        general_pattern_legacy = (
            r'^(?P<experiment_type>[^_]+)_(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)lang$'
        )
        match = re.match(general_pattern_legacy, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            result['include_target'] = 'ET'  # Default to ET for backward compatibility
            result['timestamp'] = None
            # Remove the optional _massive_k group from the result
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            return result

        # Pattern with timestamp (for backward compatibility)
        timestamp_pattern = r'^(?P<method>[^_]+)_(?P<similarity_type>URIEL|REAL)_(?P<experiment_type>[^_]+)_(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)_(?P<num_languages>\d+)lang_(?P<timestamp>\d{8}_\d{6})$'
        match = re.match(timestamp_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            return result

        # Baseline pattern with timestamp (for backward compatibility)
        baseline_timestamp_pattern = r'^baseline_(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)_(?P<timestamp>\d{8}_\d{6})$'
        match = re.match(baseline_timestamp_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['experiment_type'] = 'baseline'
            result['method'] = 'baseline'
            result['similarity_type'] = None
            result['num_languages'] = None
            return result

        raise ValueError(f"Cannot parse directory name: {dir_name}")

    def parse_merged_model_dir_name(self, dir_name: str) -> Dict[str, Any]:
        """Parse merged model directory name into components."""
        # New pattern: ties_REAL_merge_af-ZA_xlm-roberta-base_4merged_IT/ET or average_REAL_merge_af-ZA_xlm-roberta-large_4merged_IT/ET
        massive_suffix_pattern = r'(?:_massive_k(?:_[a-z]{2}-[A-Z]{2})?)'
        pattern = (
            r'^(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_merge_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)merged_'
            r'(?P<include_target>IncTar|ExcTar|IT|ET)$'
        )
        match = re.match(pattern, dir_name)

        if match:
            result = match.groupdict()
            result['experiment_type'] = 'merging'
            result['num_languages'] = int(result['num_languages'])
            result['timestamp'] = None
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            # Normalize include_target values
            result['include_target'] = self.normalize_include_target(result['include_target'])
            return result

        # Legacy pattern without IT/ET for backward compatibility: ties_REAL_merge_af-ZA_xlm-roberta-base_4merged
        pattern_legacy = (
            r'^(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_merge_'
            r'(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)'
            rf'(?P<_massive_k>{massive_suffix_pattern})?_(?P<num_languages>\d+)merged$'
        )
        match = re.match(pattern_legacy, dir_name)

        if match:
            result = match.groupdict()
            result['experiment_type'] = 'merging'
            result['num_languages'] = int(result['num_languages'])
            result['include_target'] = 'ET'  # Default to ET for backward compatibility
            result['timestamp'] = None
            if '_massive_k' in result:
                del result['_massive_k']
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            return result

        # Legacy pattern for backward compatibility (without model_family)
        legacy_pattern = r'^(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_merge_(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<num_languages>\d+)merged$'
        match = re.match(legacy_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['experiment_type'] = 'merging'
            result['model_family'] = 'unknown'  # Legacy models don't have family info
            result['num_languages'] = int(result['num_languages'])
            result['include_target'] = 'ET'  # Default to ET for backward compatibility
            result['timestamp'] = None
            return result

        # Older legacy pattern for backward compatibility
        older_legacy_pattern = r'^(?P<experiment_type>[^_]+)_(?P<method>[^_]+(?:_[^_]+)*)_(?P<similarity_type>URIEL|REAL)_(?P<locale>[a-z]{2}-[A-Z]{2})_(?P<model_family>[^_]+(?:_[^_]+)*)_(?P<num_languages>\d+)lang_(?P<timestamp>\d{8}_\d{6})$'
        match = re.match(older_legacy_pattern, dir_name)
        if match:
            result = match.groupdict()
            result['num_languages'] = int(result['num_languages'])
            if 'model_family' in result:
                result['model_family'] = self._strip_massive_suffix(result['model_family'])
            return result

        raise ValueError(f"Cannot parse merged model directory name: {dir_name}")

    def normalize_include_target(self, include_target: str) -> str:
        """Normalize include_target values to use IncTar/ExcTar format"""
        if include_target in ['IT', 'IncTar']:
            return 'IncTar'
        elif include_target in ['ET', 'ExcTar']:
            return 'ExcTar'
        else:
            # Default to ExcTar for unknown values
            return 'ExcTar'

    def find_results_directory(self, base_dir: str, experiment_type: str, method: str,
                             similarity_type: str, locale: str, model_family: str,
                             num_languages: Optional[int] = None,
                             include_target: Optional[bool] = None) -> Optional[str]:
        """Find existing results directory matching given criteria."""
        # Try multiple timestamps (most recent first)
        include_target_suffix = None
        if include_target is not None:
            include_target_suffix = "IncTar" if include_target else "ExcTar"

        # Normalize provided model family to be tolerant of massive_k suffix
        normalized_model_family = model_family
        try:
            normalized_model_family = self.extract_model_family(model_family)
        except Exception:
            normalized_model_family = self._strip_massive_suffix(model_family)

        for entry in sorted(os.listdir(base_dir), reverse=True):
            if not os.path.isdir(os.path.join(base_dir, entry)):
                continue

            try:
                parsed = self.parse_results_dir_name(entry)
                if (parsed['experiment_type'] == experiment_type and
                    parsed['method'] == method and
                    parsed['similarity_type'] == similarity_type and
                    parsed['locale'] == locale and
                    (parsed['model_family'] == model_family or parsed['model_family'] == normalized_model_family) and
                    (include_target_suffix is None or parsed.get('include_target') == include_target_suffix)):
                    if num_languages is not None:
                        if parsed['num_languages'] == num_languages:
                            return os.path.join(base_dir, entry)
                    else:
                        return os.path.join(base_dir, entry)
            except ValueError:
                continue

        return None

    def find_merged_model_directory(self, base_dir: str, method: str, similarity_type: str,
                                  locale: str, model_family: str, num_languages: Optional[int] = None,
                                  include_target: Optional[bool] = None) -> Optional[str]:
        """Find existing merged model directory matching given criteria."""
        if not os.path.exists(base_dir):
            return None

        include_target_suffix = None
        if include_target is not None:
            include_target_suffix = "IncTar" if include_target else "ExcTar"

        normalized_model_family = model_family
        try:
            normalized_model_family = self.extract_model_family(model_family)
        except Exception:
            normalized_model_family = self._strip_massive_suffix(model_family)

        for entry in sorted(os.listdir(base_dir), reverse=True):
            if not os.path.isdir(os.path.join(base_dir, entry)):
                continue

            try:
                parsed = self.parse_merged_model_dir_name(entry)
                if (parsed['method'] == method and
                    parsed['similarity_type'] == similarity_type and
                    parsed['locale'] == locale and
                    (parsed.get('model_family') == model_family or
                     parsed.get('model_family') == normalized_model_family or
                     parsed.get('model_family') == 'unknown') and
                    (include_target_suffix is None or parsed.get('include_target') == include_target_suffix)):
                    # For merged models, be more flexible about num_languages
                    # The directory name might not match the actual number of merged models
                    return os.path.join(base_dir, entry)
            except ValueError:
                continue

        return None

    def validate_similarity_type(self, similarity_type: str) -> None:
        """Validate similarity type."""
        if similarity_type not in self.config.similarity_types:
            raise ValueError(f"Invalid similarity type: {similarity_type}. Must be one of: {self.config.similarity_types}")

    def validate_required_components(self, experiment_type: str, method: str, similarity_type: str,
                                   locale: str, model_family: str, model_path: str) -> None:
        """Validate all required components are present."""
        if not experiment_type:
            raise ValueError("experiment_type is required")
        if not method:
            raise ValueError("method is required")
        if not similarity_type and experiment_type != 'baseline':
            raise ValueError("similarity_type is required for non-baseline experiments")
        if not locale:
            raise ValueError("locale is required")
        if not model_family:
            raise ValueError("model_family is required")
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path is required and must exist: {model_path}")

        # Validate locale format
        if not re.match(r'^[a-z]{2}-[A-Z]{2}$', locale):
            raise ValueError(f"Invalid locale format: {locale}. Expected format: xx-XX")

        # Validate similarity type
        if similarity_type:
            self.validate_similarity_type(similarity_type)

# Global instance for easy access
naming_manager = NamingManager()
