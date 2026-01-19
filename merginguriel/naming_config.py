"""
Centralized naming configuration for experiment directories.

Canonical naming formats (ONLY these are supported):
- Results:  {experiment_type}_{method}_{similarity_type}_{locale}_{model_family}_{num_languages}lang_{IncTar|ExcTar}
- Merged:   {method}_{similarity_type}_merge_{locale}_{model_family}_{num_merged}merged_{IncTar|ExcTar}
- Baseline: baseline_{locale}_{model_family}

Legacy formats are NOT supported. Regenerate old results if needed.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import re
import os


# =============================================================================
# NAMING COMPONENTS
# =============================================================================

@dataclass
class NamingComponents:
    """Components that make up experiment directory names.

    Add new fields here to extend the naming system. All fields are optional
    except those required for specific directory types.
    """
    experiment_type: Optional[str] = None  # merging, ensemble, iterative, baseline
    method: Optional[str] = None           # similarity, average, fisher, ties, etc.
    similarity_type: Optional[str] = None  # URIEL, REAL
    locale: Optional[str] = None           # sq-AL, en-US, etc.
    model_family: Optional[str] = None     # xlm-roberta-base, etc.
    num_languages: Optional[int] = None    # number of source languages
    include_target: str = "ExcTar"         # IncTar or ExcTar

    def __post_init__(self):
        # Normalize include_target
        if self.include_target in ("IT", "IncTar", "include", True):
            self.include_target = "IncTar"
        elif self.include_target in ("ET", "ExcTar", "exclude", False):
            self.include_target = "ExcTar"

        # Validate similarity_type if provided
        if self.similarity_type is not None and self.similarity_type not in ("URIEL", "REAL"):
            raise ValueError(f"similarity_type must be 'URIEL' or 'REAL', got '{self.similarity_type}'")

        # Validate locale format if provided
        if self.locale is not None and not re.match(r'^[a-z]{2}-[A-Z]{2}$', self.locale):
            raise ValueError(f"locale must be in format 'xx-XX', got '{self.locale}'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CANONICAL PATTERNS
# =============================================================================

# Regex building blocks
_LOCALE = r'(?P<locale>[a-z]{2}-[A-Z]{2})'
_SIMILARITY = r'(?P<similarity_type>URIEL|REAL)'
_INCLUDE_TARGET = r'(?P<include_target>IncTar|ExcTar)'
_MODEL_FAMILY = r'(?P<model_family>[a-zA-Z0-9\-]+)'
_NUM_LANG = r'(?P<num_languages>\d+)'

# Results directory pattern
# Example: merging_similarity_URIEL_sq-AL_xlm-roberta-base_3lang_ExcTar
_RESULTS_PATTERN = re.compile(
    rf'^(?P<experiment_type>\w+)_(?P<method>[\w]+)_{_SIMILARITY}_{_LOCALE}_{_MODEL_FAMILY}_{_NUM_LANG}lang_{_INCLUDE_TARGET}$'
)

# Baseline results pattern
# Example: baseline_sq-AL_xlm-roberta-base
_BASELINE_PATTERN = re.compile(
    rf'^baseline_{_LOCALE}_{_MODEL_FAMILY}$'
)

# Merged model directory pattern
# Example: similarity_URIEL_merge_sq-AL_xlm-roberta-base_2merged_ExcTar
_MERGED_PATTERN = re.compile(
    rf'^(?P<method>[\w]+)_{_SIMILARITY}_merge_{_LOCALE}_{_MODEL_FAMILY}_{_NUM_LANG}merged_{_INCLUDE_TARGET}$'
)


# =============================================================================
# NAMING MANAGER
# =============================================================================

@dataclass
class NamingConfig:
    """Configuration for naming conventions."""
    model_pattern: str = "xlm-roberta-{size}_massive_k_{locale}"
    default_model_size: str = "base"


class NamingManager:
    """Manages experiment naming conventions."""

    def __init__(self, config: Optional[NamingConfig] = None):
        self.config = config or NamingConfig()

    # -------------------------------------------------------------------------
    # Name Generation
    # -------------------------------------------------------------------------

    def get_results_dir_name(
        self,
        experiment_type: str,
        method: str,
        similarity_type: str,
        locale: str,
        model_family: str,
        num_languages: int,
        include_target: bool = False,
    ) -> str:
        """Generate canonical results directory name."""
        components = NamingComponents(
            experiment_type=experiment_type,
            method=method,
            similarity_type=similarity_type,
            locale=locale,
            model_family=self._clean_model_family(model_family),
            num_languages=num_languages,
            include_target="IncTar" if include_target else "ExcTar",
        )
        return (
            f"{components.experiment_type}_{components.method}_{components.similarity_type}_"
            f"{components.locale}_{components.model_family}_{components.num_languages}lang_"
            f"{components.include_target}"
        )

    def get_merged_model_dir_name(
        self,
        experiment_type: str,  # kept for API compatibility, not used in name
        method: str,
        similarity_type: str,
        locale: str,
        model_family: str,
        num_merged: int,
        include_target: bool = False,
    ) -> str:
        """Generate canonical merged model directory name."""
        components = NamingComponents(
            method=method,
            similarity_type=similarity_type,
            locale=locale,
            model_family=self._clean_model_family(model_family),
            num_languages=num_merged,
            include_target="IncTar" if include_target else "ExcTar",
        )
        return (
            f"{components.method}_{components.similarity_type}_merge_"
            f"{components.locale}_{components.model_family}_{components.num_languages}merged_"
            f"{components.include_target}"
        )

    def get_baseline_dir_name(self, locale: str, model_family: str) -> str:
        """Generate baseline results directory name."""
        return f"baseline_{locale}_{self._clean_model_family(model_family)}"

    def get_model_path(self, models_root: str, model_size: str, locale: str) -> str:
        """Get the model path for a specific locale."""
        model_dir = self.config.model_pattern.format(size=model_size, locale=locale)
        return os.path.join(models_root, model_dir)

    def get_plot_filename(
        self,
        method: str,
        model_family: str,
        similarity_type: str,
        plot_type: str = "comparison",
        num_languages: Optional[int] = None,
    ) -> str:
        """Generate plot filename."""
        clean_family = self._clean_model_family(model_family)
        if num_languages:
            return f"{plot_type}_{method}_{similarity_type}_{clean_family}_{num_languages}lang.png"
        return f"{plot_type}_{method}_{similarity_type}_{clean_family}.png"

    # -------------------------------------------------------------------------
    # Name Parsing
    # -------------------------------------------------------------------------

    def parse_results_dir_name(self, dir_name: str) -> Dict[str, Any]:
        """Parse a results directory name into components."""
        # Try baseline first (simpler pattern)
        match = _BASELINE_PATTERN.match(dir_name)
        if match:
            return {
                'experiment_type': 'baseline',
                'method': 'baseline',
                'similarity_type': None,
                'locale': match.group('locale'),
                'model_family': match.group('model_family'),
                'num_languages': None,
                'include_target': None,
            }

        # Try full results pattern
        match = _RESULTS_PATTERN.match(dir_name)
        if match:
            return {
                'experiment_type': match.group('experiment_type'),
                'method': match.group('method'),
                'similarity_type': match.group('similarity_type'),
                'locale': match.group('locale'),
                'model_family': match.group('model_family'),
                'num_languages': int(match.group('num_languages')),
                'include_target': match.group('include_target'),
            }

        raise ValueError(f"Cannot parse results directory name: {dir_name}")

    def parse_merged_model_dir_name(self, dir_name: str) -> Dict[str, Any]:
        """Parse a merged model directory name into components."""
        match = _MERGED_PATTERN.match(dir_name)
        if match:
            return {
                'method': match.group('method'),
                'similarity_type': match.group('similarity_type'),
                'locale': match.group('locale'),
                'model_family': match.group('model_family'),
                'num_languages': int(match.group('num_languages')),
                'include_target': match.group('include_target'),
            }

        raise ValueError(f"Cannot parse merged model directory name: {dir_name}")

    # -------------------------------------------------------------------------
    # Detection & Extraction
    # -------------------------------------------------------------------------

    def detect_model_size_from_root(self, models_root: str) -> str:
        """Detect model size (base/large) from the models root directory name."""
        root_lower = models_root.lower()
        if "large" in root_lower:
            return "large"
        return "base"

    def detect_model_family_from_path(self, model_path: str) -> str:
        """Extract model family from a model path."""
        basename = os.path.basename(model_path.rstrip('/'))
        return self._clean_model_family(basename)

    def extract_model_family(self, model_path_or_name: str) -> str:
        """Extract model family from a model path or name.

        Alias for detect_model_family_from_path for backward compatibility.
        """
        return self.detect_model_family_from_path(model_path_or_name)

    def extract_locale_from_path(self, model_path: str) -> Optional[str]:
        """Extract locale from a model path like xlm-roberta-base_massive_k_sq-AL."""
        match = re.search(r'_([a-z]{2}-[A-Z]{2})$', model_path)
        return match.group(1) if match else None

    # -------------------------------------------------------------------------
    # Finding Directories
    # -------------------------------------------------------------------------

    def find_results_directory(
        self,
        base_dir: str,
        experiment_type: str,
        method: str,
        similarity_type: str,
        locale: str,
        model_family: str,
        num_languages: Optional[int] = None,
        include_target: Optional[bool] = None,
    ) -> Optional[str]:
        """Find an existing results directory matching the criteria."""
        if not os.path.exists(base_dir):
            return None

        # Generate expected name
        expected = self.get_results_dir_name(
            experiment_type=experiment_type,
            method=method,
            similarity_type=similarity_type,
            locale=locale,
            model_family=model_family,
            num_languages=num_languages or 0,
            include_target=include_target or False,
        )

        full_path = os.path.join(base_dir, expected)
        if os.path.exists(full_path):
            return full_path

        return None

    def find_merged_model_directory(
        self,
        base_dir: str,
        method: str,
        similarity_type: str,
        locale: str,
        model_family: str,
        num_merged: Optional[int] = None,
        include_target: Optional[bool] = None,
    ) -> Optional[str]:
        """Find an existing merged model directory matching the criteria."""
        if not os.path.exists(base_dir):
            return None

        # Generate expected name
        expected = self.get_merged_model_dir_name(
            experiment_type="merging",  # Not used in name
            method=method,
            similarity_type=similarity_type,
            locale=locale,
            model_family=model_family,
            num_merged=num_merged or 0,
            include_target=include_target or False,
        )

        full_path = os.path.join(base_dir, expected)
        if os.path.exists(full_path):
            return full_path

        return None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_similarity_type(self, similarity_type: str) -> None:
        """Validate that similarity_type is valid."""
        if similarity_type not in ("URIEL", "REAL"):
            raise ValueError(f"similarity_type must be 'URIEL' or 'REAL', got '{similarity_type}'")

    def validate_required_components(
        self,
        experiment_type: str,
        method: str,
        similarity_type: str,
        locale: str,
        model_family: str,
        model_path: Optional[str] = None,
    ) -> None:
        """Validate that all required naming components are present."""
        if not experiment_type:
            raise ValueError("experiment_type is required")
        if not method:
            raise ValueError("method is required")
        if not similarity_type:
            raise ValueError("similarity_type is required")
        if not locale:
            raise ValueError("locale is required")
        if not model_family:
            raise ValueError("model_family is required")

        self.validate_similarity_type(similarity_type)

        # Validate locale format
        if not re.match(r'^[a-z]{2}-[A-Z]{2}$', locale):
            raise ValueError(f"locale must be in format 'xx-XX', got '{locale}'")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _clean_model_family(self, model_family: str) -> str:
        """Remove locale suffix and massive_k suffix from model family."""
        if not model_family:
            return model_family

        # Remove _massive_k_xx-XX suffix
        cleaned = re.sub(r'_massive_k_[a-z]{2}-[A-Z]{2}$', '', model_family)
        # Remove _massive_k suffix
        cleaned = re.sub(r'_massive_k$', '', cleaned)

        return cleaned

    def normalize_include_target(self, include_target: Any) -> str:
        """Normalize include_target to canonical form."""
        if include_target in ("IT", "IncTar", "include", True):
            return "IncTar"
        return "ExcTar"


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

naming_manager = NamingManager()
