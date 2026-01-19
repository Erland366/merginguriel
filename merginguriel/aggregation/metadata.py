"""
Data classes and parsing utilities for experiment metadata.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from merginguriel import logger
from merginguriel.naming_config import naming_manager


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment extracted from merge_details.txt or folder name."""

    experiment_type: str
    locale: str
    target_lang: Optional[str] = None
    source_languages: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    merge_mode: Optional[str] = None
    similarity_type: Optional[str] = None
    num_languages: Optional[int] = None
    include_target: Optional[str] = None
    timestamp: Optional[str] = None
    folder_name: Optional[str] = None


@dataclass
class BaselineData:
    """Baseline performance data from N-vs-N evaluation."""

    best_source_accuracy: Optional[float] = None
    best_source_language: Optional[str] = None
    best_overall_accuracy: Optional[float] = None
    best_overall_language: Optional[str] = None


def is_merge_model_path(model_path: Optional[str]) -> bool:
    """Return True if the given model path corresponds to a merged model."""
    return bool(model_path and "merged_models" in model_path)


def parse_num_languages_from_model_path(model_path: Optional[str]) -> Optional[int]:
    """Extract the merged language count from a merged model path if encoded."""
    if not model_path:
        return None

    base_name = os.path.basename(os.path.normpath(model_path))
    match = re.search(r"_(\d+)merged$", base_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.debug(f"Failed to parse num_languages from {base_name}")
    return None


def count_languages_in_merge_details(model_path: Optional[str]) -> Optional[int]:
    """Count languages listed in merge_details.txt for a merged model."""
    if not model_path:
        return None

    merge_details_path = os.path.join(model_path, "merge_details.txt")
    if not os.path.exists(merge_details_path):
        return None

    try:
        with open(merge_details_path, "r") as f:
            content = f.read()
        matches = re.findall(r"^\s*\d+\.\s*Model:", content, re.MULTILINE)
        if matches:
            return len(matches)
    except Exception as e:
        logger.warning(f"Error counting languages in {merge_details_path}: {e}")
    return None


def extract_locale_from_model_path(model_path: str) -> Optional[str]:
    """Extract locale from model path like './haryos_model/xlm-roberta-base_massive_k_sq-AL'."""
    match = re.search(r"massive_k_([a-z]{2}-[A-Z]{2})", model_path)
    if match:
        return match.group(1)
    return None


def _normalize_arch_name(name: str) -> str:
    """Normalize architecture string to match model_family style."""
    return name.replace("_", "-")


def determine_experiment_variant(
    experiment_type: str,
    num_languages: Optional[int],
    model_path: Optional[str],
    similarity_type: Optional[str] = None,
    base_model: Optional[str] = None,
    include_target: Optional[str] = None,
) -> str:
    """Build a display key that differentiates merges by language count, similarity type, and base model."""
    base_type = experiment_type or "unknown"

    if not is_merge_model_path(model_path):
        if base_model:
            return f"{base_type}_{base_model}"
        return base_type

    model_name = base_model if base_model else "unknown_model"

    if similarity_type and similarity_type in ["URIEL", "REAL"] and base_type != "baseline":
        if num_languages:
            base_name = f"{base_type}_{similarity_type}_{model_name}_{int(num_languages)}lang"
        else:
            base_name = f"{base_type}_{similarity_type}_{model_name}_unknownlang"
    else:
        if num_languages:
            base_name = f"{base_type}_{model_name}_{int(num_languages)}lang"
        else:
            base_name = f"{base_type}_{model_name}_unknownlang"

    if include_target:
        base_name = f"{base_name}_{include_target}"

    return base_name


def _parse_merge_details_file(merge_details_path: str) -> Optional[Dict[str, Any]]:
    """Parse merge_details.txt file to extract key-value pairs."""
    if not os.path.exists(merge_details_path):
        return None

    try:
        with open(merge_details_path, "r") as f:
            content = f.read()

        details = {}
        for line in content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                details[key.strip()] = value.strip()

        return details
    except Exception as e:
        logger.warning(f"Error parsing merge details from {merge_details_path}: {e}")
        return None


def _parse_metadata_from_merge_details(
    merge_details: Dict[str, Any],
    details_content: str,
    folder_name: str,
) -> ExperimentMetadata:
    """Parse experiment metadata from merge_details dictionary and raw content."""
    experiment_type = merge_details.get("Merge Mode", "unknown").lower()
    target_lang = merge_details.get("Target Language", "") or merge_details.get("Locale", "")

    locale_pattern = re.compile(r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2})", re.MULTILINE)
    weight_pattern = re.compile(
        r"^\s*- Locale:\s*([a-zA-Z]{2}-[a-zA-Z]{2}).*?Weight:\s*([0-9.]+)",
        re.MULTILINE | re.DOTALL,
    )

    source_languages = locale_pattern.findall(details_content)
    weights = {locale: float(weight) for locale, weight in weight_pattern.findall(details_content)}
    num_languages = len(source_languages) if source_languages else None

    return ExperimentMetadata(
        experiment_type=experiment_type,
        locale=target_lang,
        target_lang=target_lang,
        source_languages=source_languages or None,
        weights=weights or None,
        merge_mode=experiment_type,
        similarity_type=None,
        num_languages=num_languages,
        timestamp=merge_details.get("Timestamp (UTC)", ""),
        folder_name=folder_name,
    )


def _parse_metadata_from_naming_system(folder_name: str) -> Optional[ExperimentMetadata]:
    """Parse metadata using centralized naming system."""
    try:
        parsed = naming_manager.parse_results_dir_name(folder_name)
        return ExperimentMetadata(
            experiment_type=parsed["experiment_type"],
            locale=parsed["locale"],
            target_lang=parsed["locale"],
            source_languages=None,
            weights=None,
            merge_mode=parsed["method"],
            similarity_type=parsed.get("similarity_type", "URIEL"),
            num_languages=parsed["num_languages"],
            include_target=parsed.get("include_target"),
            timestamp=parsed.get("timestamp"),
            folder_name=folder_name,
        )
    except ValueError:
        pass

    try:
        parsed = naming_manager.parse_merged_model_dir_name(folder_name)
        return ExperimentMetadata(
            experiment_type=parsed["experiment_type"],
            locale=parsed["locale"],
            target_lang=parsed["locale"],
            source_languages=None,
            weights=None,
            merge_mode=parsed["method"],
            similarity_type=parsed.get("similarity_type", "URIEL"),
            num_languages=parsed["num_languages"],
            include_target=parsed.get("include_target"),
            timestamp=parsed.get("timestamp"),
            folder_name=folder_name,
        )
    except ValueError:
        pass

    return None


def _parse_metadata_from_model_path(
    model_path: str, folder_name: str
) -> ExperimentMetadata:
    """Fallback metadata parsing using model path and folder name heuristics."""
    experiment_type = "baseline" if not is_merge_model_path(model_path) else "unknown"
    locale = None
    num_languages = None

    if model_path:
        base_name = os.path.basename(os.path.normpath(model_path))

        merge_match = re.match(
            r"([a-zA-Z0-9]+)_(?:URIEL|REAL)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?",
            base_name,
        )
        if not merge_match:
            merge_match = re.match(
                r"([a-zA-Z0-9]+)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?",
                base_name,
            )
        if merge_match:
            experiment_type = merge_match.group(1)
            locale = merge_match.group(2)
            if merge_match.group(3):
                try:
                    num_languages = int(merge_match.group(3))
                except ValueError:
                    num_languages = None
        else:
            extracted_locale = extract_locale_from_model_path(model_path)
            if extracted_locale:
                locale = extracted_locale

    if not locale:
        locale, experiment_type, num_languages = _parse_locale_from_folder_name(
            folder_name, experiment_type, num_languages
        )

    return ExperimentMetadata(
        experiment_type=experiment_type,
        locale=locale,
        target_lang=locale,
        source_languages=None,
        weights=None,
        merge_mode=experiment_type,
        num_languages=num_languages,
        timestamp=None,
        folder_name=folder_name,
    )


def _parse_locale_from_folder_name(
    folder_name: str,
    experiment_type: str,
    num_languages: Optional[int],
) -> tuple[Optional[str], str, Optional[int]]:
    """Extract locale from folder name using various naming patterns."""
    locale = None

    patterns = [
        r"([a-zA-Z0-9]+)_(?:URIEL|REAL)_(\d+)lang_(.+)",
        r"([a-zA-Z0-9]+)_(?:URIEL|REAL)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?",
        r"(.+?)_(\d+)lang_(.+)",
        r"([a-zA-Z0-9]+)_merge_([a-z]{2}-[A-Z]{2})(?:_(\d+)merged)?",
    ]

    for pattern in patterns:
        num_lang_match = re.match(pattern, folder_name)
        if num_lang_match:
            experiment_type = num_lang_match.group(1)
            if "merge" in folder_name:
                locale = num_lang_match.group(2)
                if num_lang_match.group(3):
                    try:
                        num_languages = int(num_lang_match.group(3))
                    except ValueError:
                        num_languages = None
            else:
                try:
                    num_languages = int(num_lang_match.group(2))
                except ValueError:
                    num_languages = None
                locale = num_lang_match.group(3)
            return locale, experiment_type, num_languages

    if "_" in folder_name:
        parts = folder_name.split("_")
        if len(parts[-1]) == 5 and "-" in parts[-1]:
            locale = parts[-1]
            experiment_type_parts = [p for p in parts[:-1] if p not in ["URIEL", "REAL"]]
            if experiment_type_parts:
                experiment_type = "_".join(experiment_type_parts)
        else:
            locale = parts[-1]
            experiment_type_parts = [p for p in parts[:-1] if p not in ["URIEL", "REAL"]]
            if experiment_type_parts:
                experiment_type = "_".join(experiment_type_parts)
    else:
        locale = folder_name
        experiment_type = folder_name

    return locale, experiment_type, num_languages


def parse_experiment_metadata(
    folder_name: str,
    folder_path: str,
    model_path: Optional[str] = None,
) -> ExperimentMetadata:
    """Parse experiment metadata using merge_details if present, otherwise use centralized naming system."""
    merge_details_path = None
    if model_path and os.path.isdir(model_path):
        merge_details_path = os.path.join(model_path, "merge_details.txt")
    else:
        merge_details_path = os.path.join(folder_path, "merge_details.txt")

    merge_details = None
    details_content = None
    if merge_details_path and os.path.exists(merge_details_path):
        merge_details = _parse_merge_details_file(merge_details_path)
        try:
            with open(merge_details_path, "r") as f:
                details_content = f.read()
        except Exception as e:
            logger.warning(f"Unable to read merge details content from {merge_details_path}: {e}")
            details_content = None

    if merge_details and details_content:
        return _parse_metadata_from_merge_details(merge_details, details_content, folder_name)

    if not merge_details:
        naming_result = _parse_metadata_from_naming_system(folder_name)
        if naming_result:
            return naming_result

    return _parse_metadata_from_model_path(model_path or "", folder_name)
