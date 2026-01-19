"""
Data loading functions for experiment results and evaluation matrices.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from merginguriel import logger
from merginguriel.naming_config import naming_manager
from merginguriel.aggregation.metadata import (
    BaselineData,
    ExperimentMetadata,
    count_languages_in_merge_details,
    determine_experiment_variant,
    is_merge_model_path,
    parse_experiment_metadata,
    parse_num_languages_from_model_path,
    _normalize_arch_name,
)


def load_results_from_folder(folder_path: str) -> Optional[Dict[str, Any]]:
    """Load results.json from a folder."""
    results_file = os.path.join(folder_path, "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {results_file}: {e}")
            return None
    return None


def parse_merge_details(merge_details_path: str) -> Optional[Dict[str, Any]]:
    """Parse merge_details.txt file to extract experiment metadata."""
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


def get_experiment_folders(results_dir: str = "results") -> List[str]:
    """Get all experiment folders from the results directory."""
    if not results_dir:
        results_dir = "results"
    if not os.path.exists(results_dir):
        return []

    folders = []
    for item in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, item)
        if os.path.isdir(folder_path):
            folders.append(item)

    return sorted(folders)


def extract_accuracy(results: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract accuracy from results dictionary."""
    if results and "performance" in results:
        return results["performance"]["accuracy"]
    return None


def find_evaluation_matrices(nxn_results_dir: str = "nxn_results") -> Dict[str, str]:
    """Find evaluation matrices keyed by model family.

    Looks for subdirectories under nxn_results named for the architecture
    (e.g., xlm_roberta_base, xlm-roberta-large) containing evaluation_matrix*.csv.
    """
    matrices: Dict[str, str] = {}
    if not os.path.exists(nxn_results_dir):
        logger.warning(f"N-x-N results directory not found: {nxn_results_dir}")
        return matrices

    for sub in Path(nxn_results_dir).iterdir():
        if not sub.is_dir():
            continue
        if sub.name.startswith("nxn_eval_"):
            continue
        arch = _normalize_arch_name(sub.name)
        files = sorted(sub.glob("evaluation_matrix*.csv"), key=os.path.getmtime, reverse=True)
        if files:
            matrices[arch] = str(files[0])

    if matrices:
        for fam, path in matrices.items():
            logger.info(f"Using evaluation matrix for {fam}: {path}")
        return matrices

    logger.warning("No evaluation_matrix found under architecture folders in nxn_results/")
    return matrices


def load_evaluation_matrix(matrix_path: str) -> Optional[pd.DataFrame]:
    """Load the evaluation matrix from CSV file."""
    try:
        df = pd.read_csv(matrix_path, index_col=0)
        logger.info(f"Loaded evaluation matrix with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading evaluation matrix from {matrix_path}: {e}")
        return None


def get_baseline_for_target(
    target_locale: str,
    source_locales: List[str],
    evaluation_matrix: pd.DataFrame,
) -> BaselineData:
    """Calculate baseline data for a target locale given source locales."""
    baseline = BaselineData()

    if evaluation_matrix is None or target_locale not in evaluation_matrix.index:
        logger.warning(f"Target locale {target_locale} not found in evaluation matrix")
        return baseline

    source_performances = {}
    for source_locale in source_locales:
        if source_locale in evaluation_matrix.index and target_locale in evaluation_matrix.columns:
            accuracy = evaluation_matrix.loc[source_locale, target_locale]
            if pd.notna(accuracy):
                source_performances[source_locale] = accuracy

    if source_performances:
        best_source_locale = max(source_performances, key=source_performances.get)
        baseline.best_source_accuracy = source_performances[best_source_locale]
        baseline.best_source_language = best_source_locale

    target_column = evaluation_matrix[target_locale] if target_locale in evaluation_matrix.columns else None
    if target_column is not None:
        overall_performances = target_column.drop(target_locale, errors="ignore")
        overall_performances = overall_performances[overall_performances.notna()]

        if len(overall_performances) > 0:
            best_overall_locale = overall_performances.idxmax()
            baseline.best_overall_accuracy = overall_performances[best_overall_locale]
            baseline.best_overall_language = best_overall_locale

    return baseline


def _extract_base_model(
    model_name: Optional[str],
    folder: str,
) -> Optional[str]:
    """Extract base model identifier from model name or folder."""
    base_model = None

    if model_name and "merged_models" in model_name and "xlm-roberta" in model_name:
        match = re.search(r"(xlm-roberta-(?:base|large))", model_name)
        if match:
            base_model = match.group(1)
    elif model_name:
        try:
            base_model = naming_manager.extract_model_family(model_name)
        except ValueError:
            pass

    return base_model


def _determine_num_languages(
    metadata: ExperimentMetadata,
    model_name: Optional[str],
) -> Optional[int]:
    """Determine number of languages from metadata or model path."""
    num_languages = metadata.num_languages

    if num_languages is None and is_merge_model_path(model_name):
        num_languages = parse_num_languages_from_model_path(model_name)

    if num_languages is None and is_merge_model_path(model_name):
        counted = count_languages_in_merge_details(model_name)
        if counted is not None:
            num_languages = counted

    if num_languages is None and is_merge_model_path(model_name):
        num_languages = 4

    return num_languages


def _select_evaluation_matrix(
    base_model: Optional[str],
    evaluation_map: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Select appropriate evaluation matrix for the model family."""
    if not base_model:
        return evaluation_map.get("xlm-roberta-base") if evaluation_map else None

    for fam_key, mat in evaluation_map.items():
        if base_model in fam_key or fam_key in base_model:
            return mat

    return None


def _build_result_row(
    folder: str,
    results: Dict[str, Any],
    metadata: ExperimentMetadata,
    accuracy: Optional[float],
    locale: str,
    target_lang: Optional[str],
    num_languages: Optional[int],
    experiment_variant: str,
    baseline_data: Optional[BaselineData],
    eval_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a data row dictionary from experiment results."""
    perf_info = results.get("performance", {})
    model_name = eval_info.get("model_name")

    data_row = {
        "locale": locale,
        "experiment_type": metadata.experiment_type,
        "experiment_variant": experiment_variant,
        "folder_name": folder,
        "accuracy": accuracy,
        "correct_predictions": perf_info.get("correct_predictions"),
        "total_predictions": perf_info.get("total_predictions"),
        "error_rate": perf_info.get("error_rate"),
        "model_name": model_name,
        "subfolder": eval_info.get("subfolder"),
        "timestamp": eval_info.get("timestamp"),
        "target_lang": target_lang,
        "source_languages": metadata.source_languages,
        "weights": metadata.weights,
        "merge_mode": metadata.merge_mode,
        "num_languages": num_languages,
        "merge_timestamp": metadata.timestamp,
        "similarity_type": eval_info.get("similarity_type")
        or metadata.__dict__.get("similarity_type", "URIEL"),
    }

    if baseline_data:
        data_row.update(
            {
                "best_source_accuracy": baseline_data.best_source_accuracy,
                "best_source_language": baseline_data.best_source_language,
                "best_overall_accuracy": baseline_data.best_overall_accuracy,
                "best_overall_language": baseline_data.best_overall_language,
            }
        )

    return data_row


def aggregate_results(
    evaluation_matrix: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Aggregate results from all experiment folders with dynamic parsing.

    Args:
        evaluation_matrix: Single DataFrame or dict mapping model family to DataFrame.
        results_dir: Directory containing experiment results.

    Returns:
        DataFrame containing aggregated results.
    """
    folders = get_experiment_folders(results_dir)

    data = []

    evaluation_map: Dict[str, pd.DataFrame] = {}
    if isinstance(evaluation_matrix, dict):
        evaluation_map = evaluation_matrix
    elif evaluation_matrix is not None:
        evaluation_map["xlm-roberta-base"] = evaluation_matrix

    for folder in folders:
        folder_path = os.path.join(results_dir, folder)
        results = load_results_from_folder(folder_path)

        if not results:
            continue

        eval_info = results.get("evaluation_info", {})
        model_name = eval_info.get("model_name")

        metadata = parse_experiment_metadata(folder, folder_path, model_name)
        accuracy = extract_accuracy(results)

        locale = eval_info.get("locale") or metadata.locale
        target_lang = metadata.target_lang or locale

        num_languages = _determine_num_languages(metadata, model_name)
        base_model = _extract_base_model(model_name, folder)
        eval_matrix_for_family = _select_evaluation_matrix(base_model, evaluation_map)

        experiment_variant = determine_experiment_variant(
            metadata.merge_mode or metadata.experiment_type,
            num_languages,
            model_name,
            eval_info.get("similarity_type") or metadata.__dict__.get("similarity_type", "URIEL"),
            base_model,
            metadata.include_target,
        )

        baseline_data = None
        if eval_matrix_for_family is not None and metadata.source_languages:
            baseline_data = get_baseline_for_target(
                locale,
                metadata.source_languages,
                eval_matrix_for_family,
            )

        data_row = _build_result_row(
            folder=folder,
            results=results,
            metadata=metadata,
            accuracy=accuracy,
            locale=locale,
            target_lang=target_lang,
            num_languages=num_languages,
            experiment_variant=experiment_variant,
            baseline_data=baseline_data,
            eval_info=eval_info,
        )

        data.append(data_row)

    return pd.DataFrame(data)
