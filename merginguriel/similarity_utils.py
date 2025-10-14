"""
Similarity Utilities Module

Reusable functions for loading and processing language similarity matrices.
Used by both ensemble inference and model merging systems.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from .similarity import sinkhorn_normalize, filter_top_k


def load_similarity_matrix(matrix_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a pre-computed similarity matrix from CSV.

    Args:
        matrix_path: Path to the CSV file containing the similarity matrix
        verbose: Whether to print loading information

    Returns:
        Similarity matrix as DataFrame
    """
    if verbose:
        print(f"Loading similarity matrix from {matrix_path}")

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Similarity matrix not found: {matrix_path}")

    df = pd.read_csv(matrix_path, index_col=0)

    if verbose:
        print(f"Loaded similarity matrix with shape: {df.shape}")
        print(f"Available languages: {list(df.index)}")

    return df


def process_similarity_matrix(similarity_df: pd.DataFrame,
                             target_locale: str,
                             top_k: int = 20,
                             sinkhorn_iterations: int = 20,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Process a similarity matrix by applying top-k filtering and Sinkhorn normalization.

    Args:
        similarity_df: Pre-computed similarity matrix DataFrame
        target_locale: Target locale (e.g., "en-US")
        top_k: Number of top neighbors to keep per language
        sinkhorn_iterations: Number of Sinkhorn normalization iterations
        verbose: Whether to print progress information

    Returns:
        Processed DataFrame with normalized weights
    """
    if verbose:
        print(f"Processing similarity matrix for {target_locale}")
        print(f"Matrix shape: {similarity_df.shape}")

    if target_locale not in similarity_df.index:
        raise ValueError(f"Target locale '{target_locale}' not found in similarity matrix")

    # Convert to numpy array for processing
    similarity_matrix = similarity_df.values

    if verbose:
        print(f"Applying top-k filtering (k={top_k})...")

    # Apply top-k filtering
    sparse_matrix = filter_top_k(similarity_matrix, top_k)

    if verbose:
        print(f"Applying Sinkhorn normalization ({sinkhorn_iterations} iterations)...")

    # Apply Sinkhorn normalization to make rows sum to 1
    normalized_matrix = sinkhorn_normalize(sparse_matrix, iterations=sinkhorn_iterations)

    # Convert back to DataFrame with original index/columns
    result_df = pd.DataFrame(
        normalized_matrix,
        index=similarity_df.index,
        columns=similarity_df.columns
    )

    if verbose:
        print(f"Generated processed matrix: {result_df.shape}")

    return result_df


def get_similarity_weights(similarity_df: pd.DataFrame,
                          target_locale: str,
                          num_languages: int = 5,
                          top_k: int = 20,
                          sinkhorn_iterations: int = 20,
                          verbose: bool = True) -> List[Tuple[str, float]]:
    """
    Get similarity weights for target language.

    Args:
        similarity_df: Pre-computed similarity matrix DataFrame
        target_locale: Target locale (e.g., "en-US")
        num_languages: Number of similar languages to return
        top_k: Number of top neighbors to keep per language
        sinkhorn_iterations: Number of Sinkhorn normalization iterations
        verbose: Whether to print progress information

    Returns:
        List of (locale, weight) tuples sorted by weight (descending)
    """
    # Process the similarity matrix
    processed_df = process_similarity_matrix(
        similarity_df, target_locale, top_k, sinkhorn_iterations, verbose
    )

    # Get weights for target locale
    target_weights = processed_df.loc[target_locale]

    # Filter languages with non-zero weights and sort
    valid_languages = []
    for locale, weight in target_weights.items():
        if weight > 0:
            valid_languages.append((locale, weight))

    # Sort by weight (descending) and limit
    valid_languages.sort(key=lambda x: x[1], reverse=True)
    if num_languages > 0:
        valid_languages = valid_languages[:num_languages]

    if verbose:
        print(f"Top {len(valid_languages)} similar languages to {target_locale}:")
        for locale, weight in valid_languages:
            print(f"  - {locale}: {weight:.6f}")

    return valid_languages


def load_and_process_similarity(matrix_path: str,
                               target_locale: str,
                               num_languages: int = 5,
                               top_k: int = 20,
                               sinkhorn_iterations: int = 20,
                               verbose: bool = True) -> List[Tuple[str, float]]:
    """
    Convenience function to load and process similarity matrix in one call.

    Args:
        matrix_path: Path to the CSV file containing the similarity matrix
        target_locale: Target locale (e.g., "en-US")
        num_languages: Number of similar languages to return
        top_k: Number of top neighbors to keep per language
        sinkhorn_iterations: Number of Sinkhorn normalization iterations
        verbose: Whether to print progress information

    Returns:
        List of (locale, weight) tuples sorted by weight (descending)
    """
    # Load the similarity matrix
    similarity_df = load_similarity_matrix(matrix_path, verbose)

    # Get similarity weights
    return get_similarity_weights(
        similarity_df, target_locale, num_languages, top_k, sinkhorn_iterations, verbose
    )