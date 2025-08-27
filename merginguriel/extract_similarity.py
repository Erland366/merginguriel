import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="lang2vec")

import lang2vec.lang2vec as l2v
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import argparse


def sinkhorn_normalize(matrix: np.ndarray, iterations: int = 20):
    result = matrix.copy()

    epsilon = 1e-9

    for _ in range(iterations):
        row_sums = result.sum(axis=1, keepdims=True)
        result = result / (row_sums + epsilon)

        col_sums = result.sum(axis=0, keepdims=True)
        result = result / (col_sums + epsilon)

    return result


def filter_top_k(matrix: np.ndarray, k: int):
    sparse_matrix = matrix.copy()

    if sparse_matrix.diagonal().sum() == sparse_matrix.shape[0]:
        print(
            "Warning: The diagonal of the matrix consist entirely of ones. We will make it to 0"
        )
        np.fill_diagonal(sparse_matrix, 0)

    num_rows = sparse_matrix.shape[0]

    for i in range(num_rows):
        row = sparse_matrix[i, :]

        if len(row) > k:
            kth_largest_value = np.partition(row, -k)[-k]
            row[row < kth_largest_value] = 0

    return sparse_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save a sparsified language similarity matrix."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="sparsed_language_similarity_matrix.csv",
        help="Filename for the output CSV file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top similar languages to keep for each language.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations for Sinkhorn normalization.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        help='The type of language features to use (e.g., "syntax_wals", "phonology_wals").',
    )

    args = parser.parse_args()

    output_filename = args.output_filename
    k = args.k
    iterations = args.iterations
    feature_type = args.feature_type

    all_langs = l2v.available_languages()
    all_langs = {str(lang) for lang in all_langs if lang is not None}

    print(f"Fetching '{feature_type}' features for {len(all_langs)} languages...")
    features = l2v.get_features(list(all_langs), feature_type)

    # Compute the cosine similarity between all pairs of language vectors
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(list(features.values()))

    # Sparsify the matrix by keeping only the top k neighbors for each language
    print(f"Sparsifying matrix to top {k} neighbors...")
    sparse_k_matrix = filter_top_k(similarity_matrix, k)

    # Normalize the sparse matrix using the Sinkhorn algorithm
    print(f"Normalizing matrix with {iterations} Sinkhorn iterations...")
    normalized_matrix = sinkhorn_normalize(sparse_k_matrix, iterations=iterations)

    # Create a DataFrame for better readability and save to CSV
    normalized_df = pd.DataFrame(
        normalized_matrix, index=list(features.keys()), columns=list(features.keys())
    )

    print(f"Saving the final matrix to '{output_filename}'...")
    normalized_df.to_csv(output_filename)
    print("Done.")


if __name__ == "__main__":
    main()
