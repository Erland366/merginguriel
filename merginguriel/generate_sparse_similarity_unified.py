"""
Generate a sparsified, Sinkhorn-normalized language similarity matrix with
MASSIVE-style locale codes (e.g., 'af-ZA') as rows/columns.

Based on the notebook flow: compute cosine similarities from lang2vec,
filter top-K neighbors per row, then apply Sinkhorn normalization to
approximate a doubly-stochastic matrix. Finally, remap indices/columns
to MASSIVE locales and save as a unified CSV.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lang2vec")

import lang2vec.lang2vec as l2v

from merginguriel.similarity import locale_to_uriel_code


def sinkhorn_normalize(matrix: np.ndarray, iterations: int = 20) -> np.ndarray:
    result = matrix.copy()
    eps = 1e-9
    for _ in range(iterations):
        row_sums = result.sum(axis=1, keepdims=True)
        result = result / (row_sums + eps)
        col_sums = result.sum(axis=0, keepdims=True)
        result = result / (col_sums + eps)
    return result


def filter_top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    sparse = matrix.copy()
    # Zero out diagonal if it's all ones from cosine self-similarity
    if np.allclose(np.diag(sparse), 1.0):
        np.fill_diagonal(sparse, 0.0)
    n = sparse.shape[0]
    for i in range(n):
        row = sparse[i, :]
        if row.size > k:
            thresh = np.partition(row, -k)[-k]
            row[row < thresh] = 0.0
    return sparse


def main():
    parser = argparse.ArgumentParser(description="Build sparsified similarity matrix with MASSIVE locales")
    parser.add_argument("--k", type=int, default=20, help="Top-K neighbors to keep per language")
    parser.add_argument("--iterations", type=int, default=20, help="Sinkhorn normalization iterations")
    parser.add_argument("--feature-type", type=str, default="syntax_knn", help="lang2vec feature type")
    parser.add_argument("--locales-csv", type=str, default="model_mapping_unified.csv", help="CSV containing MASSIVE locales")
    parser.add_argument("--out-sparse", type=str, default="sparsed_language_similarity_matrix_unified.csv", help="Output CSV for sparse+Sinkhorn matrix")
    parser.add_argument("--out-dense", type=str, default="language_similarity_matrix_unified.csv", help="Optional output CSV for dense cosine matrix")
    args = parser.parse_args()

    # Load locales list from mapping CSV (expects a 'locale' column)
    df_map = pd.read_csv(args.locales_csv)
    if 'locale' not in df_map.columns:
        raise ValueError(f"'{args.locales_csv}' must contain a 'locale' column")
    locales = df_map['locale'].tolist()

    # Map locales to URIEL codes; filter out those without mapping
    uriel_codes = []
    kept_locales = []
    for loc in locales:
        code = locale_to_uriel_code(loc)
        if code:
            uriel_codes.append(code)
            kept_locales.append(loc)
        else:
            print(f"Warning: No URIEL mapping for locale '{loc}', skipping")

    # Fetch lang2vec features for URIEL codes
    feats = l2v.get_features(uriel_codes, args.feature_type)
    # Convert dict→array in given order if needed
    if isinstance(feats, dict):
        X = np.stack([feats[c] for c in uriel_codes])
    else:
        X = np.asarray(feats)

    # Cosine similarity in [0,1]
    sim = cosine_similarity(X)
    sim = (sim + 1.0) / 2.0

    # Build sparse top-K and apply Sinkhorn
    sparse = filter_top_k(sim, args.k)
    sparse_norm = sinkhorn_normalize(sparse, iterations=args.iterations)

    # Save with MASSIVE locales as index/columns
    dense_df = pd.DataFrame(sim, index=kept_locales, columns=kept_locales)
    sparse_df = pd.DataFrame(sparse_norm, index=kept_locales, columns=kept_locales)

    dense_df.to_csv(args.out_dense)
    sparse_df.to_csv(args.out_sparse)

    print(f"Saved dense matrix to {args.out_dense} with shape {dense_df.shape}")
    print(f"Saved sparsified Sinkhorn matrix to {args.out_sparse} with shape {sparse_df.shape}")


if __name__ == "__main__":
    main()

