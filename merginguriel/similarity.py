import lang2vec.lang2vec as l2v
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def sinkhorn_normalize(matrix, iterations=20):
    """
    Iteratively normalizes a matrix to make it doubly stochastic.

    Args:
        matrix (np.ndarray): The input matrix. Must be non-negative.
        iterations (int): The number of iterations to perform.

    Returns:
        np.ndarray: The normalized matrix where rows and columns sum to 1.
    """
    # Make a copy to avoid modifying the original matrix
    result = matrix.copy()
    
    # Add a small epsilon to avoid division by zero if a row/column is all zeros
    epsilon = 1e-9

    for _ in range(iterations):
        # Normalize across rows
        row_sums = result.sum(axis=1, keepdims=True)
        result = result / (row_sums + epsilon)
        
        # Normalize across columns
        col_sums = result.sum(axis=0, keepdims=True)
        result = result / (col_sums + epsilon)
        
    return result

def filter_top_k(matrix, k):
    """
    Filters the matrix to keep only the top-k values in each row.

    Args:
        matrix (np.ndarray): The input similarity matrix.
        k (int): The number of top values to keep for each row.

    Returns:
        np.ndarray: A sparse matrix with only the top-k values per row.
    """
    # Make a copy to avoid modifying the original
    sparse_matrix = matrix.copy()

    if sparse_matrix.diagonal().sum() == sparse_matrix.shape[0]:
        print("Warning: The diagonal of the matrix does consist entirely of ones. We will make it to 0")
        np.fill_diagonal(sparse_matrix, 0)
    
    # Get the number of rows
    num_rows = sparse_matrix.shape[0]
    print(num_rows)

    for i in range(num_rows):
        row = sparse_matrix[i, :]
        
        # Use np.partition to find the k-th largest value without a full sort
        # It's more efficient than np.sort
        # The [::-1] part is to sort in descending order
        # We find the k-th largest value. Note: k-1 because of 0-based indexing.
        if len(row) > k:
            kth_largest_value = np.partition(row, -k)[-k]
            # Set all values smaller than the k-th largest to zero
            row[row < kth_largest_value] = 0
            
    return sparse_matrix

def process_available_models(csv_path: str, k: int = 5, feature_type: str = "syntax_knn"):
    """
    Process all available models from CSV, calculating pairwise similarity between all languages.
    
    Args:
        csv_path: Path to CSV file containing model information
        k: Number of top-k values to keep
        feature_type: Type of features to use from lang2vec
        
    Returns:
        Tuple of (similarity_matrix, normalized_matrix, language_list, model_info)
    """
    # Read CSV and extract languages
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} models from {csv_path}")
    
    # Extract unique languages and convert locale format to URIEL format
    locale_to_uriel = {}
    model_info = {}
    
    for _, row in df.iterrows():
        locale = row['locale']
        model_name = row['model_name']
        
        # Convert locale (e.g., "af-ZA") to URIEL language code (e.g., "afr")
        uriel_lang = locale_to_uriel_code(locale)
        if uriel_lang:
            locale_to_uriel[locale] = uriel_lang
            model_info[uriel_lang] = {
                'locale': locale,
                'model_name': model_name,
                'author': row['author'],
                'base_model': row['base_model']
            }
    
    unique_langs = list(locale_to_uriel.values())
    print(f"Processing languages: {unique_langs}")
    
    # Get features for the unique languages
    try:
        features = l2v.get_features(unique_langs, feature_type)
        print(f"Successfully loaded {feature_type} features for {len(unique_langs)} languages")
        print(f"Features type: {type(features)}")
        
        # Handle different return formats from lang2vec
        if isinstance(features, dict):
            # Convert dict to array in the same order as unique_langs
            features_array = []
            for lang in unique_langs:
                if lang in features:
                    features_array.append(features[lang])
                else:
                    print(f"Warning: No features found for {lang}")
                    # Use zero vector as fallback
                    features_array.append(np.zeros(100))  # Assuming 100-dimensional features
            features = np.array(features_array)
        elif isinstance(features, list):
            features = np.array(features)
        
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None, None, None
    
    # Create similarity matrix for ALL pairs
    n = len(unique_langs)
    similarity_matrix = np.zeros((n, n))
    
    print("Calculating pairwise similarities...")
    for i, lang1 in enumerate(unique_langs):
        for j, lang2 in enumerate(unique_langs):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            else:
                feat1 = features[i].reshape(1, -1)
                feat2 = features[j].reshape(1, -1)
                sim = cosine_similarity(feat1, feat2)[0, 0]
                similarity_matrix[i, j] = sim
    
    # Remove negative values and scale to [0, 1]
    similarity_matrix = (similarity_matrix + 1) / 2
    
    print(f"Applying top-k filtering (k={k})...")
    sparse_k_matrix = filter_top_k(similarity_matrix, k)
    
    print("Applying Sinkhorn normalization...")
    normalized_matrix = sinkhorn_normalize(sparse_k_matrix, iterations=20)
    
    return similarity_matrix, normalized_matrix, unique_langs, model_info

def locale_to_uriel_code(locale: str) -> str:
    """
    Convert locale code (e.g., "af-ZA") to URIEL language code (e.g., "afr").
    
    Args:
        locale: Locale code in format "xx-XX"
        
    Returns:
        URIEL language code or None if not found
    """
    # Common mapping from locale to ISO 639-3 codes used by URIEL
    locale_map = {
        'af-ZA': 'afr', 'am-ET': 'amh', 'ar-SA': 'ara', 'az-AZ': 'aze',
        'bn-BD': 'ben', 'ca-ES': 'cat', 'cy-GB': 'cym', 'da-DK': 'dan',
        'de-DE': 'deu', 'el-GR': 'ell', 'en-US': 'eng', 'es-ES': 'spa',
        'fa-IR': 'fas', 'fi-FI': 'fin', 'fr-FR': 'fra', 'hi-IN': 'hin',
        'hu-HU': 'hun', 'hy-AM': 'hye', 'id-ID': 'ind', 'is-IS': 'isl',
        'it-IT': 'ita', 'ja-JP': 'jpn', 'jv-ID': 'jav', 'ka-GE': 'kat',
        'km-KH': 'khm', 'kn-IN': 'kan', 'ko-KR': 'kor', 'lv-LV': 'lav',
        'ml-IN': 'mal', 'mn-MN': 'mon', 'ms-MY': 'msa', 'my-MM': 'mya',
        'nb-NO': 'nob', 'nl-NL': 'nld', 'pl-PL': 'pol', 'pt-PT': 'por',
        'ro-RO': 'ron', 'ru-RU': 'rus', 'sl-SL': 'slv', 'sq-AL': 'sqi',
        'sw-KE': 'swa', 'ta-IN': 'tam', 'te-IN': 'tel', 'th-TH': 'tha',
        'tl-PH': 'tgl', 'tr-TR': 'tur', 'ur-PK': 'urd', 'vi-VN': 'vie',
        'zh-CN': 'zho', 'zh-TW': 'zho'
    }
    
    return locale_map.get(locale)

if __name__ == "__main__":
    # Process available models from CSV
    csv_path = "haryoaw_k_models.csv"
    k = 5
    feature_type = "syntax_knn"
    
    # Process all available models
    similarity_matrix, normalized_matrix, unique_langs, model_info = process_available_models(
        csv_path, k=k, feature_type=feature_type
    )
    
    if similarity_matrix is not None:
        # Create DataFrames with language codes
        normalized_df = pd.DataFrame(normalized_matrix, index=unique_langs, columns=unique_langs)
        similarity_df = pd.DataFrame(similarity_matrix, index=unique_langs, columns=unique_langs)
        
        print(f"\nProcessed {len(unique_langs)} languages:")
        for lang in unique_langs:
            print(f"  {lang} -> {model_info[lang]['locale']} ({model_info[lang]['model_name']})")
        
        print(f"\nNormalized matrix shape: {normalized_matrix.shape}")
        print("Sample of normalized weights:")
        print(normalized_df.iloc[:5, :5])
        
        # Save results
        output_filename = "language_similarity_matrix.csv"
        normalized_df.to_csv(f"sparsed_{output_filename}")
        similarity_df.to_csv(output_filename)
        
        # Save model mapping for later use in merging
        model_mapping_df = pd.DataFrame.from_dict(model_info, orient='index')
        model_mapping_df.to_csv("model_mapping.csv")
        
        print(f"\nResults saved to:")
        print(f"  - {output_filename} (raw similarity matrix)")
        print(f"  - sparsed_{output_filename} (normalized weights)")
        print(f"  - model_mapping.csv (language to model mapping)")
        
        # Show total weight distribution
        print(f"\nTotal weight per language (row sums):")
        row_sums = normalized_df.sum(axis=1)
        for lang, weight in row_sums.items():
            print(f"  {lang}: {weight:.4f}")
    else:
        print("Failed to process models.")