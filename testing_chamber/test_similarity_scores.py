import pandas as pd
import sys
import os

# Add the project root to the Python path to allow direct imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from merginguriel.utils import get_similarity_scores


def load_df():
    # Adjust path to be relative to the script's location
    df_path = os.path.join(project_root, "big_assets/language_similarity_matrix.csv")
    df = pd.read_csv(df_path, index_col=0)
    return df


def main():
    """
    This script demonstrates how to get raw and normalized similarity scores
    between a source language and a list of target languages.
    """
    print("--- Demonstrating Similarity Score Extraction ---")
    
    try:
        df = load_df()
        source_language = "eng"
        target_languages = ["ind", "jav", "spa", "fra"] # Indonesian, Javanese, Spanish, French

        print(f"\nSource Language: {source_language}")
        print(f"Target Languages: {target_languages}")

        scores = get_similarity_scores(source_language, target_languages, df)

        print("\nRaw scores:", scores["raw_scores"])
        print("Normalized scores:", scores["normalized_scores"])
        
        # Verify that the normalized scores sum to 1
        if scores["normalized_scores"]:
            print(
                "Sum of normalized scores:",
                sum(scores["normalized_scores"].values()),
            )

    except FileNotFoundError:
        print("\nError: Could not find the similarity matrix file.")
        print("Please ensure 'big_assets/language_similarity_matrix.csv' exists.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
