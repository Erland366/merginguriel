import pickle
import os
from huggingface_hub import list_models
from typing import Any
import argparse


def save_to_pickle(data: Any, filename: str) -> None:
    """
    Serializes and saves data to a pickle file. It handles both single
    objects and lists of objects from the huggingface_hub library by
    converting them to dictionaries.

    Args:
        data (Any): The data to save.
        filename (str): The name of the file to save to.
    """
    serializable_data = data
    try:
        serializable_data = [vars(d) for d in data]
        print(
            "Converted a list of objects to a list of dictionaries for serialization."
        )
    except TypeError:
        try:
            serializable_data = vars(data)
            print("Converted a single object to a dictionary for serialization.")
        except TypeError:
            print(
                "Data appears to be serializable already. Proceeding without conversion."
            )
            pass

    with open(filename, "wb") as f:
        pickle.dump(serializable_data, f)


def load_from_pickle(filename: str) -> Any:
    """
    Loads data from a pickle file.

    Args:
        filename (str): The name of the file to load from.

    Returns:
        Any: The loaded data.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def fetch_models(filter_str: str, model_name_or_path: str | None = None) -> list:
    """
    Fetches a list of models from the Hugging Face Hub based on a filter string.

    Args:
        filter_str (str): The filter to apply when searching for models (e.g., "bert").

    Returns:
        list: A list of ModelInfo objects.
    """
    models = list_models(
        filter=filter_str,
        full=True,
        library="pytorch",
        tags=[model_name_or_path] if model_name_or_path else None,
    )
    return list(models)


def main():
    """
    Main function to fetch, save, and load Hugging Face models.
    """
    parser = argparse.ArgumentParser(
        description="Fetch Hugging Face models and save them to a pickle file."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="bert",
        help='The filter string to use for fetching models (e.g., "gpt2", "t5").',
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=os.path.join("big_assets", "models.pickle"),
        help="The filename for the output pickle file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Optional model name or path to filter specific models.",
    )

    args = parser.parse_args()

    if args.model_name_or_path:
        print(f"Using model name or path filter: '{args.model_name_or_path}'")

    model_filter = args.filter
    output_filename = args.filename

    print(f"Fetching models with filter: '{model_filter}'...")
    models_list = fetch_models(model_filter)
    print(f"Fetched {len(models_list)} models.")

    save_to_pickle(models_list, output_filename)
    print(f"Saved models to '{output_filename}'.")

    loaded_models = load_from_pickle(output_filename)
    print(f"Successfully loaded {len(loaded_models)} models from '{output_filename}'.")


if __name__ == "__main__":
    main()
