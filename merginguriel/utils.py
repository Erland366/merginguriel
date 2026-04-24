import os
import pandas as pd
import pycountry
import random
import numpy as np
import torch

from collections import OrderedDict
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from typing import Optional


def load_hf_state_dict_smart(
    repo_id: str, device: str = "cpu", cache_dir: Optional[str] = None
) -> OrderedDict[str, torch.Tensor]:
    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        raise IOError(
            f"Could not list files for repo '{repo_id}'. "
            f"Please check if the repository exists and you have access. Error: {e}"
        )

    weights_files = [f for f in all_files if f.endswith(".safetensors")]
    loader_fn = load_file
    file_type = ".safetensors"

    if not weights_files:
        weights_files = [f for f in all_files if f.endswith(".bin")]
        loader_fn = lambda path, device: torch.load(path, map_location=device)
        file_type = ".bin"

    if not weights_files:
        raise IOError(
            f"No .safetensors or .bin weight files found in repo '{repo_id}'."
        )

    print(f"Found {len(weights_files)} '{file_type}' file(s) to load for '{repo_id}'.")

    state_dict = OrderedDict()
    try:
        weights_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        )

        chunk_state_dict = loader_fn(weights_path, device=device)

        state_dict.update(chunk_state_dict)

    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        pass

    return state_dict


def get_iso639_3_code_from_name(language_name: str | list) -> str | None | list:
    try:
        if isinstance(language_name, list):
            return [get_iso639_3_code_from_name(name) for name in language_name]
        language = pycountry.languages.lookup(language_name)

        return language.alpha_3

    except LookupError:
        print(f"Warning: The language '{language_name}' was not found.")
        return None
    except AttributeError:
        print(
            f"Warning: Found a language entry for '{language_name}', but it has no ISO 639-3 code."
        )
        return None


def get_language_name_from_iso639_3(code: str | list) -> str | None | list:
    try:
        if isinstance(code, list):
            return [get_language_name_from_iso639_3(c) for c in code]
        language = pycountry.languages.get(alpha_3=code)

        return language.name

    except KeyError:
        print(f"Warning: The code '{code}' was not found in the database.")
        return None


def get_all_positive_columns(
    code_or_language: str, df: pd.DataFrame
) -> dict[str, float]:
    if isinstance(code_or_language, list):
        return [get_all_positive_columns(c, df) for c in code_or_language]
    if len(code_or_language) == 3:
        language_code = code_or_language
    else:
        language_code = get_iso639_3_code_from_name(code_or_language)
        if language_code is None:
            return {}
    try:
        row = df.loc[language_code]
        positive_columns = row[row > 0]
        return positive_columns.to_dict()
    except KeyError:
        print(
            f"Warning: The language code '{language_code}' was not found in the DataFrame."
        )
        return {}


def get_similarity_scores(
    source_language: str, target_languages: list[str], df: pd.DataFrame
) -> dict:
    source_code = get_iso639_3_code_from_name(source_language)
    if source_code is None:
        return {"raw_scores": {}, "normalized_scores": {}}

    target_codes = [get_iso639_3_code_from_name(lang) for lang in target_languages]
    target_codes = [code for code in target_codes if code is not None]

    try:
        source_row = df.loc[source_code]
        raw_scores = {
            code: source_row.get(code, 0) for code in target_codes
        }

        total_score = sum(raw_scores.values())
        if total_score > 0:
            normalized_scores = {
                code: score / total_score for code, score in raw_scores.items()
            }
        else:
            normalized_scores = {code: 0 for code in target_codes}

        return {"raw_scores": raw_scores, "normalized_scores": normalized_scores}

    except KeyError:
        print(
            f"Warning: The source language code '{source_code}' was not found in the DataFrame."
        )
        return {"raw_scores": {}, "normalized_scores": {}}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["DATA_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False