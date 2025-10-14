import os
import sys
from datetime import datetime
import subprocess
import numpy as np
import argparse
import pandas as pd

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from merginguriel.utils import get_similarity_scores
from auto_merge_llm.methods import merging_methods_dict
from dotenv import load_dotenv

load_dotenv()

def save_merge_details(output_dir: str, base_model: str, models_and_weights: dict, mode: str, target_lang: str = None, base_model_info: dict = None):
    """Saves a text file with details about the merge."""
    filepath = os.path.join(output_dir, "merge_details.txt")
    with open(filepath, 'w') as f:
        f.write(f"Merge Mode: {mode}\n")
        f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}\n\n")
        f.write(f"Base Model (for architecture): {base_model}\n")
        if target_lang:
            f.write(f"Target Language: {target_lang}\n")
        f.write("\n--- Merged Models and Weights ---\n")
        
        if mode in ['similarity', 'average', 'fisher', 'fisher_simple', 'fisher_dataset']:
            # Handle the new format for similarity and average modes
            # Include base model if provided
            all_models = []
            
            if base_model_info:
                all_models.append({
                    'base_model_name': base_model_info['base_model_name'],
                    'subfolder': base_model_info['subfolder'],
                    'language': base_model_info['language'],
                    'locale': base_model_info['locale'],
                    'weight': base_model_info['weight']
                })
            
            # Add remaining models
            for model, info in models_and_weights.items():
                all_models.append({
                    'base_model_name': info['base_model_name'],
                    'subfolder': info['subfolder'],
                    'language': info['language'],
                    'locale': info['locale'],
                    'weight': info['weight']
                })
            
            # Calculate current total weight
            total_weight = sum(model_info['weight'] for model_info in all_models)
            
            # For similarity and average modes, normalize all weights to sum to 1.0 for display
            # This should match the actual weights used in the merge
            if mode in ['similarity', 'average'] and total_weight > 0:
                normalization_factor = 1.0 / total_weight
                for model_info in all_models:
                    model_info['weight'] = model_info['weight'] * normalization_factor
                total_weight = 1.0  # Now normalized to 1.0
            
            for i, model_info in enumerate(all_models):
                weight = model_info['weight']
                portion = (weight / total_weight) * 100 if total_weight > 0 else 0
                f.write(f"{i+1}. Model: {model_info['base_model_name']}\n")
                f.write(f"   - Subfolder: {model_info['subfolder']}\n")
                f.write(f"   - Language: {model_info['language']}\n")
                f.write(f"   - Locale: {model_info['locale']}\n")
                f.write(f"   - Weight: {weight:.6f} ({portion:.2f}% of total)\n")
        else:
            # Handle the old format for other modes (manual, uriel)
            total_weight = sum(models_and_weights.values())
            for i, (model, weight) in enumerate(models_and_weights.items()):
                portion = (weight / total_weight) * 100 if total_weight > 0 else 0
                f.write(f"{i+1}. Model: {model}\n")
                f.write(f"   - Weight: {weight:.6f} ({portion:.2f}% of total)\n")
        
        f.write(f"\nTotal Weight: {total_weight:.6f}\n")
    print(f"Merge details saved to: {filepath}")

def run_evaluation(model_path: str):
    """Calls the evaluation script on the specified model."""
    evaluation_script_path = os.path.join(project_root, "merginguriel/evaluate_base_encoder.py")
    if not os.path.exists(evaluation_script_path):
        print(f"Warning: Evaluation script not found at {evaluation_script_path}")
        return
    print(f"\n--- Starting evaluation for model: {model_path} ---")
    command = [sys.executable, evaluation_script_path, "--model_name_or_path", model_path]
    try:
        subprocess.run(command, check=True)
        print(f"--- Evaluation finished for {model_path} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")

def get_subfolder_for_language(locale: str, subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9"):
    """Generate subfolder pattern based on locale."""
    return subfolder_pattern.format(locale=locale)

def map_locale_to_language_code(locale: str):
    """Map MASSIVE locale format to language code using model_mapping.csv."""
    try:
        model_mapping_path = os.path.join(project_root, "model_mapping.csv")
        if os.path.exists(model_mapping_path):
            model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)
            
            # If locale is already a language code, return as-is
            if locale in model_mapping_df.index:
                return locale
            
            # If locale is in MASSIVE format, find the corresponding language code
            if locale in model_mapping_df['locale'].values:
                # Find the row where locale matches and return the index (language code)
                language_code = model_mapping_df[model_mapping_df['locale'] == locale].index[0]
                print(f"Mapped locale '{locale}' to language code '{language_code}'")
                return language_code
                
        # If no mapping found, return original locale
        print(f"No mapping found for locale '{locale}', using as-is")
        return locale
    except Exception as e:
        print(f"Could not load locale mapping: {e}")
        return locale

def load_similarity_weights(sparsed_matrix_path: str, model_mapping_path: str, target_lang: str, subfolder_pattern: str = "alpha_0.5_{locale}_epoch-9", num_languages: int = None):
    """Load similarity weights and create model-to-weight mapping for target language."""
    try:
        # Load the sparsed (normalized) similarity matrix
        sparsed_df = pd.read_csv(sparsed_matrix_path, index_col=0)
        
        # Load model mapping
        model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)
        
        print(f"Loaded similarity matrix with shape: {sparsed_df.shape}")
        print(f"Loaded model mapping with {len(model_mapping_df)} models")
        print(f"Using subfolder pattern: {subfolder_pattern}")
        
        # Find the target language row
        if target_lang not in sparsed_df.index:
            print(f"Target language '{target_lang}' not found in similarity matrix")
            print(f"Available languages: {list(sparsed_df.index)}")
            return None
        
        # Get weights for target language
        target_weights = sparsed_df.loc[target_lang]
        
        # Filter languages with non-zero weights and valid model mappings
        valid_languages = []
        for lang, weight in target_weights.items():
            if weight > 0 and lang in model_mapping_df.index:
                valid_languages.append((lang, weight))
        
        # Sort by weight in descending order and limit to num_languages
        valid_languages.sort(key=lambda x: x[1], reverse=True)
        if num_languages is not None and num_languages > 0:
            valid_languages = valid_languages[:num_languages]
            print(f"Limiting to top {num_languages} languages by similarity weight")
        
        # Create model-to-weight mapping
        models_and_weights = {}
        for lang, weight in valid_languages:
            model_info = model_mapping_df.loc[lang]
            model_name = model_info['model_name']
            locale = model_info['locale']
            
            # Generate language-specific subfolder
            subfolder = get_subfolder_for_language(locale, subfolder_pattern)
            
            # Use the format expected by auto_merge_llm: model_name@subfolder
            model_with_subfolder = f"{model_name}@{subfolder}"
            models_and_weights[model_with_subfolder] = {
                'weight': weight,
                'subfolder': subfolder,
                'language': lang,
                'locale': locale,
                'base_model_name': model_name
            }
            print(f"  - {model_with_subfolder}: {weight:.6f} (language: {lang})")
        
        return models_and_weights
        
    except Exception as e:
        print(f"Error loading similarity weights: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="A pipeline to merge models using either URIEL similarity weights, manual weights, calculated similarity weights, or equal weights.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['uriel', 'manual', 'similarity', 'average', 'fisher', 'fisher_simple', 'fisher_dataset'],
        help="The merging mode to use. 'uriel' for URIEL-based weights, 'manual' for user-defined weights, 'similarity' for calculated similarity weights, 'average' for equal weights, 'fisher' for Fisher information merging (requires training data), 'fisher_simple' for simplified Fisher merging using parameter magnitudes, 'fisher_dataset' for Fisher merging with huggingface dataset."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="sqi",
        help="Target language/locale for similarity-based merging (accepts both language codes like 'sqi' and MASSIVE locales like 'sq-AL', default: sqi for Albanian)"
    )
    parser.add_argument(
        "--subfolder-pattern",
        type=str,
        default="alpha_0.5_{locale}_epoch-9",
        help="Subfolder pattern to use for model loading (default: alpha_0.5_{locale}_epoch-9)"
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=5,
        help="Number of languages to include in merging (default: all available languages)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name for Fisher merging (e.g., 'imdb', 'ag_news')"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text data (default: text)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name containing labels (default: label)"
    )
    parser.add_argument(
        "--num-fisher-examples",
        type=int,
        default=100,
        help="Number of examples to use for Fisher computation (default: 100)"
    )
    args = parser.parse_args()

    print("*****************************************************")
    print(f"*        Model Merging Pipeline (Mode: {args.mode.upper()})      *")
    print("*****************************************************")

    # --- 1. Configuration ---
    BASE_MODEL = "xlm-roberta-base"
    models_and_weights = {}
    base_model_for_merge = BASE_MODEL  # Default fallback
    base_model_info = None  # Store base model info for similarity/average modes

    if args.mode == 'uriel':
        print("\n--- Step 1: Calculating URIEL Similarity Weights ---")
        MODELS_TO_MERGE = {
            "ind": "lur601/xlm_roberta-base-finetuned-paxn-id",
            "jav": "w11wo/xlm-roberta-base-finetuned-ud-javanese",
        }
        SOURCE_LANGUAGE = "eng"
        
        try:
            df_path = os.path.join(project_root, "big_assets/language_similarity_matrix.csv")
            df = pd.read_csv(df_path, index_col=0)
            target_langs = list(MODELS_TO_MERGE.keys())
            scores = get_similarity_scores(SOURCE_LANGUAGE, target_langs, df)
            weights = scores["normalized_scores"]
            
            if not weights or sum(weights.values()) == 0:
                print("Could not calculate weights. Aborting.")
                return
            
            models_and_weights = {MODELS_TO_MERGE[lang]: weight for lang, weight in weights.items()}
            print("Calculated Normalized Weights:")
            for lang, weight in weights.items():
                print(f"  - {MODELS_TO_MERGE[lang]}: {weight:.4f}")
            
            # Use the first model as the base model (remove it from models_to_merge)
            if models_and_weights:
                base_model_for_merge = list(models_and_weights.keys())[0]
                base_weight = list(models_and_weights.values())[0]
                # Remove the first model from models_and_weights
                first_model_key = list(models_and_weights.keys())[0]
                models_and_weights.pop(first_model_key)
                print(f"Using {base_model_for_merge} as base model")
                print(f"Base model weight: {base_weight:.4f}")
            else:
                base_model_for_merge = BASE_MODEL

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    elif args.mode == 'similarity':
        # Map target language to language code if it's in MASSIVE format
        target_lang_code = map_locale_to_language_code(args.target_lang)
        print(f"\n--- Step 1: Loading Similarity Weights for {args.target_lang} (mapped to: {target_lang_code}) ---")
        
        sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix.csv")
        model_mapping_path = os.path.join(project_root, "model_mapping.csv")
        
        models_and_weights = load_similarity_weights(sparsed_matrix_path, model_mapping_path, target_lang_code, args.subfolder_pattern, args.num_languages)
        
        if not models_and_weights:
            print("Could not load similarity weights. Aborting.")
            return
        
        # Extract model names and weights for merging
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [info['weight'] for info in models_and_weights.values()]
        
        total_weight = sum(weight_values)
        print(f"\nTotal weight: {total_weight:.6f}")
        print(f"Number of models to merge: {len(models_to_merge_paths)}")
        
        # Use the first model as the base model (remove it from models_to_merge)
        if models_to_merge_paths:
            base_model_for_merge = models_to_merge_paths[0]
            models_to_merge_paths = models_to_merge_paths[1:]  # Remove first model from merge list
            base_weight = weight_values[0]
            weight_values = weight_values[1:]  # Remove first weight
            
            # Store base model info for save_merge_details before removing it
            base_model_info = {
                'model_with_subfolder': base_model_for_merge,
                'weight': base_weight,
                'base_model_name': models_and_weights[base_model_for_merge]['base_model_name'],
                'subfolder': models_and_weights[base_model_for_merge]['subfolder'],
                'language': models_and_weights[base_model_for_merge]['language'],
                'locale': models_and_weights[base_model_for_merge]['locale']
            }
            
            # Remove base model from models_and_weights
            models_and_weights.pop(base_model_for_merge)
            
              # Normalize ALL weights (including base model) to sum to 1.0
            total_similarity_weight = base_weight + sum(weight_values)
            if total_similarity_weight > 0:
                # Update base model weight to normalized value
                base_weight = base_weight / total_similarity_weight
                # Renormalize remaining weights to sum to (1 - base_weight)
                if weight_values:
                    total_remaining_weight = sum(weight_values)
                    weight_values = [w * (1 - base_weight) / total_remaining_weight for w in weight_values]
                
                # Update base_model_info with normalized weight
                base_model_info['weight'] = base_weight
                
                # Update weights in models_and_weights dictionary for save_merge_details
                for i, model_key in enumerate(models_and_weights.keys()):
                    models_and_weights[model_key]['weight'] = weight_values[i]
            
            print(f"Using {base_model_for_merge} as base model")
            print(f"Base model weight: {base_weight:.6f}")
            print(f"Remaining models total weight: {sum(weight_values):.6f}")
        else:
            base_model_for_merge = BASE_MODEL
        
    elif args.mode == 'manual':
        print("\n--- Step 1: Validating Manual Configuration ---")
        MODELS_AND_WEIGHTS_MANUAL = {
            "lur601/xlm-roberta-base-finetuned-panx-en": 0.6,
            "lur601/xlm-roberta-base-finetuned-panx-it": 0.4,
        }
        total_weight = sum(MODELS_AND_WEIGHTS_MANUAL.values())
        if not np.isclose(total_weight, 1.0):
            print(f"Error: Your weights must sum to 1.0, but they sum to {total_weight}.")
            return
        
        models_and_weights = MODELS_AND_WEIGHTS_MANUAL
        print("Manual weights are valid.")
        print("Models to Merge:")
        for model, weight in models_and_weights.items():
            print(f"  - Weight {weight:.4f}: {model}")

    elif args.mode == 'average':
        print("\\n--- Step 1: Setting Up Average (Equal) Weights ---")
        
        # Use the same setup as similarity mode but with equal weights
        target_lang_code = map_locale_to_language_code(args.target_lang)
        print(f"Using target language: {args.target_lang} (mapped to: {target_lang_code})")
        
        sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix.csv")
        model_mapping_path = os.path.join(project_root, "model_mapping.csv")
        
        # Load similarity matrix and model mapping to get the same set of models
        sparsed_df = pd.read_csv(sparsed_matrix_path, index_col=0)
        model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)
        
        print(f"Loaded similarity matrix with shape: {sparsed_df.shape}")
        print(f"Loaded model mapping with {len(model_mapping_df)} models")
        print(f"Using subfolder pattern: {args.subfolder_pattern}")
        
        # Find the target language row
        if target_lang_code not in sparsed_df.index:
            print(f"Target language '{target_lang_code}' not found in similarity matrix")
            print(f"Available languages: {list(sparsed_df.index)}")
            return
        
        # Get the same models as similarity mode, but assign equal weights
        target_weights = sparsed_df.loc[target_lang_code]
        
        # Filter languages with non-zero weights and valid model mappings (same as similarity mode)
        valid_languages = []
        for lang, weight in target_weights.items():
            if weight > 0 and lang in model_mapping_df.index:
                valid_languages.append((lang, weight))
        
        # Sort by weight in descending order and limit to num_languages (same as similarity mode)
        valid_languages.sort(key=lambda x: x[1], reverse=True)
        if args.num_languages is not None and args.num_languages > 0:
            valid_languages = valid_languages[:args.num_languages]
            print(f"Limiting to top {args.num_languages} languages by similarity weight")
        
        if not valid_languages:
            print("No models found for the target language")
            return
        
        equal_weight = 1.0 / len(valid_languages)
        models_and_weights = {}
        
        for lang, weight in valid_languages:
            model_info = model_mapping_df.loc[lang]
            model_name = model_info['model_name']
            locale = model_info['locale']
            
            # Generate language-specific subfolder
            subfolder = get_subfolder_for_language(locale, args.subfolder_pattern)
            
            # Use the format expected by auto_merge_llm: model_name@subfolder
            model_with_subfolder = f"{model_name}@{subfolder}"
            models_and_weights[model_with_subfolder] = {
                'weight': equal_weight,
                'subfolder': subfolder,
                'language': lang,
                'locale': locale,
                'base_model_name': model_name
            }
            print(f"  - {model_with_subfolder}: {equal_weight:.6f} (language: {lang})")
        
        print(f"\\nUsing equal weights for {len(valid_languages)} models: {equal_weight:.6f} each")
        
        # Use the first model as the base model (remove it from models_to_merge)
        if models_and_weights:
            base_model_for_merge = list(models_and_weights.keys())[0]
            base_weight = list(models_and_weights.values())[0]['weight']
            
            # Store base model info for save_merge_details
            first_model_info = list(models_and_weights.values())[0]
            base_model_info = {
                'model_with_subfolder': base_model_for_merge,
                'weight': base_weight,
                'base_model_name': first_model_info['base_model_name'],
                'subfolder': first_model_info['subfolder'],
                'language': first_model_info['language'],
                'locale': first_model_info['locale']
            }
            
            # Remove the first model from models_and_weights
            first_model_key = list(models_and_weights.keys())[0]
            models_and_weights.pop(first_model_key)
            
            # For average mode, we want equal weights that sum to 1.0 including base model
            num_models = len(valid_languages)
            equal_weight = 1.0 / num_models
            
            # Update base weight and remaining weights to equal weights
            base_weight = equal_weight
            weight_values = [equal_weight] * (num_models - 1)
            
            # Update base_model_info with normalized weight
            base_model_info['weight'] = base_weight
            
            # Update weights in models_and_weights dictionary for save_merge_details
            for i, model_key in enumerate(models_and_weights.keys()):
                models_and_weights[model_key]['weight'] = weight_values[i]
            
            print(f"Using {base_model_for_merge} as base model")
            print(f"Equal weight for each model: {equal_weight:.6f}")
        else:
            base_model_for_merge = BASE_MODEL

    elif args.mode in ['fisher', 'fisher_simple']:
        # Use the same setup as similarity/average mode but with Fisher merging
        target_lang_code = map_locale_to_language_code(args.target_lang)
        print(f"Using target language: {args.target_lang} (mapped to: {target_lang_code})")

        sparsed_matrix_path = os.path.join(project_root, "sparsed_language_similarity_matrix.csv")
        model_mapping_path = os.path.join(project_root, "model_mapping.csv")

        # Load similarity matrix and model mapping to get the same set of models
        sparsed_df = pd.read_csv(sparsed_matrix_path, index_col=0)
        model_mapping_df = pd.read_csv(model_mapping_path, index_col=0)

        print(f"Loaded similarity matrix with shape: {sparsed_df.shape}")
        print(f"Loaded model mapping with {len(model_mapping_df)} models")
        print(f"Using subfolder pattern: {args.subfolder_pattern}")

        # Find the target language row
        if target_lang_code not in sparsed_df.index:
            print(f"Target language '{target_lang_code}' not found in similarity matrix")
            print(f"Available languages: {list(sparsed_df.index)}")
            return

        # Get the same models as similarity mode
        target_weights = sparsed_df.loc[target_lang_code]

        # Filter languages with non-zero weights and valid model mappings (same as similarity mode)
        valid_languages = []
        for lang, weight in target_weights.items():
            if weight > 0 and lang in model_mapping_df.index:
                valid_languages.append((lang, weight))

        # Sort by weight in descending order and limit to num_languages (same as similarity mode)
        valid_languages.sort(key=lambda x: x[1], reverse=True)
        if args.num_languages is not None and args.num_languages > 0:
            valid_languages = valid_languages[:args.num_languages]
            print(f"Limiting to top {args.num_languages} languages by similarity weight")

        if not valid_languages:
            print("No models found for the target language")
            return

        # For Fisher merging, store both URIEL weights and model info
        models_and_weights = {}
        uriel_weights = []  # Store URIEL similarity weights for Fisher merging

        for lang, weight in valid_languages:
            model_info = model_mapping_df.loc[lang]
            model_name = model_info['model_name']
            locale = model_info['locale']

            # Generate language-specific subfolder
            subfolder = get_subfolder_for_language(locale, args.subfolder_pattern)

            # Use the format expected by auto_merge_llm: model_name@subfolder
            model_with_subfolder = f"{model_name}@{subfolder}"
            models_and_weights[model_with_subfolder] = {
                'weight': weight,  # Store URIEL weight for reference
                'subfolder': subfolder,
                'language': lang,
                'locale': locale,
                'base_model_name': model_name
            }
            uriel_weights.append(weight)
            print(f"  - {model_with_subfolder}: {weight:.6f} (URIEL weight) (language: {lang})")

        print(f"\\nUsing Fisher merging with URIEL weighting for {len(valid_languages)} models")

        if models_and_weights:
            base_model_for_merge = list(models_and_weights.keys())[0]
            base_urial_weight = uriel_weights[0] 

            first_model_info = list(models_and_weights.values())[0]
            base_model_info = {
                'model_with_subfolder': base_model_for_merge,
                'weight': base_urial_weight,
                'base_model_name': first_model_info['base_model_name'],
                'subfolder': first_model_info['subfolder'],
                'language': first_model_info['language'],
                'locale': first_model_info['locale']
            }

            first_model_key = list(models_and_weights.keys())[0]
            models_and_weights.pop(first_model_key)
            uriel_weights = uriel_weights[1:]

            print(f"Using {base_model_for_merge} as base model")
            print(f"Fisher merging mode: {args.mode}")
            print(f"Base model URIEL weight: {base_urial_weight:.6f}")
            print(f"Remaining models URIEL weights: {[f'{w:.6f}' for w in uriel_weights]}")
        else:
            base_model_for_merge = BASE_MODEL
            uriel_weights = []

    if args.mode in ['fisher', 'fisher_simple', 'fisher_dataset']:
        merger = merging_methods_dict[args.mode]()
    else:
        merger = merging_methods_dict["linear"]()

    if args.mode in ['similarity', 'average', 'fisher', 'fisher_simple', 'fisher_dataset']:
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [info['weight'] for info in models_and_weights.values()]
    elif args.mode == 'manual':
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [models_and_weights[model] for model in models_to_merge_paths]

        if models_to_merge_paths:
            base_model_for_merge = models_to_merge_paths[0]
            models_to_merge_paths = models_to_merge_paths[1:]  # Remove first model from merge list
            base_weight = weight_values[0]
            weight_values = weight_values[1:]  # Remove first weight

            # Renormalize ALL weights (including base model) to sum to 1.0
            total_manual_weight = base_weight + sum(weight_values)
            if total_manual_weight > 0:
                # Update base model weight to normalized value
                base_weight = base_weight / total_manual_weight
                # Renormalize remaining weights to sum to (1 - base_weight)
                if weight_values:
                    total_remaining_weight = sum(weight_values)
                    weight_values = [w * (1 - base_weight) / total_remaining_weight for w in weight_values]

                # Note: Manual mode doesn't use base_model_info, so we don't update it here

            print(f"Normalized base model weight: {base_weight:.6f}")
            print(f"Remaining models total weight: {sum(weight_values):.6f}")

            print(f"Using {base_model_for_merge} as base model")
            print(f"Base model weight: {base_weight:.6f}")
            print(f"Remaining models total weight: {sum(weight_values):.6f}")
        else:
            base_model_for_merge = BASE_MODEL
    elif args.mode == 'uriel':
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = list(models_and_weights.values())
        if weight_values:
            total_urial_weight = base_weight + sum(weight_values)
            if total_urial_weight > 0:
                base_weight = base_weight / total_urial_weight
                total_weight = sum(weight_values)
                weight_values = [w * (1 - base_weight) / total_weight for w in weight_values]

        print(f"Normalized base model weight: {base_weight:.6f}")
        print(f"Remaining models total weight: {sum(weight_values):.6f}")
    else:
        models_to_merge_paths = list(models_and_weights.keys())
        weight_values = [info['weight'] for info in models_and_weights.values()]
        base_model_for_merge = BASE_MODEL

    if args.mode in ['fisher', 'fisher_simple']:
        if 'uriel_weights' in locals() and uriel_weights:
            if uriel_weights:
                total_urial_weight = sum(uriel_weights)
                if total_urial_weight > 0:
                    normalized_urial_weights = [w / total_urial_weight for w in uriel_weights]
                    method_params = {"weights": normalized_urial_weights}
                    print(f"Passing normalized URIEL weights to Fisher merging: {[f'{w:.6f}' for w in normalized_urial_weights]}")
                else:
                    method_params = {"weights": [1.0 / len(uriel_weights)] * len(uriel_weights)}
                    print("Using equal weights for Fisher merging (URIEL weights sum to 0)")
            else:
                method_params = {}
                print("No URIEL weights to pass to Fisher merging")
        else:
            method_params = {}
            print("No URIEL weights available, using pure Fisher merging")
    elif args.mode == 'fisher_dataset':
        if args.dataset_name is None:
            print("Error: --dataset-name is required for fisher_dataset mode")
            return

        method_params = {
            "dataset_config": {
                "dataset_name": args.dataset_name,
                "dataset_split": args.dataset_split,
                "text_column": args.text_column,
                "label_column": args.label_column
            },
            "num_fisher_examples": args.num_fisher_examples,
            "normalize_fisher_weight": True,
            "minimal_fisher_weight": 1e-6,
            "fisher_scaling_coefficients": None
        }
        print(f"Using dataset: {args.dataset_name}")
        print(f"Text column: {args.text_column}, Label column: {args.label_column}")
        print(f"Number of Fisher examples: {args.num_fisher_examples}")
    else:
        method_params = {"weights": weight_values}
    
    print("Starting merge...")
    print(f"Base model: {base_model_for_merge}")
    print(f"Models to merge: {models_to_merge_paths}")
    print(f"Weights: {weight_values}")
    
    # Perform the merge
    result = merger.merge(
        base_model=base_model_for_merge,
        models_to_merge=models_to_merge_paths,
        method_params=method_params,
    )
    merged_model = result['merged_model']
    tokenizer = result['base_tokenizer']
    print("Merge successful!")

    # --- 3. Save the Merged Model and Details ---
    print("\n--- Step 3: Saving Merged Model & Details ---")
    output_dir = os.path.join(project_root, "merged_models", f"{args.mode}_merge_{args.target_lang}")
    
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to: {output_dir}")
    save_merge_details(output_dir, BASE_MODEL, models_and_weights, args.mode, args.target_lang, base_model_info)

    # --- 4. Evaluate the Merged Model ---
    print("\n--- Step 4: Evaluating Merged Model ---")
    run_evaluation(output_dir)
    
    print("\n*****************************************************")
    print("*                  Pipeline Finished                *")
    print("*****************************************************")

if __name__ == "__main__":
    main()
