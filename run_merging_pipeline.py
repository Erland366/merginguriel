import os
import sys
from datetime import datetime
import subprocess
import numpy as np
import argparse
import pandas as pd

# --- Add submodules and project root to Python path ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'submodules/auto_merge_llm')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
# --- End Path Setup ---

from merginguriel.utils import get_similarity_scores
from auto_merge_llm.methods import merging_methods_dict

def save_merge_details(output_dir: str, base_model: str, models_and_weights: dict, mode: str):
    """Saves a text file with details about the merge."""
    filepath = os.path.join(output_dir, "merge_details.txt")
    with open(filepath, 'w') as f:
        f.write(f"Merge Mode: {mode}\n")
        f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}\n\n")
        f.write(f"Base Model (for architecture): {base_model}\n\n")
        f.write("--- Merged Models and Weights ---\n")
        total_weight = sum(models_and_weights.values())
        for i, (model, weight) in enumerate(models_and_weights.items()):
            portion = (weight / total_weight) * 100 if total_weight > 0 else 0
            f.write(f"{i+1}. Model: {model}\n")
            f.write(f"   - Weight: {weight:.6f} ({portion:.2f}% of total)\n")
    print(f"Merge details saved to: {filepath}")

def run_evaluation(model_path: str):
    """Calls the evaluation script on the specified model."""
    evaluation_script_path = os.path.join(project_root, "testing_chamber/evaluate_base_encoder.py")
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

def main():
    parser = argparse.ArgumentParser(description="A pipeline to merge models using either URIEL similarity weights or manual weights.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['uriel', 'manual'],
        help="The merging mode to use. 'uriel' for automatic similarity-based weights, 'manual' for user-defined weights."
    )
    args = parser.parse_args()

    print("*****************************************************")
    print(f"*        Model Merging Pipeline (Mode: {args.mode.upper()})      *")
    print("*****************************************************")

    # --- 1. Configuration ---
    BASE_MODEL = "xlm-roberta-base"
    models_and_weights = {}

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

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

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

    # --- 2. Perform Model Merging ---
    print("\n--- Step 2: Performing Model Merge ---")
    linear_merger = merging_methods_dict["linear"]()
    models_to_merge_paths = list(models_and_weights.keys())
    weight_values = list(models_and_weights.values())
    method_params = {"weights": weight_values}
    
    print("Starting merge...")
    result = linear_merger.merge(
        base_model=BASE_MODEL,
        models_to_merge=models_to_merge_paths,
        method_params=method_params,
    )
    merged_model = result['merged_model']
    tokenizer = result['base_tokenizer']
    print("Merge successful!")

    # --- 3. Save the Merged Model and Details ---
    print("\n--- Step 3: Saving Merged Model & Details ---")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(project_root, "merged_models", f"{args.mode}_merge_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to: {output_dir}")
    save_merge_details(output_dir, BASE_MODEL, models_and_weights, args.mode)

    # --- 4. Evaluate the Merged Model ---
    print("\n--- Step 4: Evaluating Merged Model ---")
    run_evaluation(output_dir)
    
    print("\n*****************************************************")
    print("*                  Pipeline Finished                *")
    print("*****************************************************")

if __name__ == "__main__":
    main()
