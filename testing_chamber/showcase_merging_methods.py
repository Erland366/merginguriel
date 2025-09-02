import torch
import sys
import os

# Add the submodule path to the Python path to allow direct imports
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/auto_merge_llm'))
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from auto_merge_llm.methods import merging_methods_dict

def showcase_linear_merging():
    """
    Demonstrates how to use the LinearMerging method.
    This method performs a weighted average of the models' parameters.
    """
    print("--- Showcasing Linear Merging ---")
    
    # 1. Instantiate the merging method
    linear_merger = merging_methods_dict["linear"]()

    # 2. Define the base model and the models you want to merge.
    # Using small models from Hugging Face Hub for this example.
    # In a real scenario, the base_model would be the original pre-trained model.
    base_model_path = "gpt2"
    models_to_merge_paths = ["sshleifer/tiny-gpt2", "thecr7guy/gpt2-pretrain"]
    
    print(f"Base model: {base_model_path}")
    print(f"Models to merge: {models_to_merge_paths}")

    # 3. Define method-specific parameters
    # For linear merging, you can specify the weights for each model.
    # If weights are not provided, it defaults to an equal-weight average.
    method_params = {
        "weights": [0.5, 0.5]
    }
    print(f"Merging with weights: {method_params['weights']}")

    try:
        # 4. Run the merge process
        # The merge method handles loading the models, merging them, and returns the merged model.
        # This is a resource-intensive operation.
        print("\nStarting merge... (This may take a while and consume significant memory)")
        result = linear_merger.merge(
            base_model=base_model_path,
            models_to_merge=models_to_merge_paths,
            method_params=method_params,
        )
        
        merged_model = result['merged_model']
        print("\nLinear merging successful!")
        print("Merged model class:", merged_model.__class__.__name__)

        # You can now save the merged model
        # merged_model.save_pretrained("./merged_models/linear_merged_model")
        # result['base_tokenizer'].save_pretrained("./merged_models/linear_merged_model")
        print("Model ready to be used or saved.")

    except Exception as e:
        print(f"\nAn error occurred during merging: {e}")
        print("This is likely due to resource constraints. The code demonstrates the API usage.")

def showcase_slerp_merging():
    """
    Demonstrates how to use the SlerpMerging method.
    Slerp performs a spherical linear interpolation between the weights of two models.
    It's particularly useful when you want to interpolate between two fine-tuned models.
    """
    print("\n\n--- Showcasing Slerp Merging ---")
    
    # 1. Instantiate the merging method
    slerp_merger = merging_methods_dict["slerp"]()

    # 2. Define models. Slerp requires exactly two models to merge.
    base_model_path = "gpt2"
    # For slerp, the models to merge should ideally be fine-tunes of the same base model.
    models_to_merge_paths = ["sshleifer/tiny-gpt2", "microsoft/DialogRPT-tiny"]
    
    print(f"Base model: {base_model_path}")
    print(f"Models to merge: {models_to_merge_paths}")

    # 3. Define method-specific parameters
    # 'slerp_t' controls the interpolation between the two models (0.0 -> model A, 1.0 -> model B)
    # 'dot_threshold' is for stability. If models are too similar, it falls back to linear interpolation.
    method_params = {
        "slerp_t": 0.5,
        "dot_threshold": 0.99
    }
    print(f"Merging with t={method_params['slerp_t']}")

    try:
        # 4. Run the merge process
        print("\nStarting merge... (This may take a while)")
        result = slerp_merger.merge(
            base_model=base_model_path,
            models_to_merge=models_to_merge_paths,
            method_params=method_params,
        )
        
        merged_model = result['merged_model']
        print("\nSlerp merging successful!")
        print("Merged model class:", merged_model.__class__.__name__)
        print("Model ready to be used or saved.")

    except Exception as e:
        print(f"\nAn error occurred during merging: {e}")
        print("This is likely due to resource constraints. The code demonstrates the API usage.")


def showcase_ties_merging():
    """
    Demonstrates how to use the TiesMerging method.
    TIES merging is more advanced. It identifies and resolves conflicts between models
    by looking at the task vectors (the difference between fine-tuned and base models).
    """
    print("\n\n--- Showcasing TIES Merging ---")
    
    # 1. Instantiate the merging method
    ties_merger = merging_methods_dict["ties"]()

    # 2. Define models.
    base_model_path = "gpt2"
    models_to_merge_paths = ["sshleifer/tiny-gpt2", "microsoft/DialogRPT-tiny"]
    
    print(f"Base model: {base_model_path}")
    print(f"Models to merge: {models_to_merge_paths}")

    # 3. Define method-specific parameters
    # 'param_value_mask_rate' masks the smallest magnitude parameter values in task vectors.
    # 'scaling_coefficient' scales the final merged task vector before adding it to the base model.
    method_params = {
        "param_value_mask_rate": 0.8,
        "scaling_coefficient": 1.0
    }
    print(f"Merging with params: {method_params}")

    try:
        # 4. Run the merge process
        print("\nStarting merge... (This may take a while)")
        result = ties_merger.merge(
            base_model=base_model_path,
            models_to_merge=models_to_merge_paths,
            method_params=method_params,
        )
        
        merged_model = result['merged_model']
        print("\nTIES merging successful!")
        print("Merged model class:", merged_model.__class__.__name__)
        print("Model ready to be used or saved.")

    except Exception as e:
        print(f"\nAn error occurred during merging: {e}")
        print("This is likely due to resource constraints. The code demonstrates the API usage.")


if __name__ == "__main__":
    print("*****************************************************")
    print("*          Showcasing LLM Merging Methods           *")
    print("*****************************************************")
    print("This script demonstrates how to use different model merging methods.")
    print("NOTE: Running the actual merge is resource-intensive and may fail on")
    print("machines with limited RAM or CPU. This script is primarily for API demonstration.\n")
    
    # You can comment out methods you don't want to run.
    showcase_linear_merging()
    # showcase_slerp_merging()
    # showcase_ties_merging()

    print("\n\nScript finished.")
