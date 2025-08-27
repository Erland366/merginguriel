# Taken from this : https://github.com/Guinan-Su/auto-merge-llm/blob/main/methods/base_method.py
import os
import torch
import merginguriel
from merginguriel import logger
from abc import ABC, abstractmethod


class BaseMerge(ABC):
    def __init__(self):
        pass

    def copy_params_to_model(
        self,
        params: dict,
        model: torch.nn.Module,
    ):
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name])
            else:
                print(
                    f"Warning: Parameter '{name}' not found in provided params. Skipping."
                )

    def mask_params(
        self,
        base_model: torch.nn.Module,
        models_to_merge: list[torch.nn.Module],
        exclude_param_names_regex: list[str],
        mask_merging: dict[str, torch.Tensor] | None,
    ):
        if not mask_merging:
            logger.info("No mask provided, skipping masking step.")
            return
        
        all_models = [base_model] + models_to_merge
        logger.info(f"Applying masks to {len(all_models)} models.")

        with torch.no_grad():
            for model in all_models:
                for name, param in model.named_parameters():
                    if any(re.search(pattern, name) for pattern in exclude_param_names_regex):
                        logger.info(f"Excluding parameter '{name}' from masking.")
                        continue

                    if param_name in mask_merging:
                        mask = mask_merging[name].to(param.device)
                        param.mul_(mask)

    @abstractmethod
    def merge(
        self,
        base_model: torch.nn.Module,
        models_to_merge: list[torch.nn.Module],
        method_params: dict,
        mask_merging=None,
        exclude_param_names_regex=[],
    ):
        pass

    @abstractmethod
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default",
    ):
        pass

    def finalize_merge(
        self,
        base_model: torch.nn.Module,
        base_model_dict: dict,
        merging_model_list: list[dict],
        averaged_params
    ):
        self.copy_params_to_model(params=averaged_params, model=base_model)
        merged_res = {
            'merged_model': base_model,
            'base_tokenizer': base_model_dict['tokenizer'],
            'merged_model_tokenizers': [merging_model['tokenizer']
                                       for merging_model
                                       in merging_model_list]
        }
        return merged_res

    def copy_params_to_model(
        self,
        params: dict[str, torch.Tensor],
        model: torch.nn.Module
    ):
        """
        Copies parameters from a dictionary to a model's state_dict.
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def _load_checkpoint(self, model_name_or_path):
        res = {}
        try:
            temp_model_path = get_model_storage_path(model_name_or_path)
            res["model"] = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=temp_model_path, device_map="cpu"
            )
            res["tokenizer"] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
            res["config"] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
        except Exception as e:
            logger.error(e)
            res["model"] = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                cache_dir=CACHE_DIR,
                device_map="cpu",
            )
            res["tokenizer"] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path, cache_dir=CACHE_DIR
            )
            res["config"] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path, cache_dir=CACHE_DIR
            )
        return res

    def _load_checkpoints(self, base_model_path, models_to_merge_paths):
        based_model = {}
        merging_model_list = []
        based_model = self._load_checkpoint(base_model_path)
        for model_merge_path in models_to_merge_paths:
            merging_model_list.append(self._load_checkpoint(model_merge_path))
        return based_model, merging_model_list
