
#!/usr/bin/env python
# Moved into merginguriel/ package
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "accelerate >= 0.12.0",
#     "datasets >= 1.8.0",
#     "sentencepiece != 0.1.92",
#     "scipy",
#     "scikit-learn",
#     "protobuf",
#     "torch >= 1.3",
#     "evaluate",
#     "wandb",
# ]
# ///

"""Finetuning the library models for intent classification on MASSIVE dataset."""
# This script adapts the original GLUE finetuning script for the MASSIVE intent classification task.

import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Union

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from dotenv import load_dotenv

load_dotenv()

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def generate_wandb_run_name(model_args, data_args, training_args):
    """
    Generate a descriptive wandb run name based on training configuration.

    Format: {model_name}_{dataset}_{config}_lr{lr}_ep{epochs}_compile{status}

    Examples:
    - roberta-base_massive_en-US_lr5e-5_ep3
    - roberta-base_massive_en-US_lr5e-5_ep3_compile1
    - bert-base-uncased_massive_en-US_lr3e-5_ep5_compile1
    """
    try:
        # Extract short model name (remove organization prefixes)
        model_name = model_args.model_name_or_path.split('/')[-1]

        # Extract dataset name (remove organization prefixes)
        dataset_name = data_args.dataset_name.split('/')[-1]

        # Get dataset config(s) - handle both single and multi-language cases
        if isinstance(data_args.dataset_config_name, str):
            if ',' in data_args.dataset_config_name:
                # Multi-language case
                configs = [lang.strip() for lang in data_args.dataset_config_name.split(',')]
                if len(configs) <= 3:
                    dataset_config = '-'.join(configs)
                else:
                    dataset_config = f"{len(configs)}langs-{configs[0]}"
            else:
                # Single language case
                dataset_config = data_args.dataset_config_name
        else:
            # List case
            if len(data_args.dataset_config_name) <= 3:
                dataset_config = '-'.join(data_args.dataset_config_name)
            else:
                dataset_config = f"{len(data_args.dataset_config_name)}langs-{data_args.dataset_config_name[0]}"

        # Format learning rate (e.g., 5e-5 instead of 0.00005)
        lr = training_args.learning_rate
        if lr == 0:
            lr_str = "lr0"
        elif lr < 0.001:
            # For small values like 0.00005, convert to 5e-5
            lr_str = f"lr{lr:.0e}"
        else:
            # For larger values, use fewer decimal places
            lr_str = f"lr{lr:.4f}"

        # Get number of epochs
        epochs = training_args.num_train_epochs
        epochs_str = f"ep{int(epochs)}"

        # Add compile flag if enabled (check both model_args and training_args)
        compile_suffix = "_compile" if (
            getattr(model_args, 'torch_compile', False) or
            getattr(training_args, 'torch_compile', False)
        ) else ""

        # Build run name
        run_name = f"{model_name}_{dataset_name}_{dataset_config}_{lr_str}_{epochs_str}{compile_suffix}"

        # Clean up any problematic characters for wandb
        run_name = run_name.replace("/", "_").replace("\\", "_").replace(":", "_")

        return run_name

    except Exception as e:
        logger.warning(f"Failed to generate wandb run name: {e}. Using default naming.")
        return None


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default="AmazonScience/massive",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Union[str, List[str]] = field(
        default="en-US",
        metadata={"help": "The configuration name(s) of the dataset to use (via the datasets library). Can be a single language (e.g., en-US) or a comma-separated list of languages for multi-language training (e.g., fr-FR,de-DE,es-ES)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    # Early stopping arguments
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Number of evaluation calls with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "Minimum change in the monitored quantity to qualify as improvement."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="FacebookAI/roberta-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    # Wandb arguments
    wandb_project: str = field(
        default="MergingUriel",
        metadata={"help": "Wandb project name for experiment tracking"}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb entity (team/user) for the project"}
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated tags for wandb run categorization"}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run wandb in offline mode"}
    )


def parse_dataset_configs(dataset_config_name: Union[str, List[str]]) -> List[str]:
    """
    Parse dataset configuration names to support both single language and multiple languages.

    Args:
        dataset_config_name: Either a string (single language or comma-separated languages) or list of strings

    Returns:
        List of language configuration names
    """
    if isinstance(dataset_config_name, str):
        # Check if it's a comma-separated list
        if ',' in dataset_config_name:
            return [lang.strip() for lang in dataset_config_name.split(',')]
        else:
            return [dataset_config_name]
    elif isinstance(dataset_config_name, list):
        # Handle case where HfArgumentParser creates a list with comma-separated strings
        if len(dataset_config_name) == 1 and ',' in dataset_config_name[0]:
            return [lang.strip() for lang in dataset_config_name[0].split(',')]
        else:
            # Return the list as-is (assuming it's already a list of language codes)
            return dataset_config_name
    else:
        raise ValueError(f"dataset_config_name must be str or list, got {type(dataset_config_name)}")


def load_and_combine_datasets(
    dataset_name: str,
    dataset_configs: List[str],
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    shuffle_seed: Optional[int] = None
):
    """
    Load and combine multiple language datasets.

    Args:
        dataset_name: Name of the dataset
        dataset_configs: List of language configuration names
        cache_dir: Cache directory
        token: HuggingFace token
        trust_remote_code: Whether to trust remote code
        shuffle_seed: Seed for shuffling the combined dataset

    Returns:
        Combined dataset dictionary with train/validation/test splits
    """
    logger.info(f"Loading and combining {len(dataset_configs)} language configurations: {dataset_configs}")

    combined_datasets = {}

    # Initialize lists to collect datasets for each split
    train_datasets = []
    validation_datasets = []
    test_datasets = []

    # Load each language dataset
    for lang_config in dataset_configs:
        try:
            logger.info(f"Loading {lang_config}...")
            lang_dataset = load_dataset(
                dataset_name,
                lang_config,
                cache_dir=cache_dir,
                token=token,
                trust_remote_code=trust_remote_code,
            )

            # Rename 'intent' column to 'labels' for compatibility
            lang_dataset = lang_dataset.rename_column("intent", "labels")

            # Add to respective split collections
            if 'train' in lang_dataset:
                train_datasets.append(lang_dataset['train'])
                logger.info(f"✓ {lang_config}: {len(lang_dataset['train'])} training samples")

            if 'validation' in lang_dataset:
                validation_datasets.append(lang_dataset['validation'])
                logger.info(f"✓ {lang_config}: {len(lang_dataset['validation'])} validation samples")

            if 'test' in lang_dataset:
                test_datasets.append(lang_dataset['test'])
                logger.info(f"✓ {lang_config}: {len(lang_dataset['test'])} test samples")

        except Exception as e:
            logger.error(f"Failed to load {lang_config}: {e}")
            continue

    if not train_datasets:
        raise ValueError("No training datasets could be loaded")

    # Combine datasets for each split
    logger.info("Combining datasets...")

    combined_train = concatenate_datasets(train_datasets)
    logger.info(f"Combined training set: {len(combined_train)} samples")

    combined_validation = None
    if validation_datasets:
        combined_validation = concatenate_datasets(validation_datasets)
        logger.info(f"Combined validation set: {len(combined_validation)} samples")

    combined_test = None
    if test_datasets:
        combined_test = concatenate_datasets(test_datasets)
        logger.info(f"Combined test set: {len(combined_test)} samples")

    # Shuffle the combined datasets if seed is provided
    if shuffle_seed is not None:
        logger.info(f"Shuffling datasets with seed {shuffle_seed}")
        combined_train = combined_train.shuffle(seed=shuffle_seed)
        if combined_validation is not None:
            combined_validation = combined_validation.shuffle(seed=shuffle_seed)
        if combined_test is not None:
            combined_test = combined_test.shuffle(seed=shuffle_seed)

    # Create combined dataset dictionary as DatasetDict
    from datasets import DatasetDict
    combined_dataset = DatasetDict()
    combined_dataset['train'] = combined_train
    if combined_validation is not None:
        combined_dataset['validation'] = combined_validation
    if combined_test is not None:
        combined_dataset['test'] = combined_test

    return combined_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Preprocess command line arguments to handle comma-separated dataset configs
    argv = sys.argv[1:]
    processed_argv = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--dataset_config_name' and i + 1 < len(argv):
            # Keep the flag
            processed_argv.append(arg)
            # Process the value (don't split here, just pass through)
            # We'll handle the splitting in parse_dataset_configs()
            processed_argv.append(argv[i + 1])
            i += 2
        else:
            processed_argv.append(arg)
            i += 1

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(processed_argv) == 1 and processed_argv[0].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(processed_argv[0]))
    else:
        # Temporarily modify sys.argv for parsing
        original_argv = sys.argv
        sys.argv = ['training_bert.py'] + processed_argv
        try:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        finally:
            sys.argv = original_argv

    # Apply opinionated defaults matching project conventions unless explicitly provided
    def _has_flag(name: str) -> bool:
        return any(arg == name for arg in processed_argv)

    # Model default
    if not _has_flag('--model_name_or_path'):
        model_args.model_name_or_path = 'xlm-roberta-base'

    # Train/eval/predict toggles
    if not _has_flag('--do_train'):
        training_args.do_train = True
    if not _has_flag('--do_eval'):
        training_args.do_eval = True
    if not _has_flag('--do_predict'):
        training_args.do_predict = True

    # Epochs and batch sizes
    if not _has_flag('--num_train_epochs'):
        training_args.num_train_epochs = 15
    if not _has_flag('--per_device_train_batch_size'):
        training_args.per_device_train_batch_size = 128
    if not _has_flag('--per_device_eval_batch_size'):
        training_args.per_device_eval_batch_size = 128

    # Early stopping patience (already default 3 in DataTrainingArguments)
    if not _has_flag('--early_stopping_patience'):
        data_args.early_stopping_patience = 3

    # Strategies must match when loading best model at end
    if not _has_flag('--evaluation_strategy'):
        training_args.evaluation_strategy = IntervalStrategy.EPOCH
    if not _has_flag('--save_strategy'):
        training_args.save_strategy = IntervalStrategy.EPOCH
    if not _has_flag('--save_total_limit'):
        training_args.save_total_limit = 15
    if not _has_flag('--load_best_model_at_end'):
        training_args.load_best_model_at_end = True
    if not _has_flag('--metric_for_best_model'):
        training_args.metric_for_best_model = 'accuracy'
    if not _has_flag('--greater_is_better'):
        training_args.greater_is_better = True
    if not _has_flag('--overwrite_output_dir'):
        training_args.overwrite_output_dir = True
    # Optional compilation
    if not _has_flag('--torch_compile'):
        try:
            training_args.torch_compile = True
        except Exception:
            pass

    # Setup wandb environment
    if wandb_available and not model_args.wandb_offline:
        # Generate descriptive run name
        auto_run_name = generate_wandb_run_name(model_args, data_args, training_args)

        os.environ["WANDB_PROJECT"] = model_args.wandb_project
        if model_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = model_args.wandb_entity
        if auto_run_name:
            os.environ["WANDB_RUN_NAME"] = auto_run_name
            logger.info(f"Generated wandb run name: {auto_run_name}")
        if model_args.wandb_tags:
            os.environ["WANDB_TAGS"] = model_args.wandb_tags

        # Configure wandb reporting
        if not hasattr(training_args, 'report_to') or training_args.report_to is None:
            training_args.report_to = "wandb"
        elif isinstance(training_args.report_to, list) and "wandb" not in training_args.report_to:
            training_args.report_to.append("wandb")
        elif training_args.report_to == "none":
            training_args.report_to = "wandb"
    elif model_args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

        # Generate descriptive run name even in offline mode
        auto_run_name = generate_wandb_run_name(model_args, data_args, training_args)
        if auto_run_name:
            os.environ["WANDB_RUN_NAME"] = auto_run_name
            logger.info(f"Generated wandb run name (offline): {auto_run_name}")

        training_args.report_to = "wandb"
    else:
        training_args.report_to = "none"

    # Configure early stopping and checkpointing settings
    if training_args.do_train and training_args.do_eval:
        # Auto-generate output directory based on wandb run name
        auto_run_name = generate_wandb_run_name(model_args, data_args, training_args)
        if auto_run_name:
            # Create output directory in checkpoint_results subfolder
            original_output_dir = training_args.output_dir
            training_args.output_dir = os.path.join("checkpoint_results", auto_run_name)
            logger.info(f"Auto-generated output directory: {training_args.output_dir}")
            logger.info(f"Original output_dir argument: {original_output_dir}")
        else:
            # Fallback to default if wandb name generation fails
            logger.info("Using provided output directory (wandb name generation failed)")

        # Enable evaluation and saving for early stopping
        if not hasattr(training_args, 'eval_strategy') or training_args.eval_strategy is None or training_args.eval_strategy == "no":
            training_args.eval_strategy = "epoch"
        if not hasattr(training_args, 'save_strategy') or training_args.save_strategy is None:
            training_args.save_strategy = "epoch"
        if not hasattr(training_args, 'save_total_limit'):
            training_args.save_total_limit = 3
        if not hasattr(training_args, 'load_best_model_at_end'):
            training_args.load_best_model_at_end = True
        if not hasattr(training_args, 'metric_for_best_model'):
            training_args.metric_for_best_model = "eval_accuracy"
        if not hasattr(training_args, 'greater_is_better'):
            training_args.greater_is_better = True
        if not hasattr(training_args, 'logging_strategy'):
            training_args.logging_strategy = "steps"
        training_args.logging_steps = 1

        # Configure torch_compile if enabled
        torch_compile_enabled = (
            getattr(training_args, 'torch_compile', False) or
            getattr(model_args, 'torch_compile', False)
        )
        if torch_compile_enabled:
            if not hasattr(training_args, 'torch_compile_backend'):
                training_args.torch_compile_backend = "inductor"
            if not hasattr(training_args, 'torch_compile_mode'):
                training_args.torch_compile_mode = "default"
            logger.info("torch_compile enabled")

        # Ensure output directory exists
        os.makedirs(training_args.output_dir, exist_ok=True)

        # Log checkpointing configuration
        logger.info(f"Early stopping configured with patience: {data_args.early_stopping_patience}")
        logger.info(f"Load best model at end: {training_args.load_best_model_at_end}")
        logger.info(f"Metric for best model: {training_args.metric_for_best_model}")
        logger.info(f"Evaluation strategy: {training_args.eval_strategy}")
        logger.info(f"Save strategy: {training_args.save_strategy}")
        logger.info(f"Save total limit: {training_args.save_total_limit}")
        logger.info(f"Output directory: {training_args.output_dir}")
        logger.info(f"Logging strategy: {training_args.logging_strategy}")
        logger.info(f"Logging steps: {training_args.logging_steps}")
        if torch_compile_enabled:
            logger.info(f"Torch compile backend: {training_args.torch_compile_backend}")
            logger.info(f"Torch compile mode: {training_args.torch_compile_mode}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Parse dataset configurations to support multi-language training
    dataset_configs = parse_dataset_configs(data_args.dataset_config_name)

    if len(dataset_configs) == 1:
        # Single language loading (original behavior)
        logger.info(f"Loading single language dataset {data_args.dataset_name} with config {dataset_configs[0]}")
        raw_datasets = load_dataset(
            data_args.dataset_name,
            dataset_configs[0],
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Rename 'intent' column to 'labels' for compatibility with transformers
        raw_datasets = raw_datasets.rename_column("intent", "labels")
    else:
        # Multi-language loading with concatenation and shuffling
        logger.info(f"Loading multi-language dataset {data_args.dataset_name} with configs {dataset_configs}")
        raw_datasets = load_and_combine_datasets(
            dataset_name=data_args.dataset_name,
            dataset_configs=dataset_configs,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            shuffle_seed=training_args.seed,
        )

    # Labels - MASSIVE dataset has 60 intent classes
    is_regression = False
    label_list = raw_datasets["train"].features["labels"].names
    num_labels = len(label_list)
    logger.info(f"Found {num_labels} intent classes: {label_list}")

    # Log additional configuration to wandb if available
    if wandb_available and not model_args.wandb_offline:
        wandb_config = {
            "dataset": {
                "name": data_args.dataset_name,
                "configs": dataset_configs,
                "num_languages": len(dataset_configs),
                "multi_language": len(dataset_configs) > 1,
                "num_labels": num_labels,
                "label_list": label_list,
                "max_seq_length": data_args.max_seq_length
            },
            "model": {
                "name": model_args.model_name_or_path,
                "num_parameters": None,  # Will be set after model loading
                "num_labels": num_labels
            }
        }

        if data_args.max_train_samples:
            wandb_config["dataset"]["max_train_samples"] = data_args.max_train_samples
        if data_args.max_eval_samples:
            wandb_config["dataset"]["max_eval_samples"] = data_args.max_eval_samples

        try:
            wandb.config.update(wandb_config)
        except Exception as e:
            logger.warning(f"Failed to update wandb config: {e}")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Log model parameters to wandb if available
    if wandb_available and not model_args.wandb_offline:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            wandb.config.update({
                "model": {
                    "name": model_args.model_name_or_path,
                    "num_parameters": total_params,
                    "num_trainable_parameters": trainable_params,
                    "num_labels": num_labels,
                    "model_type": model.config.model_type,
                    "vocab_size": model.config.vocab_size,
                    "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 'unknown')
                }
            })
            logger.info(f"Logged model info to wandb: {total_params:,} total parameters, {trainable_params:,} trainable")
        except Exception as e:
            logger.warning(f"Failed to log model parameters to wandb: {e}")

    # Preprocessing the raw_datasets
    # For MASSIVE dataset, we use the 'utt' field as input text and there's only one sentence
    sentence1_key = "utt"
    sentence2_key = None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Create label to id mappings for the MASSIVE dataset
    label_to_id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}

    # Update model config with label mappings
    model.config.label2id = label_to_id
    model.config.id2label = id2label

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

              # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "labels" in examples:
            # Check if labels are already integers (MASSIVE dataset) or strings
            if isinstance(examples["labels"][0], str):
                # Labels are strings, need to map to IDs
                result["labels"] = [(label_to_id[l] if l != -1 else -1) for l in examples["labels"]]
            else:
                # Labels are already integers, use them directly
                result["labels"] = examples["labels"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def print_class_distribution(dataset, split_name, languages_info=None):
        label_counts = Counter(dataset["labels"])
        total = sum(label_counts.values())
        logger.info(f"Class distribution in {split_name} set:")
        if languages_info:
            logger.info(f"  Languages: {languages_info}")
        logger.info(f"  Total samples: {total}")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  Label {label}: {count} ({count / total:.2%})")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        languages_info = f"{len(dataset_configs)} languages: {dataset_configs}" if len(dataset_configs) > 1 else dataset_configs[0]
        print_class_distribution(train_dataset, "train", languages_info)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if training_args.do_train:  # languages_info was defined in the training block
            print_class_distribution(eval_dataset, "validation", languages_info)
        else:
            # Define languages_info if we're not doing training
            languages_info = f"{len(dataset_configs)} languages: {dataset_configs}" if len(dataset_configs) > 1 else dataset_configs[0]
            print_class_distribution(eval_dataset, "validation", languages_info)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        if training_args.do_train:  # languages_info was defined in the training block
            print_class_distribution(predict_dataset, "test", languages_info)
        else:
            # Define languages_info if we're not doing training
            languages_info = f"{len(dataset_configs)} languages: {dataset_configs}" if len(dataset_configs) > 1 else dataset_configs[0]
            print_class_distribution(predict_dataset, "test", languages_info)

    # Log dataset statistics to wandb if available
    if wandb_available and not model_args.wandb_offline:
        try:
            dataset_stats = {
                "dataset/train_size": len(train_dataset) if training_args.do_train else 0,
                "dataset/eval_size": len(eval_dataset) if training_args.do_eval else 0,
                "dataset/test_size": len(predict_dataset) if training_args.do_predict else 0,
                "dataset/num_labels": num_labels,
                "dataset/max_seq_length": max_seq_length
            }

            # Log class distribution for training set
            if training_args.do_train:
                label_counts = Counter(train_dataset["labels"])
                for label_id, count in label_counts.items():
                    if label_id < len(label_list):
                        dataset_stats[f"class_counts/{label_list[label_id]}"] = count

            wandb.log(dataset_stats, step=0)
            logger.info("Logged dataset statistics to wandb")
        except Exception as e:
            logger.warning(f"Failed to log dataset statistics to wandb: {e}")

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric functions - use accuracy and F1 for intent classification
    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        if not training_args.eval_do_concat_batches:
            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(p.label_ids, axis=0)
        preds = np.argmax(preds, axis=1)  # Get the predicted class indices

        # Compute both accuracy and F1 score
        accuracy_result = accuracy_metric.compute(predictions=preds, references=labels)
        f1_result = f1_metric.compute(predictions=preds, references=labels, average="weighted")

        # Combine results
        result = {
            "accuracy": accuracy_result["accuracy"],
            "f1": f1_result["f1"]
        }
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Setup callbacks for early stopping
    callbacks = []
    if training_args.do_train and training_args.do_eval:
        if not hasattr(training_args, 'metric_for_best_model') or training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_accuracy"

        if not hasattr(training_args, 'save_strategy') or training_args.save_strategy is None:
            training_args.save_strategy = "epoch"

        if not hasattr(training_args, 'load_best_model_at_end'):
            training_args.load_best_model_at_end = True

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=data_args.early_stopping_patience,
            early_stopping_threshold=data_args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Added EarlyStoppingCallback with patience={data_args.early_stopping_patience}")
        logger.info(f"Monitoring metric: {training_args.metric_for_best_model}")
        logger.info(f"Load best model at end: {training_args.load_best_model_at_end}")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # Log checkpointing info before training
        logger.info(f"Starting training. Output directory: {training_args.output_dir}")
        logger.info(f"Save strategy: {training_args.save_strategy}")
        logger.info(f"Save total limit: {training_args.save_total_limit}")

        # Ensure output directory exists before training
        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {training_args.output_dir}")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Save model and log info
        logger.info("Saving model to output directory...")
        trainer.save_model()  # Saves the tokenizer too for easy upload
        logger.info(f"Model saved to: {training_args.output_dir}")

        # List contents of output directory for debugging
        if os.path.exists(training_args.output_dir):
            files = os.listdir(training_args.output_dir)
            logger.info(f"Files in output directory after saving: {files}")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

    # Create model card with multi-language support
    if len(dataset_configs) == 1:
        language_info = dataset_configs[0]
        dataset_info = f"MASSIVE {dataset_configs[0]}"
    else:
        language_info = f"{len(dataset_configs)} languages"
        dataset_info = f"MASSIVE Multi-Language ({len(dataset_configs)} languages)"

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
        "language": language_info,
        "dataset_tags": "AmazonScience/massive",
        "dataset_args": ",".join(dataset_configs) if len(dataset_configs) > 1 else dataset_configs[0],
        "dataset": dataset_info
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # Log final summary to wandb if available
    if wandb_available and not model_args.wandb_offline:
        try:
            final_summary = {
                "training_completed": True,
                "output_dir": training_args.output_dir,
                "final_model_path": os.path.join(training_args.output_dir),
                "dataset_used": f"{data_args.dataset_name}/{dataset_configs}",
                "dataset_configs": dataset_configs,
                "num_languages": len(dataset_configs),
                "multi_language": len(dataset_configs) > 1,
                "model_used": model_args.model_name_or_path
            }

            # Log final evaluation results if available
            if training_args.do_eval:
                final_metrics = trainer.evaluate()
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        final_summary[f"final_{key}"] = value

            wandb.config.update(final_summary)
            logger.info("Logged final training summary to wandb")
        except Exception as e:
            logger.warning(f"Failed to log final summary to wandb: {e}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
