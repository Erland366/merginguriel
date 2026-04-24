#!/usr/bin/env python
"""LoRA fine-tuning of XLM-RoBERTa for MASSIVE intent classification.

Single-locale LoRA training script. Saves adapter-only weights and optionally
pushes to HuggingFace Hub. Designed to be called per-locale by
run_all_lora_training.py for large-scale runs.
"""

import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

load_dotenv()

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


logger = logging.getLogger(__name__)


ALL_MASSIVE_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "hi-IN",
    "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE",
    "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM",
    "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL",
    "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN",
    "zh-TW",
]


@dataclass
class LoraModelArguments:
    """Arguments for model and LoRA configuration."""

    model_name_or_path: str = field(
        default="FacebookAI/xlm-roberta-base",
        metadata={"help": "Pretrained model identifier."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name (defaults to model_name_or_path)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for pretrained models."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Use fast tokenizer."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace auth token."},
    )

    # LoRA configuration
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank (r)."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling factor."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability."},
    )
    lora_target_modules: str = field(
        default="query,value",
        metadata={"help": "Comma-separated list of target modules for LoRA."},
    )

    # Wandb
    wandb_project: str = field(
        default="MergingUriel",
        metadata={"help": "Wandb project name."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb entity."},
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated wandb tags."},
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run wandb offline."},
    )


@dataclass
class LoraDataArguments:
    """Arguments for data loading."""

    locale: str = field(
        default="en-US",
        metadata={"help": "MASSIVE locale code (e.g. en-US, de-DE)."},
    )
    dataset_name: str = field(
        default="AmazonScience/massive",
        metadata={"help": "Dataset identifier."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length after tokenization."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Pad all samples to max_seq_length."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training set for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate eval set for debugging."},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Early stopping patience (eval calls with no improvement)."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum improvement to qualify as progress."},
    )

    # Hub push
    push_adapter_to_hub: bool = field(
        default=False,
        metadata={"help": "Push adapter to HuggingFace Hub after training."},
    )
    hub_adapter_id: Optional[str] = field(
        default=None,
        metadata={"help": "Hub repo ID for adapter. Auto-generated if not set."},
    )


def generate_wandb_run_name(model_args, data_args, training_args):
    """Generate descriptive wandb run name."""
    model_name = model_args.model_name_or_path.split("/")[-1]
    locale = data_args.locale
    rank = model_args.lora_rank
    lr = training_args.learning_rate
    epochs = int(training_args.num_train_epochs)

    if lr < 0.001:
        lr_str = f"lr{lr:.0e}"
    else:
        lr_str = f"lr{lr:.4f}"

    return f"lora_{model_name}_massive_{locale}_r{rank}_{lr_str}_ep{epochs}"


def main():
    parser = HfArgumentParser((LoraModelArguments, LoraDataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Default output_dir if not explicitly set
    if training_args.output_dir is None or training_args.output_dir == "tmp_trainer":
        training_args.output_dir = (
            f"haryos_model_loras/xlm-roberta-base_massive_lora_{data_args.locale}"
        )

    # Configure wandb
    if wandb_available and not model_args.wandb_offline:
        run_name = generate_wandb_run_name(model_args, data_args, training_args)
        os.environ["WANDB_RUN_NAME"] = run_name
        if model_args.wandb_project:
            os.environ["WANDB_PROJECT"] = model_args.wandb_project
        if model_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = model_args.wandb_entity
        if model_args.wandb_tags:
            os.environ["WANDB_TAGS"] = model_args.wandb_tags
        logger.info(f"Wandb run name: {run_name}")

    # Default training flags
    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        training_args.do_train = True
        training_args.do_eval = True
        training_args.do_predict = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training LoRA adapter for locale: {data_args.locale}")
    logger.info(f"LoRA config: rank={model_args.lora_rank}, alpha={model_args.lora_alpha}, "
                f"target_modules={model_args.lora_target_modules}")

    # Detect last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")

    set_seed(training_args.seed)

    # Load dataset for this locale
    logger.info(f"Loading {data_args.dataset_name} locale={data_args.locale}")
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.locale,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    raw_datasets = raw_datasets.rename_column("intent", "labels")

    # Labels
    label_list = raw_datasets["train"].features["labels"].names
    num_labels = len(label_list)
    logger.info(f"Found {num_labels} intent classes")

    # Load model, tokenizer, config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # Label mappings
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {i: v for i, v in enumerate(label_list)}

    # Apply LoRA
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    # Tokenize
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        result = tokenizer(
            examples["utt"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        result["labels"] = examples["labels"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Tokenizing",
        )

    # Prepare splits
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    test_dataset = raw_datasets["test"]

    logger.info(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}, Test: {len(test_dataset)}")

    # Metrics
    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.bf16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Early stopping
    callbacks = []
    if training_args.do_train and training_args.do_eval:
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_accuracy"
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = True
        if training_args.save_strategy == "no":
            training_args.save_strategy = "epoch"
        # Need at least 2 checkpoints to load best model at end
        if training_args.save_total_limit is not None and training_args.save_total_limit < 2:
            training_args.save_total_limit = 2
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=data_args.early_stopping_patience,
                early_stopping_threshold=data_args.early_stopping_threshold,
            )
        )

    # Trainer
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

    # Log LoRA config to wandb (after Trainer initializes wandb)
    if wandb_available and not model_args.wandb_offline and wandb.run is not None:
        wandb.config.update({
            "lora": {
                "rank": model_args.lora_rank,
                "alpha": model_args.lora_alpha,
                "dropout": model_args.lora_dropout,
                "target_modules": target_modules,
            },
            "locale": data_args.locale,
            "total_params": total_params,
            "trainable_params": trainable_params,
        })

    # Train
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        os.makedirs(training_args.output_dir, exist_ok=True)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Save adapter only (PEFT save_pretrained saves only LoRA weights)
        logger.info(f"Saving LoRA adapter to {training_args.output_dir}")
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate on validation
    if training_args.do_eval:
        logger.info("*** Evaluate (validation) ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Evaluate on test
    if training_args.do_predict:
        logger.info("*** Evaluate (test) ***")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        test_metrics["test_samples"] = len(test_dataset)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info(f"Test accuracy for {data_args.locale}: {test_metrics.get('test_accuracy', 'N/A')}")

    # Push to Hub
    if data_args.push_adapter_to_hub:
        hub_id = data_args.hub_adapter_id or (
            f"Erland/xlm-roberta-base-massive-lora-{data_args.locale}"
        )
        logger.info(f"Pushing adapter to Hub: {hub_id}")
        model.push_to_hub(hub_id, token=model_args.token)
        tokenizer.push_to_hub(hub_id, token=model_args.token)

    logger.info(f"Done. Locale={data_args.locale}, output={training_args.output_dir}")


if __name__ == "__main__":
    main()
