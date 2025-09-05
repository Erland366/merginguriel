import argparse
import torch
import shutil
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,   
    XLMRobertaForSequenceClassification,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)
from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import pandas as pd
import os
import json
from merginguriel.model_definition import compute_metrics, EarlyStoppingEpochCallback

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Set the pre-trained model.",
    )  # "bigscience/mt0-base"
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The epochs set for training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Only train on the first 100 examples.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./ablation",
        help="Set the pre-trained model.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./eval_results",
        help="Set the evaluation dump file path.",
    )
    parser.add_argument(
        "--override_results",
        action="store_true",
        default=False,
        help="When enabled, remove the previous checkpoints results.",
    )
    parser.add_argument(
        "--wandb_offline", default=False, action="store_true", help="wandb offline mode"
    )
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None, help="Hub model ID (if not provided, will be auto-generated)"
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="Hugging Face Hub token"
    )
    parser.add_argument(
        "--single_lang", type=str, default=None, help="Train on single language only (format: mn-MN)"
    )
    
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        assert (
            args.debug or args.override_results
        ), f"Output dir {args.out_path} already exists!"
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path)


    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # Load environment variables for Hub token
    load_dotenv()
    
    # Handle Hub authentication
    if args.push_to_hub:
        hub_token = args.hub_token or os.environ.get("HF_TOKEN")
        if hub_token:
            HfFolder.save_token(hub_token)
            print("Hugging Face Hub token configured successfully")
        else:
            print("Warning: No Hub token provided. Please set HF_TOKEN environment variable or use --hub_token")


    # Language Codes
    complete_langs = [
        "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK", "de-DE",
        "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "he-IL", "hi-IN", "hu-HU",
        "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH", "kn-IN",
        "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL", "pl-PL",
        "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN",
        "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "zh-CN", "zh-TW"
    ]

    # Use single language if specified, otherwise use all training languages
    if args.single_lang:
        train_langs = [args.single_lang]
    else:
        train_langs = [
            "ar-SA", "hy-AM", "bn-BD", "my-MM", "zh-CN", "zh-TW", "en-US", "fi-FI", "fr-FR",
            "ka-GE", "de-DE", "el-GR", "hi-IN", "hu-HU", "is-IS", "id-ID", "ja-JP", "jv-ID",
            "ko-KR", "lv-LV", "pt-PT", "ru-RU", "es-ES", "vi-VN", "tr-TR",
        ]

    def simplify_lang_code(lang):
        if lang == "zh-CN":
            return "zh_yue"
        elif lang == "zh-TW":
            return "zh"
        return lang.split("-")[0]

    simple_train_langs = [simplify_lang_code(lang) for lang in train_langs]
    simple_complete_langs = [simplify_lang_code(lang) for lang in complete_langs]

    # Loading and processing datasets
    dataset_train, dataset_valid = [], []
    for lang in tqdm(train_langs):
        dataset_train.append(
            load_dataset("AmazonScience/massive", lang, split='train').remove_columns(
                ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
            )
        )
        dataset_valid.append(
            load_dataset("AmazonScience/massive", lang, split='validation').remove_columns(
                ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
            )
        )

    dset_dict = DatasetDict(
        {
            "train": (
                concatenate_datasets(dataset_train).select(range(100))
                if args.debug
                else concatenate_datasets(dataset_train)
            ),
            "valid": (
                concatenate_datasets(dataset_valid).select(range(10))
                if args.debug
                else concatenate_datasets(dataset_valid)
            ),
        }
    )

    dataset_test = {}
    for lang in tqdm(complete_langs):
        dataset_test[lang] = load_dataset("AmazonScience/massive", lang, split='test').remove_columns(
            ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
        )

    dset_test_dict = DatasetDict(dataset_test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        lang_label = batch['locale'][0]
        encoding = tokenizer(batch["utt"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        lang_index = lang_to_index[lang_label]
        encoding['language_labels'] = torch.tensor([lang_index] * len(batch['utt']))
        
        return encoding

    if "intent" not in dset_dict.column_names:
        dset_dict = dset_dict.rename_column("intent", "labels")
    if "intent" not in dset_test_dict.column_names:
        dset_test_dict = dset_test_dict.rename_column("intent", "labels")


    dset_dict = dset_dict.map(encode_batch, batched=True)
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True)

    # Initialize model
    config = AutoConfig.from_pretrained(args.model_name, num_labels=60)
    if args.model_name == "bert-base-multilingual-cased":
        print(args.model_name)
        model = BertForSequenceClassification(config)

        dset_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "token_type_ids", "attention_mask", "language_labels"])
        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "token_type_ids", "attention_mask", "language_labels"])

    elif args.model_name == "xlm-roberta-base":
        print(args.model_name)
        model = XLMRForSequenceClassification(config)

        dset_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "attention_mask", "language_labels"])
        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "attention_mask", "language_labels"])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    training_args = TrainingArguments(
        output_dir=args.out_path,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_steps=100,
        dataloader_num_workers=32,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=dset_dict['train'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingEpochCallback(early_stopping_patience=3)],
    )

    trainer.train()

    trainer.model.save_pretrained(args.out_path)
    tokenizer.save_pretrained(args.out_path)
    
    # Push to Hub if requested
    if args.push_to_hub:
        # Generate hub model ID if not provided
        if not args.hub_model_id:
            # Format: Erland/{model_name}_massive_{lang_code}
            model_base_name = args.model_name.split('/')[-1]
            # For multiple languages, join with underscores
            lang_codes = '_'.join([simplify_lang_code(lang) for lang in train_langs])
            hub_model_id = f"Erland/{model_base_name}_massive_{lang_codes}"
        else:
            hub_model_id = args.hub_model_id
        
        print(f"Pushing model to Hub as: {hub_model_id}")
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(hub_model_id, exist_ok=True, private=False)
            print(f"Repository {hub_model_id} created/verified")
        except Exception as e:
            print(f"Error creating repository: {e}")
        
        # Push model and tokenizer
        try:
            trainer.model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            print(f"Model successfully pushed to {hub_model_id}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    model.eval()
    results = {}

    all_preds, all_labels = [], []

    for lang, dataset in dset_test_dict.items():
        test_loader = DataLoader(dataset, batch_size=64)
        for batch in tqdm(test_loader, desc=f"Testing lang {lang}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                logits, lang_logits, pooled_output = model(input_ids, attention_mask, None, None, None)[0]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)

        results[lang] = {
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy': accuracy
        }

    results_file_path = f"{args.eval_path}/{args.vector}_scores.json"
    if os.path.exists(results_file_path):
        if args.override_results:
            os.remove(results_file_path)
        else:
            raise AssertionError(f"Output file {results_file_path} already exists!")

    os.makedirs(f"{args.eval_path}", exist_ok=True)

    with open(
        results_file_path, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)


    print(results)