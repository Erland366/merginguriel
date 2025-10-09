#!/usr/bin/env python
"""
Test script to list all available MASSIVE dataset configurations (languages).

This helps you see what language codes are available for training.
"""

from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_massive_configs():
    """List all available MASSIVE dataset configurations."""

    try:
        # Load the MASSIVE dataset builder to get available configs
        dataset_module = load_dataset("AmazonScience/massive", trust_remote_code=False)

        # Get all available configurations
        configs = dataset_module.builder_configs

        print("=== Available MASSIVE Dataset Configurations ===\n")

        # Group by language family for better readability
        language_groups = {
            "English": ["en-US"],
            "African languages": ["af-ZA", "am-ET", "ha-NE", "ig-NG", "rw-RW", "sw-KE", "yo-NG", "zu-ZA"],
            "Arabic": ["ar-SA", "ar-AE", "ar-EG", "ar-DZ", "ar-JO", "ar-KW", "ar-MA", "ar-TN"],
            "East Asian": ["zh-CN", "zh-TW", "ja-JP", "ko-KR"],
            "European": ["de-DE", "en-GB", "es-ES", "es-MX", "fr-FR", "it-IT", "nl-NL", "pt-BR", "ru-RU", "sv-SE"],
            "South Asian": ["bn-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN", "ur-PK"],
            "Southeast Asian": ["id-ID", "ms-MY", "th-TH", "tl-PH", "vi-VN"],
            "Other": ["cy-GB", "da-DK", "el-GR", "fi-FI", "ga-IE", "he-IL", "hu-HU", "is-IS", "no-NO", "pl-PL", "pt-PT", "ro-RO", "sk-SK", "sl-SI", "tr-TR", "uk-UA"]
        }

        print("Usage examples:")
        print(f"python training_bert.py --dataset_config_name en-US  # English (US)")
        print(f"python training_bert.py --dataset_config_name af-ZA  # Afrikaans")
        print(f"python training_bert.py --dataset_config_name sw-KE  # Swahili")
        print(f"python training_bert.py --dataset_config_name fr-FR  # French")
        print()

        print("Available configurations by language group:")
        print("=" * 60)

        for group_name, lang_codes in language_groups.items():
            print(f"\n{group_name}:")
            for lang_code in lang_codes:
                if lang_code in configs:
                    print(f"  {lang_code}")

        print(f"\nAll available configurations ({len(configs)} total):")
        print("=" * 60)
        for config_name in sorted(configs.keys()):
            print(f"  {config_name}")

        print("\n=== Quick Start Examples ===\n")

        examples = [
            ("English (US)", "en-US"),
            ("Afrikaans", "af-ZA"),
            ("Swahili", "sw-KE"),
            ("French", "fr-FR"),
            ("German", "de-DE"),
            ("Chinese (Simplified)", "zh-CN"),
            ("Japanese", "ja-JP"),
            ("Arabic (Saudi Arabia)", "ar-SA"),
            ("Hindi", "hi-IN"),
            ("Spanish (Spain)", "es-ES"),
            ("Portuguese (Brazil)", "pt-BR"),
            ("Russian", "ru-RU")
        ]

        for desc, config in examples:
            if config in configs:
                print(f"# {desc}")
                print(f"python training_bert.py --dataset_config_name {config} --do_train --do_eval --num_train_epochs 5")
                print()

        print("Note: Some languages may have less training data than others.")
        print("The MASSIVE dataset contains intent classification data in 60 languages.")

    except Exception as e:
        logger.error(f"Failed to load MASSIVE dataset configurations: {e}")
        print("Make sure you have internet connection and the datasets library installed.")

if __name__ == "__main__":
    list_massive_configs()