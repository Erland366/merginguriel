#!/usr/bin/env python
"""
Multi-language training script for MASSIVE dataset.

This script wraps training_bert.py to provide three main training modes:
1. LOLO (Leave One Language Out): Train on all languages except one
2. Super Model: Train on all languages at once
3. All LOLO: Run LOLO training for all languages in batch mode

Usage Examples:
    # LOLO - Train on all languages except en-US
    python run_multilang_training.py --exclude en-US

    # LOLO - Train on all languages except sq-AL
    python run_multilang_training.py --exclude sq-AL

    # Super Model - Train on all 49 languages
    python run_multilang_training.py --super-model

    # All LOLO - Run LOLO training for all 49 languages
    python run_multilang_training.py --all-lolo

    # Custom output directory
    python run_multilang_training.py --exclude fr-FR --output_dir my_custom_model
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All available locales from the MASSIVE dataset
ALL_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "hi-IN",
    "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE",
    "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM",
    "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL",
    "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN",
    "zh-TW"
]


def get_training_languages(exclude_language=None, super_model=False):
    """
    Get the list of languages to train on based on the mode.

    Args:
        exclude_language: Language to exclude (for LOLO mode)
        super_model: Whether to use all languages (super model mode)

    Returns:
        List of languages to train on
    """
    if super_model:
        return ALL_LOCALES.copy()

    if exclude_language:
        if exclude_language not in ALL_LOCALES:
            raise ValueError(f"Language '{exclude_language}' not found in available locales")

        training_languages = [lang for lang in ALL_LOCALES if lang != exclude_language]
        return training_languages

    raise ValueError("Must specify either --exclude <language> or --super-model")


def generate_output_name(exclude_language=None, super_model=False, custom_output=None):
    """
    Generate appropriate output directory name following haryos_model conventions.

    Args:
        exclude_language: Language being excluded
        super_model: Whether this is super model mode
        custom_output: Custom output directory override

    Returns:
        Output directory name (relative to haryos_model/)
    """
    if custom_output:
        return custom_output

    if super_model:
        # Super model: xlm-roberta-base_massive_LOLO_all_languages
        return "xlm-roberta-base_massive_LOLO_all_languages"

    if exclude_language:
        # LOLO mode: xlm-roberta-base_massive_LOLO_without_{locale}
        safe_name = exclude_language.replace("-", "_")
        return f"xlm-roberta-base_massive_LOLO_without_{safe_name}"

    return "xlm-roberta-base_massive_multilang"


def run_training(languages, output_dir, additional_args=None, exclude_language=None):
    """
    Run the training using training_bert.py.

    Args:
        languages: List of languages to train on (ignored if exclude_language is provided)
        output_dir: Output directory name (relative to haryos_model/)
        additional_args: Additional arguments to pass to training_bert.py
        exclude_language: Language to exclude (for LOLO mode)
    """
    # Get the path to training_bert.py
    script_dir = Path(__file__).parent
    training_script = script_dir / "training_bert.py"

    if not training_script.exists():
        raise FileNotFoundError(f"training_bert.py not found at {training_script}")

    # Construct full output path in haryos_model directory
    project_root = script_dir.parent
    full_output_dir = project_root / "haryos_model" / output_dir

    # Build the command
    if exclude_language:
        # For LOLO mode, use exclude language parameter
        cmd = [
            sys.executable, str(training_script),
            "--exclude_language", exclude_language,
            "--output_dir", str(full_output_dir),
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--model_name_or_path", "xlm-roberta-base",
            "--num_train_epochs", "15"
        ]
        print(f"🚀 Starting LOLO training (excluding {exclude_language})...")
    else:
        # Normal mode with specified languages
        cmd = [
            sys.executable, str(training_script),
            "--dataset_config_name", ",".join(languages),
            "--output_dir", str(full_output_dir),
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--model_name_or_path", "xlm-roberta-base",
            "--num_train_epochs", "15"
        ]
        print(f"🚀 Starting multi-language training...")
        print(f"📚 Training on {len(languages)} languages: {', '.join(languages[:5])}{'...' if len(languages) > 5 else ''}")

    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)

    print(f"💾 Output directory: {full_output_dir}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("-" * 80)

    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"❌ Error: {script_dir} not found")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Multi-language training script for MASSIVE dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LOLO - Train on all languages except en-US
  python run_multilang_training.py --exclude en-US

  # Super Model - Train on all languages
  python run_multilang_training.py --super-model

  # All LOLO - Run LOLO training for all 49 languages
  python run_multilang_training.py --all-lolo

  # Custom output directory
  python run_multilang_training.py --exclude fr-FR --output_dir my_model
        """
    )

    # Create mutually exclusive group for mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--exclude",
        help="Language to exclude (LOLO mode). Available: " + ", ".join(ALL_LOCALES[:5]) + "... (49 total)"
    )
    mode_group.add_argument(
        "--super-model",
        action="store_true",
        help="Train on all 49 languages (super model mode)"
    )
    mode_group.add_argument(
        "--all-lolo",
        action="store_true",
        help="Run LOLO training for all 49 languages (batch mode)"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        help="Custom output directory (auto-generated if not provided)"
    )

    parser.add_argument(
        "--additional-args",
        nargs="*",
        help="Additional arguments to pass to training_bert.py (e.g., --learning_rate 1e-4 --batch_size 32)"
    )

    args = parser.parse_args()

    try:
        if args.all_lolo:
            # Get all available languages
            all_languages = get_available_languages()
            logger.info(f"Running LOLO experiments for all {len(all_languages)} languages")

            successful_runs = 0
            failed_runs = 0

            for language in tqdm(all_languages, desc="Running LOLO experiments"):
                logger.info(f"Starting LOLO training without {language}")

                # Generate output name for this language
                output_name = generate_output_name(exclude_language=language)

                # Run training
                exit_code = run_training(None, output_name, args.additional_args, exclude_language=language)

                if exit_code == 0:
                    logger.info(f"✅ Successfully completed LOLO training without {language}")
                    successful_runs += 1
                else:
                    logger.error(f"❌ Failed LOLO training without {language}")
                    failed_runs += 1
                    # Continue with other languages even if one fails

            # Print summary
            total_runs = len(all_languages)
            logger.info(f"🎉 LOLO batch training completed!")
            logger.info(f"📊 Summary: {successful_runs}/{total_runs} successful, {failed_runs}/{total_runs} failed")

        else:
            # Get training languages
            languages = get_training_languages(
                exclude_language=args.exclude,
                super_model=args.super_model
            )

            # Generate output name
            output_dir = generate_output_name(
                exclude_language=args.exclude,
                super_model=args.super_model,
                custom_output=args.output_dir
            )

            # Construct full output path for success message
            project_root = Path(__file__).parent.parent
            full_output_dir = project_root / "haryos_model" / output_dir

            # Run training
            exit_code = run_training(languages, output_dir, args.additional_args)

            if exit_code == 0:
                print(f"\n🎉 Training complete! Model saved to: {full_output_dir}")
            else:
                print(f"\n💥 Training failed! Exit code: {exit_code}")
                sys.exit(exit_code)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()