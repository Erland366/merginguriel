#!/usr/bin/env python3
"""
Quick test to verify all merging modes are recognized by the argument parser.
"""

import subprocess
import sys

def test_mode_recognition():
    """Test if all modes are recognized by the argument parser."""
    print("🧪 Testing mode recognition in argument parser...")

    # All implemented modes
    modes = [
        'linear', 'fisher_simple', 'fisher_dataset', 'average', 'similarity',
        'ties', 'task_arithmetic', 'slerp', 'regmean', 'manual', 'uriel'
    ]

    for mode in modes:
        print(f"Testing {mode}...", end=" ")

        cmd = [
            sys.executable, "merginguriel/run_merging_pipeline_refactored.py",
            "--mode", mode, "--help"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            # Check if help output appears (means mode was accepted)
            if "usage:" in result.stdout.lower() or "choices:" in result.stdout.lower():
                print("✅")
            else:
                print("❌ - Help not shown")

        except subprocess.TimeoutExpired:
            print("⏰")
        except Exception as e:
            print(f"❌ - {e}")

if __name__ == "__main__":
    test_mode_recognition()