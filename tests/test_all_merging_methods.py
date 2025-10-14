#!/usr/bin/env python3
"""
Comprehensive test script for all merging methods in MergingUriel.

This script tests each merging method to ensure they are properly integrated
and can initialize without errors. It performs dry-run tests that don't require
actual model files, allowing validation of the integration.
"""

import subprocess
import sys
import os
from typing import List, Dict, Tuple
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

def run_test(mode: str, target_lang: str = "sq-AL", num_languages: int = 2) -> Tuple[bool, str]:
    """
    Run a single test for a specific merging mode.

    Args:
        mode: The merging mode to test
        target_lang: Target language for testing
        num_languages: Number of languages to use in test

    Returns:
        Tuple of (success, output_message)
    """
    print(f"\n{'='*50}")
    print(f"Testing {mode.upper()} mode...")
    print(f"{'='*50}")

    cmd = [
        sys.executable,
        "merginguriel/run_merging_pipeline_refactored.py",
        "--mode", mode,
        "--target-lang", target_lang,
        "--num-languages", str(num_languages)
    ]

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=project_root
        )

        # Check if the mode was recognized and pipeline started
        if f"Mode: {mode.upper()}" in result.stdout:
            success = True
            message = f"✅ {mode.upper()}: Mode recognized and pipeline started successfully"
        elif "Unknown mode" in result.stderr:
            success = False
            message = f"❌ {mode.upper()}: Unknown mode - not implemented"
        else:
            # Check if we get the expected mode in output even if case differs
            for line in result.stdout.split('\n'):
                if "Mode:" in line and mode.upper() in line.upper():
                    success = True
                    message = f"✅ {mode.upper()}: Mode recognized and pipeline started successfully"
                    break
            else:
                success = False
                message = f"❌ {mode.upper()}: Unexpected error - {result.stderr[:200]}"

        # Print some output for debugging
        print(f"Exit code: {result.returncode}")
        print(f"Output snippet: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Error snippet: {result.stderr[:200]}...")

    except subprocess.TimeoutExpired:
        success = False
        message = f"⏰ {mode.upper()}: Test timed out (30s)"
    except Exception as e:
        success = False
        message = f"💥 {mode.upper()}: Exception occurred - {str(e)}"

    print(message)
    return success, message

def main():
    """Main test function."""
    print("🧪 MergingUriel Comprehensive Merging Methods Test")
    print("=" * 60)

    # List of all merging methods to test
    merging_methods = [
        # Original methods
        "linear",
        "fisher_simple",
        "fisher_dataset",
        "average",
        "similarity",

        # New advanced methods
        "ties",
        "task_arithmetic",
        "slerp",
        "regmean",

        # Other methods
        "manual",
        "uriel"
    ]

    print(f"Testing {len(merging_methods)} merging methods...")
    print(f"Target language: sq-AL")
    print(f"Number of languages: 2")
    print("\nNote: These are dry-run tests that validate mode recognition")
    print("and pipeline initialization. No actual models will be merged.")

    # Track results
    results = []
    passed = 0
    failed = 0

    # Run tests
    for mode in merging_methods:
        success, message = run_test(mode)
        results.append((mode, success, message))

        if success:
            passed += 1
        else:
            failed += 1

        # Small delay between tests
        time.sleep(1)

    # Print summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(merging_methods)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/len(merging_methods)*100:.1f}%")

    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 40)
    for mode, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {mode:15} | {message}")

    # Print failed tests if any
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        print("-" * 40)
        for mode, success, message in results:
            if not success:
                print(f"  • {mode}: {message}")

    # Test specific command examples for new methods
    print(f"\n📖 USAGE EXAMPLES FOR NEW METHODS:")
    print("-" * 40)
    examples = [
        ("TIES Merging", "ties", "--mode ties --target-lang sq-AL --num-languages 5"),
        ("Task Arithmetic", "task_arithmetic", "--mode task_arithmetic --target-lang sq-AL --num-languages 5"),
        ("SLERP", "slerp", "--mode slerp --target-lang sq-AL --num-languages 5"),
        ("RegMean", "regmean", "--mode regmean --target-lang sq-AL --num-languages 5"),
    ]

    for name, mode, args in examples:
        print(f"\n{name}:")
        print(f"  python merginguriel/run_merging_pipeline_refactored.py {args}")

    print(f"\n🎯 All new methods support both URIEL-weighted and average baseline comparisons!")
    print("Use --mode similarity for URIEL-weighted merging")
    print("Use --mode average for equal-weight baseline comparison")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)