#!/usr/bin/env python3
"""
Test runner script for MergingUriel project.

This script provides convenient ways to run different categories of tests:
- Unit tests: Test individual components
- Integration tests: Test complete workflows
- Verification scripts: Validate system properties
- All tests: Run everything

Usage:
    python run_tests.py --type unit
    python run_tests.py --type integration
    python run_tests.py --type verification
    python run_tests.py --type all
    python run_tests.py --file tests/unit/test_similarity_utils.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def find_project_root():
    """Find the project root directory."""
    current = Path(__file__).parent
    while (current / "CLAUDE.md").exists() == False:
        current = current.parent
        if current.parent == current:
            break
    return current


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_pytest(test_path, test_type="all"):
    """Run pytest on a specific test path."""
    project_root = find_project_root()

    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]

    # Add pytest options based on test type
    if test_type == "unit":
        cmd.extend(["-x", "--tb=short"])
    elif test_type == "integration":
        cmd.extend(["-x", "--tb=short", "-s"])
    elif test_type == "verification":
        cmd.extend(["-x", "--tb=short"])

    return run_command(cmd, cwd=project_root)


def run_python_script(script_path):
    """Run a Python script directly."""
    project_root = find_project_root()
    return run_command([sys.executable, script_path], cwd=project_root)


def main():
    parser = argparse.ArgumentParser(description="MergingUriel Test Runner")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "verification", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--file",
        help="Specific test file to run"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test files"
    )

    args = parser.parse_args()

    project_root = find_project_root()
    tests_dir = project_root / "tests"

    if args.list:
        print("Available test files:")
        print("\nUnit Tests:")
        for test_file in (tests_dir / "unit").glob("*.py"):
            if test_file.name != "__init__.py":
                print(f"  {test_file}")

        print("\nIntegration Tests:")
        for test_file in (tests_dir / "integration").glob("*.py"):
            if test_file.name != "__init__.py":
                print(f"  {test_file}")

        print("\nVerification Scripts:")
        for test_file in (tests_dir / "verification").glob("*.py"):
            if test_file.name != "__init__.py":
                print(f"  {test_file}")

        return

    if args.file:
        # Run specific file
        if not os.path.exists(args.file):
            print(f"Error: Test file {args.file} not found")
            return 1

        print(f"Running specific test file: {args.file}")
        if args.file.endswith(".py"):
            # Try as pytest first, fall back to direct execution
            if not run_pytest(args.file):
                print("Pytest failed, trying direct execution...")
                success = run_python_script(args.file)
                return 0 if success else 1
        return 0

    success_count = 0
    total_count = 0

    # Run tests based on type
    if args.type in ["unit", "all"]:
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)

        unit_tests = list((tests_dir / "unit").glob("test_*.py"))
        total_count += len(unit_tests)

        for test_file in unit_tests:
            print(f"\nRunning {test_file.name}...")
            if run_pytest(str(test_file), "unit"):
                success_count += 1
                print(f"✅ {test_file.name} PASSED")
            else:
                print(f"❌ {test_file.name} FAILED")

    if args.type in ["integration", "all"]:
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)

        integration_tests = list((tests_dir / "integration").glob("test_*.py"))
        total_count += len(integration_tests)

        for test_file in integration_tests:
            print(f"\nRunning {test_file.name}...")
            if run_pytest(str(test_file), "integration"):
                success_count += 1
                print(f"✅ {test_file.name} PASSED")
            else:
                print(f"❌ {test_file.name} FAILED")

    if args.type in ["verification", "all"]:
        print("\n" + "="*60)
        print("RUNNING VERIFICATION SCRIPTS")
        print("="*60)

        verification_scripts = list((tests_dir / "verification").glob("test_*.py"))
        total_count += len(verification_scripts)

        for script_file in verification_scripts:
            print(f"\nRunning {script_file.name}...")
            if run_python_script(str(script_file)):
                success_count += 1
                print(f"✅ {script_file.name} PASSED")
            else:
                print(f"❌ {script_file.name} FAILED")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {success_count}/{total_count}")
    print(f"Failed: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())