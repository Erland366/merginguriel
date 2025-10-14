#!/usr/bin/env python3
"""
Simple test for the new merging methods.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

from merginguriel.run_merging_pipeline_refactored import MergingStrategyFactory

def test_strategy_factory():
    """Test if all new strategies can be created."""
    print("🧪 Testing MergingStrategyFactory for new methods...")

    new_modes = ['ties', 'task_arithmetic', 'slerp', 'regmean']

    for mode in new_modes:
        print(f"Testing {mode}...", end=" ")
        try:
            strategy = MergingStrategyFactory.create(mode)
            print(f"✅ {strategy.__class__.__name__}")
        except Exception as e:
            print(f"❌ - {e}")

if __name__ == "__main__":
    test_strategy_factory()