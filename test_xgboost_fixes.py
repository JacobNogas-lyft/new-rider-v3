#!/usr/bin/env python3
"""
Test script to verify XGBoost training script fixes work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train_xgboost_gridsearch_for_mode_and_segment import (
    main, RANDOM_SEARCH_ITERATIONS, MAX_PARALLEL_PROCESSES, CV_FOLDS
)

def test_configuration():
    """Test that configuration parameters are properly set."""
    print("Testing configuration parameters...")
    print(f"RANDOM_SEARCH_ITERATIONS: {RANDOM_SEARCH_ITERATIONS}")
    print(f"MAX_PARALLEL_PROCESSES: {MAX_PARALLEL_PROCESSES}")
    print(f"CV_FOLDS: {CV_FOLDS}")
    
    assert RANDOM_SEARCH_ITERATIONS > 1, "Should have more than 1 iteration for random search"
    assert MAX_PARALLEL_PROCESSES > 0, "Should have positive number of parallel processes"
    assert CV_FOLDS > 1, "Should have more than 1 CV fold"
    
    print("✓ Configuration parameters are valid")

def test_small_run():
    """Test a small run with minimal data."""
    print("\nTesting small run...")
    
    # Use minimal configuration for testing
    mode_list = ['premium']  # Just one mode
    segment_type_list = ['all']  # Just one segment
    
    try:
        main(mode_list, segment_type_list)
        print("✓ Small run completed successfully")
    except Exception as e:
        print(f"✗ Small run failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing XGBoost training script fixes...")
    print("=" * 50)
    
    test_configuration()
    test_small_run()
    
    print("\n" + "=" * 50)
    print("All tests passed! The fixes should work correctly.") 