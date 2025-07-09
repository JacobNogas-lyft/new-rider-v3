#!/usr/bin/env python3
"""
Standalone script to compare features between v2 and v3 decision tree models.
This helps debug feature discrepancies when analyzing models across data versions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_decision_tree_optimized_thresholds import compare_model_features

def main():
    print("=" * 80)
    print("DECISION TREE MODEL FEATURE COMPARISON")
    print("Comparing features between V2 and V3 models")
    print("=" * 80)
    
    # Test different segment/mode combinations
    test_combinations = [
        ('all', 'standard'),
        ('all', 'plus'),
        ('airport', 'luxsuv'),
        ('churned', 'premium'),
    ]
    
    for segment, mode in test_combinations:
        try:
            compare_model_features(segment, mode, max_depth=10)
        except Exception as e:
            print(f"\n❌ Error comparing {segment}/{mode}: {e}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 