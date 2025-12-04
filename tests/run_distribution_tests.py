"""Standalone test runner for batch distribution tests.

Run this script directly with: python tests/run_distribution_tests.py
"""

import sys
import traceback


def run_test(test_func):
    """Run a single test function and report results."""
    test_name = test_func.__name__
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ {test_name}")
        print(f"  AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ {test_name}")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# Import all test functions from test_batch_distribution
sys.path.insert(0, 'tests')
from test_batch_distribution import (
    test_round_robin_distribution_correctness,
    test_round_robin_with_two_devices,
    test_token_balance_improvement_with_sorted_samples,
    test_token_balance_with_three_devices,
    test_edge_case_single_device,
    test_edge_case_more_devices_than_batches,
    test_edge_case_empty_batches,
)


def main():
    """Run all tests and report summary."""
    print("=" * 70)
    print("Running Batch Distribution Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_round_robin_distribution_correctness,
        test_round_robin_with_two_devices,
        test_token_balance_improvement_with_sorted_samples,
        test_token_balance_with_three_devices,
        test_edge_case_single_device,
        test_edge_case_more_devices_than_batches,
        test_edge_case_empty_batches,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if run_test(test):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
