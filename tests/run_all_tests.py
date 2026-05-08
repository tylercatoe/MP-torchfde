#!/usr/bin/env python3
"""Run torchfde test suites."""

import argparse
import os
import sys
import unittest


def run_suite(test_dir: str, pattern: str = "test_*.py") -> unittest.result.TestResult:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run torchfde tests.")
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Also run tests in tests/performance.",
    )
    args = parser.parse_args()

    repo_tests_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(repo_tests_dir, "core")
    performance_dir = os.path.join(repo_tests_dir, "performance")

    print("Running core tests...")
    core_result = run_suite(core_dir)
    success = core_result.wasSuccessful()

    if args.include_performance and os.path.isdir(performance_dir):
        print("\nRunning performance tests...")
        perf_result = run_suite(performance_dir)
        success = success and perf_result.wasSuccessful()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
