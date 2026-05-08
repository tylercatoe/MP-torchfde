# torchfde Tests

This directory mirrors the testing layout used in `rampde`:

- `core/`: correctness and gradient tests
- `performance/`: optional CUDA performance sanity checks
- `run_all_tests.py`: convenience runner

## Run Core Tests

```bash
python tests/run_all_tests.py
```

## Run All Tests (including performance folder)

```bash
python tests/run_all_tests.py --include-performance
```

## Run Only Performance Sanity Test

```bash
python tests/performance/test_fractional_performance_sanity.py
```

## Notes

- Mixed precision tests are GPU-aware and will skip CUDA-only checks when CUDA is unavailable.
- Core tests here are fractional-specific and use `beta < 1` wherever relevant.
- Set `TORCHFDE_TEST_QUIET=1` to suppress extra benchmark summary prints.
