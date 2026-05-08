# torchfde Tests

This directory mirrors the testing layout used in `rampde`:

- `core/`: correctness and gradient tests
- `performance/`: optional performance tests (placeholder for future additions)
- `run_all_tests.py`: convenience runner

## Run Core Tests

```bash
python tests/run_all_tests.py
```

## Run All Tests (including performance folder)

```bash
python tests/run_all_tests.py --include-performance
```

## Notes

- Mixed precision tests are GPU-aware and will skip CUDA-only checks when CUDA is unavailable.
- Core tests here are fractional-specific and use `beta < 1` wherever relevant.
