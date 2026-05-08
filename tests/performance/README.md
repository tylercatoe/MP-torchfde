# Performance Tests

This folder contains lightweight CUDA performance sanity tests for the
fractional adjoint mixed-precision implementation.

## Included Test

- `test_fractional_performance_sanity.py`
  - Verifies adjoint backend selection across precision/scaler configurations
  - Runs forward+backward timing loops for fractional adjoint (`beta < 1`)
  - Checks output/gradient dtype and finiteness
  - Prints a short timing summary table

## Run

```bash
python tests/performance/test_fractional_performance_sanity.py
```

Or from the test runner:

```bash
python tests/run_all_tests.py --include-performance
```

## Notes

- Requires CUDA; the test is skipped automatically if CUDA is unavailable.
- bfloat16 rows run only on GPUs that report `torch.cuda.is_bf16_supported()`.
- Set `TORCHFDE_TEST_QUIET=1` to suppress summary prints.
