# Performance Tests

This folder contains performance-oriented tests for fractional mixed-precision
adjoint behavior in `torchfde`.

Unlike pure unit tests, these tests combine:

- correctness checks (finite outputs/gradients, expected solver routing),
- numerical-behavior checks (mixed vs low-only precision error patterns),
- lightweight timing checks (sanity-level runtime metrics).

## Test Inventory

- `test_fractional_performance_sanity.py`
- `test_forward_precision_error_matrix.py`

Both files are discoverable via `tests/run_all_tests.py --include-performance`.

## 1) Fractional Adjoint Performance Sanity

File: `test_fractional_performance_sanity.py`

### Mathematical Model

The test exercises a learned vector field in a fractional dynamics setting:

$$
D_t^\beta y(t) = f_\theta(t, y(t)), \qquad 0 < \beta < 1
$$

with `beta = 0.8` and a predictor-style adjoint solve (`method="predictor-f"`).

The vector field is a small MLP:

$$
f_\theta(y)=\tanh(W_2 \tanh(W_1 y + b_1)+b_2).
$$

The scalar objective used for backward timing/stability is:

$$
\mathcal{L}(y_T)=\text{mean}(y_T^2).
$$

### What Is Verified

For each precision/scaler configuration, the test verifies:

- correct backend selection from `_select_adjoint_solver(...)`,
- forward output dtype is preserved,
- loss is finite,
- input and parameter gradients exist and are finite,
- measured runtime statistics are finite and positive.

For dynamic scaling runs, it also verifies that scaler history is populated
(meaning scaling logic was actually exercised).

### Configurations

The matrix includes:

- `float32_unscaled` -> expected unscaled backend,
- `bfloat16_unscaled` -> expected unscaled backend (if supported),
- `float16_safe` -> expected unscaled-safe backend,
- `float16_dynamic` -> expected dynamic backend.

### Timing Protocol

Per configuration:

- warmup iterations: 2
- measured iterations: 4
- reports mean and population std of measured runtimes.

This is not a hard regression gate against absolute wall-clock numbers; it is
a fast sanity check that execution is stable and timings are valid.

## 2) Forward Precision Error Matrix Test

File: `test_forward_precision_error_matrix.py`

This test imports the manufactured-IVP forward benchmark logic from
`tests/performance/forward_precision_error_matrix.py` and turns it into pass/fail checks
over multiple seeds and step sizes.

### Manufactured Fractional IVP

For each seed, a coupled system is generated:

$$
D_t^\beta y(t) = A y(t) + g(t), \qquad y(0)=c_0,\qquad \beta=0.73
$$

where `A` and coefficient vectors are seed-generated.

The exact trajectory is prescribed as:

$$
y_{\text{true}}(t) = c_0 + c_1 t^\beta + c_2 t^{2\beta} + c_3 t^{3\beta}.
$$

Using the Caputo derivative identity

$$
D_t^\beta t^{k\beta}
=\frac{\Gamma(k\beta+1)}{\Gamma((k-1)\beta+1)}t^{(k-1)\beta},
$$

the forcing is defined as:

$$
g(t)=D_t^\beta y_{\text{true}}(t)-A y_{\text{true}}(t),
$$

so `y_true` is an exact solution by construction.

### Sweep Axes

The test runs across:

- seeds: `[2026, 2027]`
- step sizes: `[0.005, 0.0025]`

for both bf16 and fp16 families (fp16 family is CUDA-only in this suite).

### Precision Families and Modes

Each family compares:

- `mixed`: fp32 solve context with autocast low dtype for eligible ops,
- `low-only`: direct solve with tensors/ops in low dtype.

Reference `float32` solves are also computed per case.

### Error Metrics

At final time `T`, absolute error against exact solution is measured by:

- max absolute error:
  $$\|e\|_\infty = \max_i |y_i - y_i^{\text{true}}|$$
- mean absolute error:
  $$\frac{1}{d}\sum_i |y_i - y_i^{\text{true}}|$$
- L2 absolute error:
  $$\|e\|_2 = \sqrt{\sum_i |y_i - y_i^{\text{true}}|^2}$$

Ratios used in assertions:

$$
r_{\text{mean}}=\frac{\text{err}_{\text{low-only, mean}}}{\text{err}_{\text{mixed, mean}}},
\qquad
r_{\text{max}}=\frac{\text{err}_{\text{low-only, max}}}{\text{err}_{\text{mixed, max}}}.
$$

### Assertions

Common checks per case:

- all error metrics are finite,
- all error metrics are non-negative.

bf16 family checks:

- at least one case shows meaningful mixed benefit:
  `max(r_mean) > 1.05`
- aggregate behavior is not catastrophically worse:
  `geometric_mean(r_mean) > 0.50`
  and `geometric_mean(r_max) > 0.50`

fp16 family checks (ranking may vary by case):

- at least one metric/case where mixed improves:
  `max(r_mean U r_max) > 1.01`
- enforce bounded behavior:
  `min(r_mean U r_max) > 0.10`
  and `max(r_mean U r_max) < 10.0`

These thresholds intentionally avoid brittle overfitting to one GPU/driver
combination while still catching severe regressions.

## Run Commands

Run individual performance tests:

```bash
python tests/performance/test_fractional_performance_sanity.py
python tests/performance/test_forward_precision_error_matrix.py
```

Run full suite including performance:

```bash
python tests/run_all_tests.py --include-performance
```

## Output and Quiet Mode

By default, tests may print concise summary lines for ratios/timing.

Set:

```bash
TORCHFDE_TEST_QUIET=1
```

to suppress those informational prints.

## Device and Skip Behavior

- If CUDA is unavailable, CUDA-dependent tests are skipped automatically.
- bf16 rows run only when `torch.cuda.is_bf16_supported()` is true on GPU.
- fp16 forward-comparison test path is CUDA-only in this suite.
