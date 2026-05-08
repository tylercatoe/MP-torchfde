# Forward Precision Error Matrix (Known Fractional IVP)

This README explains `examples/forward_precision_error_matrix.py`.

## Goal

Benchmark forward-solve accuracy for a fractional ODE when:

- using mixed precision (fp32 state/params + low-precision autocast),
- versus using only low precision everywhere.

The script computes absolute errors against a known exact solution.

## Mathematical Setup

We solve a manufactured Caputo IVP:

$$
D_t^\beta y(t) = A y(t) + g(t), \qquad y(0)=c_0,\qquad 0<\beta<1
$$

where:

- $A$ is a dense coupled matrix,
- the exact solution is prescribed as

$$
y_{\text{true}}(t)=c_0+c_1 t^\beta+c_2 t^{2\beta}+c_3 t^{3\beta}.
$$

Using
$$
D_t^\beta t^{k\beta}=\frac{\Gamma(k\beta+1)}{\Gamma((k-1)\beta+1)}t^{(k-1)\beta},
$$
the forcing is constructed as:

$$
g(t)=D_t^\beta y_{\text{true}}(t)-A y_{\text{true}}(t),
$$

so $y_{\text{true}}$ is the exact solution by construction.

## What Is Compared

The script reports:

- `float64_reference`
- `float32_baseline`
- `bf16_mixed` and `bf16_low_only` (if bf16 supported)
- `fp16_mixed` and `fp16_low_only` (CUDA only)

Absolute error metrics against exact $y_{\text{true}}(T)$:

- max absolute error
- mean absolute error
- L2 absolute error

It also prints ratios:

- `bf16 low_only / mixed`
- `fp16 low_only / mixed`

Values greater than 1 mean mixed precision had lower error than low-only.

## Commands

From repo root:

```bash
python examples/forward_precision_error_matrix.py --device cuda
```

Save outputs:

```bash
python examples/forward_precision_error_matrix.py \
  --device cuda \
  --csv-out forward_precision_errors.csv \
  --json-out forward_precision_errors.json
```

CPU run:

```bash
python examples/forward_precision_error_matrix.py --device cpu
```

## Useful Arguments

- `--method` (default `predictor`)
- `--beta` (default `0.73`)
- `--t-final` (default `1.2`)
- `--step-size` (default `0.005`)
- `--dim` (default `6`)
- `--seed` (default `2026`)

If `step-size` is too large, discretization error can dominate and hide precision
differences. Use smaller `step-size` (for example `0.005` or `0.0025`) to better
separate mixed-vs-low precision behavior.
