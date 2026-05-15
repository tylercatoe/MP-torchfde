# Predictor Mixed Precision Guide

This note explains the mixed-precision implementation of:

- `forward_predictor` in `torchfde/torchfde/fdeadjoint.py`
- `backward_predictor` in `torchfde/torchfde/fdeadjoint.py`

The goal is to show:

1. where low precision is used,
2. where values are promoted back to higher precision,
3. why this split is useful for speed and memory.


## Scope

This document describes the current implementation around:

- `forward_predictor(...)`
- `backward_predictor(...)`
- helper casts:
  - `_state_storage_dtype(...)`
  - `_cast_state_dtype(...)`
  - `_cast_state_like(...)`


## Precision Strategy At A Glance

The predictor path uses a hybrid policy:

- Evaluate model dynamics (`func`) in `dtype_store` (often lower precision under autocast).
- Store long histories (`fhistory`, adjoint histories, `yhistory`) in `dtype_store`.
- Promote the active history window back to `dtype_hi` for convolution reduction (`tensordot`).
- Apply updates (`yn`, `adj_y`) in high precision state space.

In short: low precision for expensive per-step model eval + long-term storage, high precision for weighted history reductions.


## Forward Predictor Algorithm (Mixed Precision)

### Flow Diagram

```mermaid
flowchart TD
    A[Init y0, beta, tspan] --> B[Select dtype_hi from y0]
    B --> C[Select dtype_store = _state_storage_dtype(dtype_hi)]
    C --> D[Loop k = 0..N-2]
    D --> E[Cast yn -> dtype_store]
    E --> F[Compute f_k = func(t_k, yn_low)]
    F --> G[Store f_k in fhistory (dtype_store)]
    G --> H[Store y_k in yhistory buffer (dtype_store)]
    H --> I[Build predictor weights b_j_k_1 in dtype_hi]
    I --> J[Slice history window and cast window -> dtype_hi]
    J --> K[Vectorized weighted sum via stack + tensordot]
    K --> L[Update yn = y0 + gamma_beta * convolution_sum]
    L --> D
    D --> M[Return yn and yhistory]
```

### Step Summary

1. Determine `dtype_hi` from state tensors.
2. Set `dtype_store` with `_state_storage_dtype(dtype_hi)`.
3. Each step:
   - cast `yn` to `dtype_store` before calling `func`,
   - store `f_k` in `fhistory` as `dtype_store`,
   - store `y_k` in `_StateHistoryBuffer` as `dtype_store`.
4. For convolution:
   - slice history window,
   - cast window items to `dtype_hi`,
   - compute weighted sum with `tensordot`.
5. Update `yn` in high precision state space.


## Backward Predictor Algorithm (Mixed Precision)

### Flow Diagram

```mermaid
flowchart TD
    A[Init y0, adj_y0, adj_params0] --> B[Select dtype_hi and dtype_store]
    B --> C[Loop k = 0..N-2]
    C --> D[Cast y, adj_y, adj_params -> dtype_store]
    D --> E[Compute func_eval, vjp_y, vjp_params = func(...)]
    E --> F[Store vjp_y in fadj_history (dtype_store)]
    F --> G[Optional: store func_eval in fy_history (dtype_store)]
    G --> H[Build backward weights b_j_k_1 in dtype_hi]
    H --> I[Slice fadj_history window and cast -> dtype_hi]
    I --> J[Vectorized weighted sum via stack + tensordot]
    J --> K[Update adj_y in high precision]
    K --> L[Accumulate adj_params += h * vjp_params]
    L --> M[Update y from yhistory or reconstructed fy_history]
    M --> C
    C --> N[Return adj_y, adj_params]
```

### Key Point

Backward uses the same mixed-precision split as forward:

- low precision where model/VJP evaluation and long history storage dominate runtime/memory,
- high precision where long weighted reductions are numerically sensitive.


## Why This Helps

### 1) Lower History Memory

If stored in FP16/BF16 instead of FP32, history footprints are roughly cut in half.

- Forward history (`fhistory`, `yhistory`) is long-lived.
- Backward history (`fadj_history`, optional `fy_history`) is also long-lived.

This can directly reduce peak GPU memory pressure in long time grids.


### 2) Faster Model Evaluations

Casting state inputs to `dtype_store` before `func(...)` lets Tensor Cores and autocast kernels run faster on supported hardware (especially FP16/BF16 on NVIDIA GPUs).


### 3) Better Numerical Stability Than "All-Low-Precision"

The convolution reduction is promoted to `dtype_hi` before `tensordot`.

That avoids doing long weighted sums entirely in low precision, which typically reduces accumulated rounding error for large memory windows.


## Cost / Tradeoff

The mixed design adds cast traffic and temporary tensors for `stack + tensordot`.

- Benefit side: lower persistent history memory, often faster `func` eval.
- Cost side: per-step window promotion to high precision and temporary stacked buffers.

This is usually favorable when model eval dominates and/or memory pressure is high.


## Practical Expectations

- Best-case: lower memory and better throughput than pure FP32.
- Worst-case: if state tensors are very large spatial maps and memory window is large, stacked temporaries can become expensive.
- Typical compromise in this implementation:
  - store long-term history low precision,
  - reduce weighted sums high precision.


## Implementation Notes

- The predictor code currently uses vectorized window reduction (`stack` + `tensordot`) instead of the older inner Python accumulation loop.
- `dtype_store` tracks active autocast dtype via `_state_storage_dtype(...)`.
- For tuple states, casting is done component-wise using `_cast_state_dtype(...)`.


## Suggested Validation Checks

When changing precision policy, compare against a full-precision baseline:

1. Final loss / validation error drift.
2. Gradient norms (`adj_y`, parameter grads) for NaN/Inf.
3. Peak memory (`torch.cuda.max_memory_allocated`).
4. Train/inference wall time.

Run at multiple memory lengths (`options["memory"]`) and step counts to cover both short and long history regimes.
