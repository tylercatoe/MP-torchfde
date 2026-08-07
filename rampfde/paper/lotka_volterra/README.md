# Fractional Lotka-Volterra Parameter Estimation

Replicates Section 5.1 of "Efficient Training of Neural FDE via Adjoint
Backpropagation" (Kang et al., AAAI 2025, arXiv:2503.16666) — the paper that
names the "basic predictor" (Volterra product-rectangle) scheme implemented
in `rampde.predictor_fdeint`.

## Overview

Fits the 4 free parameters `[a, b, c, d]` of the fractional Lotka-Volterra
system

```
D^β x =  x (a - c y)
D^β y = -y (b - d x)
```

to noisy synthetic trajectories (β=0.7, true params `[1.0, 0.5, 1.0, 0.3]`),
comparing three solver backends on identical data/optimizer:

- **`torchfde_fp32`** — reference: torchfde's L1 scheme, float32, standard autograd.
- **`rampde_fp16`** — `rampde.fdeint`, Gao L1 scheme, mixed precision, continuous-adjoint
  backward approximation.
- **`rampde_predictor_fp16`** — `rampde.predictor_fdeint`, Volterra product-rectangle
  ("basic predictor") scheme, mixed precision, **exact discrete-adjoint** backward.

Metrics compared:
- Final / best mean-absolute parameter error vs ground truth
- Peak GPU memory during training
- Training loss convergence

## Files

- `train_lotka_volterra.py`: main training script (`--solver` selects the backend)
- `compare_results.py`: N-way comparison table across `results.json` files
- `run_test.sh`: quick 100-iteration smoke test (all three solvers)
- `run_experiment.sh`: full 500-iteration experiment (all three solvers)

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: a few minutes total (100 iterations × 3 solvers, small
2-D state).

## Full Experiment

```bash
./run_experiment.sh
```

Override defaults via env vars, e.g. `NITERS=1000 ./run_experiment.sh`.

## Comparing Results

```bash
python compare_results.py \
  results/torchfde_fp32/results.json \
  results/rampde_fp16/results.json \
  results/rampde_predictor_fp16/results.json
```

Also works pairwise with just two paths.

## Notes

- Ground-truth trajectories are always generated with `torchfde` FP32 for a
  reproducible reference, regardless of which solver is being evaluated.
- `rampde_fp16` and `rampde_predictor_fp16` both run under
  `torch.autocast(dtype=torch.float16)` with `loss_scaler=False` and
  `adj_dtype=torch.float16` — i.e. the RHS is evaluated in fp16 and the
  adjoint history buffer is also stored in fp16, matching the memory profile
  described in each solver's module docstring.
- Requires a GPU and the `implicit-oc` conda environment (same as
  `fde_mnist/` and `fde_stl10/`).
