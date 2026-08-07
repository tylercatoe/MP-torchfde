# Neural FDE MNIST

Trains a CNN + FDE-block classifier on MNIST, comparing three solver
backends for the fractional-order block on identical architecture/data/
optimizer.

## Overview

Architecture: downsampling CNN stem → FDE block (`D^β z = f(t, z)`) →
classifier head. Reports per epoch:
- Test accuracy
- Peak GPU memory
- Eval time

Solvers compared:
- **`torchfde_fp32`** — reference: torchfde's L1 scheme, float32, standard autograd.
- **`rampde_fp16`** — `rampde.fdeint`, Gao L1 scheme, mixed precision, continuous-adjoint
  backward approximation.
- **`rampde_predictor_fp16`** — `rampde.predictor_fdeint`, Volterra product-rectangle
  ("basic predictor") scheme, mixed precision, **exact discrete-adjoint** backward.

## Files

- `train_fde_mnist.py`: main training script (`--solver` selects the backend)
- `compare_results.py`: N-way comparison table across `results.json` files
- `run_test.sh`: quick 3-epoch smoke test (all three solvers)
- `run_experiment.sh`: full 30-epoch experiment (all three solvers)

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: ~15-20 minutes total (3 epochs × 3 solvers).

## Full Experiment

```bash
./run_experiment.sh
```

Override defaults via env vars, e.g. `NEPOCHS=50 BETA=0.5 ./run_experiment.sh`.

## Comparing Results

```bash
python compare_results.py \
  results/torchfde_fp32_b0.7_T2.0/results.json \
  results/rampde_fp16_b0.7_T2.0/results.json \
  results/rampde_predictor_fp16_b0.7_T2.0/results.json
```

Also works pairwise with just two paths.

## Notes

- `rampde_fp16` and `rampde_predictor_fp16` both run under
  `torch.autocast(dtype=torch.float16)` with `loss_scaler=False` and
  `adj_dtype=torch.float16` — the RHS is evaluated in fp16 and the adjoint
  history buffer is stored in fp16.
- Requires a GPU and the `implicit-oc` conda environment.
- `results/` and `data/` are experiment outputs / downloaded MNIST — not
  committed (see repo `.gitignore`).
