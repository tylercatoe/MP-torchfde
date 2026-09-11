# Lotka--Volterra mesh comparison

This example estimates the four positive parameters in the fractional
Lotka--Volterra system

```text
D^β x = x (a - c y)
D^β y = -y (b - d x)
```

It compares a uniform mesh with the double-graded mesh, in both FP32 and
FP16. All four runs use the same predictor--corrector recurrence, optimizer,
initial parameter values, beta, data, and train/validation split. The only
experimental factors are mesh and precision.

The synthetic targets are generated once per run from the same seeded initial
conditions using a finer uniform predictor--corrector solve, then independent
fixed Gaussian noise. The model is trained on terminal observations from 50
trajectories and evaluated on 25 held-out trajectories.

## Quick smoke test

From this directory on a GPU node:

```bash
NITERS=10 LOG_FREQ=5 ./run_experiment.sh
```

## Full experiment

```bash
./run_experiment.sh
```

The launcher runs:

```text
uniform_fp32, graded_fp32, uniform_fp16, graded_fp16
```

Results are stored in `results/<mesh>_<precision>/results.json`. The launcher
also creates:

- `results/comparison.csv`, which is convenient for pandas or custom plots;
- `results/comparison.md`, a compact summary table.

The launcher uses the `implicit-oc` conda environment by default. Override it
with `ENV_NAME=your_environment`. If `conda` is unavailable but the desired
environment is already active, it falls back to `python`.

Useful overrides include:

```bash
NITERS=1000 NTRAIN=100 NVAL=50 GPU=1 ./run_experiment.sh
```

To regenerate the comparison files later:

```bash
python compare_results.py --output-dir results results/*/results.json
```

## Interpretation

`final_val_loss` and `best_val_loss` measure trajectory prediction on held-out
initial conditions. `final_param_err` measures mean absolute error from the
known parameters `[1.0, 0.5, 1.0, 0.3]`. `peak_mem_mb` is the maximum allocated
GPU memory recorded during a training iteration, and the logged iteration time
is a rough runtime comparison. Since the fractional solve is full-batch,
iterations are the natural training-step unit rather than data epochs.
