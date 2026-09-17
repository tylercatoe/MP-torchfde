# Lotka--Volterra mesh comparison

This example estimates the four positive parameters in the fractional
Lotka--Volterra system

```text
D^β x = x (a - c y)
D^β y = -y (b - d x)
```

It compares the predictor and predictor-corrector methods on a uniform mesh
and the double-graded mesh, in both FP32 and FP16, from two parameter
initializations: `near_true` and `worse`. Within each initialization regime,
all runs use the same optimizer, beta, data, and train/validation split. The
experimental factors are initialization, numerical method, mesh, and precision.

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

The launcher runs all 16 cases: two initializations, two numerical methods,
two meshes, and two precisions. Its directory layout is:

```text
<init_regime>/<method>/<mesh>_<precision>/results.json
```

The `near_true` initialization is `[0.99, 0.48, 1.05, 0.33]`; the `worse`
initialization is `[0.65, 0.75, 1.35, 0.18]`. The true parameters are
`[1.0, 0.5, 1.0, 0.3]`.

`<method>` is either `predictor` or `predictor-corrector`, and `RESULTS_DIR`
defaults to `results`.
The launcher also creates the following files under `RESULTS_DIR`:

- `comparison.csv`, which is convenient for pandas or custom plots;
- `comparison.md`, a full summary with metrics, memory comparisons,
  experimental parameters, learned parameters, notes, and embedded plots;
- `validation_loss_vs_iteration.png`, faceted convergence curves;
- `validation_loss_vs_time.png`, the same curves against estimated
  elapsed time;
- `parameter_trajectories.png`, the learned `a`, `b`, `c`, and `d`
  trajectories with their true values;
- `relative_parameter_error_vs_iteration.png`, relative L2 parameter error
  on a logarithmic scale;
- `accuracy_cost_tradeoff.png`, best validation loss versus runtime and
  peak GPU memory.

The launcher uses the `torch28` conda environment by default. Override it with
`ENV_NAME=your_environment`. If `conda` is unavailable but the desired
environment is already active, it falls back to `python`.

Useful overrides include:

```bash
NITERS=1000 NTRAIN=100 NVAL=50 GPU=1 ./run_experiment.sh
```

To regenerate the comparison files later:

```bash
python compare_results.py --output-dir results
```

## Interpretation

`final_val_loss` and `best_val_loss` measure trajectory prediction on held-out
initial conditions. `final_param_err` measures mean absolute error from the
known parameters `[1.0, 0.5, 1.0, 0.3]`. Comparing `near_true` with `worse`
shows whether mesh or precision effects depend on optimization difficulty.
`peak_mem_mb` is the maximum allocated GPU memory recorded during a training
iteration, and the logged iteration time is a rough runtime comparison. Since
the fractional solve is full-batch, iterations are the natural training-step
unit rather than data epochs.
