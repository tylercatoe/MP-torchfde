# MNIST Mixed-Precision Matrix (Fractional Adjoint)

This document explains `examples/mnist_mixed_precision_matrix.py`.

It runs MNIST training under multiple precision/scaler configurations, prints one comparison table, and can save a plot of test accuracy across epochs.

## Configurations Included

- `fp32_unscaled`
- `bf16_unscaled` (when `torch.cuda.is_bf16_supported()` is true)
- `fp16_safe` (`loss_scaler=False`)
- `fp16_dynamic` (`loss_scaler=DynamicScaler(torch.float16)`)

The script uses `fdeint_adjoint` and explicitly forwards `loss_scaler` through a mixed-precision-aware FDE block, so `fp16_safe` and `fp16_dynamic` are truly different runs.

## What It Reports

For each configuration:

- `final_train_loss`
- `final_train_acc`
- `final_test_loss`
- `final_test_acc`
- `best_test_acc`
- `nan_inf_events`
- `mean_epoch_s`
- `train_samples_per_s`
- `peak_mem_mib`
- `dynamic_scale_steps`

## Colab Commands

```bash
# clone or refresh
git clone https://github.com/tylercatoe/MP-torchfde.git
cd MP-torchfde
# if already cloned:
# git pull origin main

pip install -e .
```

Quick smoke run:

```bash
python examples/mnist_mixed_precision_matrix.py \
  --device cuda \
  --epochs 1 \
  --max-train-batches 120 \
  --max-test-batches 40 \
  --batch-size 256 \
  --test-batch-size 512 \
  --plot-out mnist_mp_smoke_test_acc.png \
  --history-json-out mnist_mp_smoke_history.json \
  --json-out mnist_mp_smoke.json \
  --csv-out mnist_mp_smoke.csv
```

Larger run:

```bash
python examples/mnist_mixed_precision_matrix.py \
  --device cuda \
  --epochs 3 \
  --batch-size 256 \
  --test-batch-size 512 \
  --plot-out mnist_mp_full_test_acc.png \
  --history-json-out mnist_mp_full_history.json \
  --json-out mnist_mp_full.json \
  --csv-out mnist_mp_full.csv
```

## Key Arguments

- `--method` (default `predictor-f`)
- `--beta` (default `0.9`)
- `--t-final` (default `2.0`)
- `--step-size` (default `0.1`)
- `--memory` (default `-1`)
- `--network {odenet,resnet}` (default `odenet`)
- `--downsampling-method {conv,res}` (default `conv`)
- `--data-aug` / `--no-data-aug`
- `--max-train-batches` and `--max-test-batches` for fast debug runs
- `--plot-out <png_path>` saves a line plot of test accuracy vs epoch (one line per configuration)
- `--history-json-out <json_path>` saves per-epoch train/test metrics for each configuration

## Recommended Interpretation

1. First require `status == ok` and `nan_inf_events == 0`.
2. Compare `final_test_acc` against `fp32_unscaled`.
3. Among acceptable accuracy configurations, choose the fastest (`mean_epoch_s`) and/or lowest memory (`peak_mem_mib`).

For the accuracy plot:

- X-axis is epoch number.
- Y-axis is test accuracy.
- Each line is one configuration (`fp32`, `bf16`, `fp16_safe`, `fp16_dynamic`).
