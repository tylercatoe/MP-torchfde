#!/bin/bash
# Full 30-epoch experiment comparing torchfde FP32 vs rampde FP16 on MNIST.
# Usage: bash run_experiment.sh [--nepochs 30] [--beta 0.7] [--T 2.0]

set -e
cd "$(dirname "$0")"

NEPOCHS=${NEPOCHS:-30}
BETA=${BETA:-0.7}
T=${T:-2.0}
STEP=${STEP:-0.1}
DIM=${DIM:-64}
BATCH=${BATCH:-128}

echo "=== FDE MNIST Experiment ==="
echo "    β=${BETA}  T=${T}  h=${STEP}  N=$((${T%.*} * 10 + 1))  epochs=${NEPOCHS}"
echo ""

echo "--- [1/2] torchfde FP32 ---"
conda run -n implicit-oc python train_fde_mnist.py \
  --solver torchfde_fp32 \
  --nepochs "$NEPOCHS" \
  --batch_size "$BATCH" \
  --T "$T" --step_size "$STEP" --beta "$BETA" \
  --dim "$DIM" \
  --save "results/torchfde_fp32_b${BETA}_T${T}"

echo ""
echo "--- [2/2] rampde FP16 ---"
conda run -n implicit-oc python train_fde_mnist.py \
  --solver rampde_fp16 \
  --nepochs "$NEPOCHS" \
  --batch_size "$BATCH" \
  --T "$T" --step_size "$STEP" --beta "$BETA" \
  --dim "$DIM" \
  --save "results/rampde_fp16_b${BETA}_T${T}"

echo ""
echo "=== Both runs complete. Compare with: ==="
echo "  python compare_results.py results/torchfde_fp32_b${BETA}_T${T}/results.json results/rampde_fp16_b${BETA}_T${T}/results.json"
