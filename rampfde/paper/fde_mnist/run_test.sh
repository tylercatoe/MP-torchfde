#!/bin/bash
# Quick 3-epoch smoke test — checks that both solvers train and report results.
# Run: bash run_test.sh

set -e
cd "$(dirname "$0")"

echo "=== FDE MNIST Quick Test (3 epochs) ==="
echo ""

echo "--- torchfde FP32 ---"
conda run -n implicit-oc python train_fde_mnist.py \
  --solver torchfde_fp32 \
  --nepochs 3 \
  --batch_size 64 \
  --test_batch_size 128 \
  --T 1.0 \
  --step_size 0.1 \
  --beta 0.7 \
  --save results/test/torchfde_fp32

echo ""
echo "--- rampde FP16 ---"
conda run -n implicit-oc python train_fde_mnist.py \
  --solver rampde_fp16 \
  --nepochs 3 \
  --batch_size 64 \
  --test_batch_size 128 \
  --T 1.0 \
  --step_size 0.1 \
  --beta 0.7 \
  --save results/test/rampde_fp16

echo ""
echo "=== Test complete. Results in results/test/ ==="
