#!/bin/bash
# Quick smoke test (100 iterations) — checks that all three solvers train and
# report parameter-recovery results.
# Run: bash run_test.sh

set -e
cd "$(dirname "$0")"

echo "=== Fractional Lotka-Volterra Quick Test (100 iters) ==="
echo ""

echo "--- [1/3] torchfde FP32 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver torchfde_fp32 \
  --niters 100 \
  --log_freq 20 \
  --save results/test/torchfde_fp32

echo ""
echo "--- [2/3] rampde L1 FP16 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver rampde_fp16 \
  --niters 100 \
  --log_freq 20 \
  --save results/test/rampde_fp16

echo ""
echo "--- [3/3] rampde predictor FP16 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver rampde_predictor_fp16 \
  --niters 100 \
  --log_freq 20 \
  --save results/test/rampde_predictor_fp16

echo ""
echo "=== Test complete. Results in results/test/ ==="
echo "Compare with:"
echo "  python compare_results.py results/test/torchfde_fp32/results.json results/test/rampde_fp16/results.json results/test/rampde_predictor_fp16/results.json"
