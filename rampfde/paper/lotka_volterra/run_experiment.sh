#!/bin/bash
# Full experiment comparing torchfde FP32 vs rampde L1 FP16 vs rampde predictor
# FP16 on fractional Lotka-Volterra parameter estimation (Kang et al. AAAI 2025,
# Section 5.1 setup).
# Usage: bash run_experiment.sh [--niters 500] [--n_traj 50]

set -e
cd "$(dirname "$0")"

NITERS=${NITERS:-500}
NTRAJ=${NTRAJ:-50}
NOISE=${NOISE:-0.05}
LR=${LR:-0.01}
SEED=${SEED:-42}

echo "=== Fractional Lotka-Volterra Experiment ==="
echo "    niters=${NITERS}  n_traj=${NTRAJ}  noise_std=${NOISE}  lr=${LR}  seed=${SEED}"
echo ""

echo "--- [1/3] torchfde FP32 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver torchfde_fp32 \
  --niters "$NITERS" --n_traj "$NTRAJ" --noise_std "$NOISE" --lr "$LR" --seed "$SEED" \
  --save "results/torchfde_fp32"

echo ""
echo "--- [2/3] rampde L1 FP16 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver rampde_fp16 \
  --niters "$NITERS" --n_traj "$NTRAJ" --noise_std "$NOISE" --lr "$LR" --seed "$SEED" \
  --save "results/rampde_fp16"

echo ""
echo "--- [3/3] rampde predictor FP16 ---"
conda run -n implicit-oc python train_lotka_volterra.py \
  --solver rampde_predictor_fp16 \
  --niters "$NITERS" --n_traj "$NTRAJ" --noise_std "$NOISE" --lr "$LR" --seed "$SEED" \
  --save "results/rampde_predictor_fp16"

echo ""
echo "=== All three runs complete. Compare with: ==="
echo "  python compare_results.py results/torchfde_fp32/results.json results/rampde_fp16/results.json results/rampde_predictor_fp16/results.json"
