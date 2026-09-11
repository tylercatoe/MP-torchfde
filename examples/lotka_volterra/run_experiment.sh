#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NITERS="${NITERS:-500}"
LOG_FREQ="${LOG_FREQ:-25}"
NTRAIN="${NTRAIN:-50}"
NVAL="${NVAL:-25}"
NOISE="${NOISE:-0.05}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-implicit-oc}"

if command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" python)
else
    echo "conda was not found; using the currently active Python environment."
    PYTHON_CMD=(python)
fi

common_args=(
    --niters "$NITERS"
    --log_freq "$LOG_FREQ"
    --n_train "$NTRAIN"
    --n_val "$NVAL"
    --noise_std "$NOISE"
    --lr "$LR"
    --seed "$SEED"
    --gpu "$GPU"
)

echo "=== Lotka--Volterra uniform/graded predictor-corrector experiment ==="
echo "niters=$NITERS n_train=$NTRAIN n_val=$NVAL noise=$NOISE lr=$LR seed=$SEED"

for precision in fp32 fp16; do
    for mesh in uniform graded; do
        output="results/${mesh}_${precision}"
        echo "--- mesh=$mesh precision=$precision ---"
        "${PYTHON_CMD[@]}" train_lotka_volterra.py \
            --mesh "$mesh" \
            --precision "$precision" \
            "${common_args[@]}" \
            --save "$output"
    done
done

echo "=== Runs complete; creating comparison files ==="
"${PYTHON_CMD[@]}" compare_results.py \
    --output-dir results \
    results/*/results.json
