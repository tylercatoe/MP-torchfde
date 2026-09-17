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
ENV_NAME="${ENV_NAME:-torch28}"
RESULTS_DIR="${RESULTS_DIR:-results}"

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

echo "=== Lotka--Volterra predictor/predictor-corrector mesh experiment ==="
echo "niters=$NITERS n_train=$NTRAIN n_val=$NVAL noise=$NOISE lr=$LR seed=$SEED env=$ENV_NAME"

for init_regime in near_true worse; do
    for method in predictor predictor-corrector; do
        for precision in fp32 fp16; do
            for mesh in uniform graded; do
                output="${RESULTS_DIR}/${init_regime}/${method}/${mesh}_${precision}"
                echo "--- init=$init_regime mesh=$mesh method=$method precision=$precision ---"
                solver_args=(
                    --init-regime "$init_regime"
                    --mesh "$mesh"
                    --precision "$precision"
                )
                if [ "$method" = "predictor-corrector" ]; then
                    solver_args+=(--predictor_corrector)
                fi
                "${PYTHON_CMD[@]}" train_lotka_volterra.py \
                    "${solver_args[@]}" \
                    "${common_args[@]}" \
                    --save "$output"
            done
        done
    done
done

echo "=== Runs complete; creating comparison files ==="
"${PYTHON_CMD[@]}" compare_results.py \
    --output-dir "$RESULTS_DIR"
