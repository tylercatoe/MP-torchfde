#!/bin/bash
#SBATCH --job-name=mp-fde-stl10
#SBATCH --output=slurm_logs/mp_fde_stl10_%j.out
#SBATCH --error=slurm_logs/mp_fde_stl10_%j.err
#SBATCH --partition=work1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1

set -euo pipefail

module load cuda/12.3.0
module load anaconda3/2023.09-0
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$project_root/slurm_logs"
cd "$project_root"

ENV_NAME="${ENV_NAME:-torch28}"
MODE="${MODE:-direct}"   # direct | adjoint | adjoint-mixed | adjoint-mixed-bfloat
EPOCHS="${EPOCHS:-160}"
SAVE_ROOT="${SAVE_ROOT:-exp_mp_stl10}"
DATA_ROOT="${DATA_ROOT:-.data/stl10}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"

ADJOINT_METHOD="${ADJOINT_METHOD:-predictor-f}"
DIRECT_METHOD="${DIRECT_METHOD:-predictor}"
MP_DTYPE="${MP_DTYPE:-float16}"
MP_LOSS_SCALER="${MP_LOSS_SCALER:-auto}"

WIDTH="${WIDTH:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
BETA="${BETA:-0.6}"
T_FINAL="${T_FINAL:-1.0}"
STEP_SIZE="${STEP_SIZE:-0.1}"
TRAIN_SIZE="${TRAIN_SIZE:-4000}"
TIME_BINS="${TIME_BINS:-4}"
UNSTABLE="${UNSTABLE:-0}"

ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"
if [ -d "${ENV_PATH}/lib" ]; then
  export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
CUDNN_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

# Resolve driver path from either repo root submit or torchfde root submit.
driver_script="torchfde/examples/STL10/mp_fde_stl10.py"
pythonpath_root="$project_root/torchfde"
if [ ! -f "$driver_script" ]; then
  driver_script="examples/STL10/mp_fde_stl10.py"
  pythonpath_root="$project_root"
fi

if [ ! -f "$driver_script" ]; then
  echo "ERROR: Could not find mp_fde_stl10.py from submit dir: $project_root"
  echo "Expected either:"
  echo "  - torchfde/examples/STL10/mp_fde_stl10.py"
  echo "  - examples/STL10/mp_fde_stl10.py"
  exit 1
fi

case "$MODE" in
  direct|adjoint|adjoint-mixed|adjoint-mixed-bfloat)
    ;;
  *)
    echo "ERROR: MODE must be one of: direct, adjoint, adjoint-mixed, adjoint-mixed-bfloat"
    exit 1
    ;;
esac

export PYTHONPATH="$pythonpath_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_ROOT

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Submit dir: $project_root"
echo "Conda env: $ENV_NAME"
echo "Driver: $driver_script"
echo "Mode: $MODE"
echo "Epochs: $EPOCHS"
echo "Data root: $DATA_ROOT"

# Optional one-job STL10 download setup.
if [ "$DOWNLOAD_DATA" = "1" ]; then
  conda run -n "$ENV_NAME" python - <<'PY'
import os
from torchvision.datasets import STL10

data_root = os.environ["DATA_ROOT"]
os.makedirs(data_root, exist_ok=True)
STL10(root=data_root, split="train", download=True)
print(f"STL10 ready at {data_root}")
PY
fi

download_arg="--no-download-data"
if [ "$DOWNLOAD_DATA" = "1" ]; then
  download_arg="--download-data"
fi

unstable_arg=""
if [ "$UNSTABLE" = "1" ]; then
  unstable_arg="--unstable"
fi

conda run -n "$ENV_NAME" python -u "$driver_script" \
  --modes "$MODE" \
  --nepochs "$EPOCHS" \
  --save "$SAVE_ROOT" \
  --data-root "$DATA_ROOT" \
  "$download_arg" \
  --adjoint_method "$ADJOINT_METHOD" \
  --direct_method "$DIRECT_METHOD" \
  --mp_dtype "$MP_DTYPE" \
  --mp_loss_scaler "$MP_LOSS_SCALER" \
  --beta "$BETA" \
  --T "$T_FINAL" \
  --step_size "$STEP_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --weight_decay "$WEIGHT_DECAY" \
  --width "$WIDTH" \
  --train-size "$TRAIN_SIZE" \
  --time-bins "$TIME_BINS" \
  $unstable_arg
