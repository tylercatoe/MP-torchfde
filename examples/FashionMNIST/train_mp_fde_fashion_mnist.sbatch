#!/bin/bash
#SBATCH --job-name=mp-fde-fashion-mnist
#SBATCH --output=slurm_logs/mp_fde_fashion_mnist_%j.out
#SBATCH --error=slurm_logs/mp_fde_fashion_mnist_%j.err
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
SAVE_ROOT="${SAVE_ROOT:-exp_mp_fashion_mnist}"

ADJOINT_METHOD="${ADJOINT_METHOD:-predictor-f}"
DIRECT_METHOD="${DIRECT_METHOD:-predictor}"
DTYPE_HI="${DTYPE_HI:-float32}"
MP_DTYPE="${MP_DTYPE:-float32}"
MP_LOSS_SCALER="${MP_LOSS_SCALER:-auto}"

WIDTH="${WIDTH:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
LR="${LR:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SEED="${SEED:-25}"
BETA="${BETA:-0.3}"
T_FINAL="${T_FINAL:-1.0}"
STEP_SIZE="${STEP_SIZE:-0.1}"
MEMORY="${MEMORY:--1}"
DATA_AUG="${DATA_AUG:-0}"
BENCHMARK_ONLY="${BENCHMARK_ONLY:-0}"

ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"
if [ -d "${ENV_PATH}/lib" ]; then
  export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
CUDNN_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

# Resolve driver path from either repo root submit or torchfde root submit.
driver_script="torchfde/examples/FashionMNIST/mp_fde_fashion_mnist.py"
pythonpath_root="$project_root/torchfde"
if [ ! -f "$driver_script" ]; then
  driver_script="examples/FashionMNIST/mp_fde_fashion_mnist.py"
  pythonpath_root="$project_root"
fi

if [ ! -f "$driver_script" ]; then
  echo "ERROR: Could not find mp_fde_fashion_mnist.py from submit dir: $project_root"
  echo "Expected either:"
  echo "  - torchfde/examples/FashionMNIST/mp_fde_fashion_mnist.py"
  echo "  - examples/FashionMNIST/mp_fde_fashion_mnist.py"
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

data_aug_arg=""
if [ "$DATA_AUG" = "1" ]; then
  data_aug_arg="--data-aug"
fi

benchmark_only_arg=""
if [ "$BENCHMARK_ONLY" = "1" ]; then
  benchmark_only_arg="--benchmark_only"
fi

export PYTHONPATH="$pythonpath_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Submit dir: $project_root"
echo "Conda env: $ENV_NAME"
echo "Driver: $driver_script"
echo "Mode: $MODE"
echo "Epochs: $EPOCHS"

conda run -n "$ENV_NAME" python -u "$driver_script" \
  --mode "$MODE" \
  --nepochs "$EPOCHS" \
  --save "$SAVE_ROOT" \
  --adjoint_method "$ADJOINT_METHOD" \
  --direct_method "$DIRECT_METHOD" \
  --dtype_hi "$DTYPE_HI" \
  --mp_dtype "$MP_DTYPE" \
  --mp_loss_scaler "$MP_LOSS_SCALER" \
  --beta "$BETA" \
  --T "$T_FINAL" \
  --step_size "$STEP_SIZE" \
  --memory "$MEMORY" \
  --batch_size "$BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --width "$WIDTH" \
  --seed "$SEED" \
  --gpu 0 \
  $benchmark_only_arg \
  $data_aug_arg
