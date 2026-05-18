#!/bin/bash
#SBATCH --job-name=mp-fde-peaks
#SBATCH --output=slurm_logs/mp_fde_peaks_%j.out
#SBATCH --error=slurm_logs/mp_fde_peaks_%j.err
#SBATCH --partition=work1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1

set -euo pipefail

module load cuda/12.3.0WIDT
module load anaconda3/2023.09-0
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
c
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$project_root/slurm_logs"
cd "$project_root"

ENV_NAME="${ENV_NAME:-torch28}"
MODE="${MODE:-direct}"  # direct | adjoint | adjoint-mixed | adjoint-mixed-bfloat
EPOCHS="${EPOCHS:-160}"
SAVE_ROOT="${SAVE_ROOT:-exp_mp_peaks}"

ADJOINT_METHOD="${ADJOINT_METHOD:-predictor-f}"
DIRECT_METHOD="${DIRECT_METHOD:-predictor}"
MP_DTYPE="${MP_DTYPE:-float16}"
MP_LOSS_SCALER="${MP_LOSS_SCALER:-auto}"

WIDTH="${WIDTH:-64}"
NUM_LAYERS="${NUM_LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-20000}"
TEST_SPLIT="${TEST_SPLIT:-0.2}"
LR="${LR:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SEED="${SEED:-42}"
BETA="${BETA:-0.5}"
T_FINAL="${T_FINAL:-1.0}"
STEP_SIZE="${STEP_SIZE:-0.1}"
MEMORY="${MEMORY:--1}"

driver_script="torchfde/examples/Peaks/mp_fde_peaks.py"
pythonpath_root="$project_root/torchfde"
if [ ! -f "$driver_script" ]; then
  driver_script="examples/Peaks/mp_fde_peaks.py"
  pythonpath_root="$project_root"
fi

if [ ! -f "$driver_script" ]; then
  echo "ERROR: Could not find mp_fde_peaks.py from submit dir: $project_root"
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

echo "Job ID: $SLURM_JOB_ID"
echo "Submit dir: $project_root"
echo "Conda env: $ENV_NAME"
echo "Driver: $driver_script"
echo "Mode: $MODE"
echo "Epochs: $EPOCHS"
echo "Save root: $SAVE_ROOT"

conda run -n "$ENV_NAME" python -u "$driver_script" \
  --mode "$MODE" \
  --nepochs "$EPOCHS" \
  --save "$SAVE_ROOT" \
  --adjoint_method "$ADJOINT_METHOD" \
  --direct_method "$DIRECT_METHOD" \
  --mp_dtype "$MP_DTYPE" \
  --mp_loss_scaler "$MP_LOSS_SCALER" \
  --width "$WIDTH" \
  --num_layers "$NUM_LAYERS" \
  --batch_size "$BATCH_SIZE" \
  --num_samples "$NUM_SAMPLES" \
  --test_split "$TEST_SPLIT" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --beta "$BETA" \
  --T "$T_FINAL" \
  --step_size "$STEP_SIZE" \
  --memory "$MEMORY" \
  --gpu 0
