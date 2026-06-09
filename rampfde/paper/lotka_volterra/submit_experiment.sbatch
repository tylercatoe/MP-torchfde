#!/bin/bash
#SBATCH --job-name=fde-LV
#SBATCH --partition=work1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h200:1

set -euo pipefail

module load cuda/12.3.0
module load anaconda3/2023.09-0

eval "$(conda shell.bash hook)"

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$project_root/slurm_logs"
cd "$project_root"

ENV_NAME="${ENV_NAME:-torch28}"
#SAVE_ROOT="${SAVE_ROOT:-exp_LV}"

ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"
if [ -d "${ENV_PATH}/lib" ]; then
  export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
CUDNN_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

driver_script="rampfde/paper/lotka_volterra/train_lotka_volterra.py"
pythonpath_root="$project_root/torchfde"
if [ ! -f "$driver_script" ]; then
  driver_script="paper/lotka_volterra/train_lotka_volterra.py"
  pythonpath_root="$project_root"
fi

if [ ! -f "$driver_script" ]; then
  echo "ERROR: Could not find train_lotka_volterra.py from submit dir: $project_root"
  echo "Expected either:"
  echo "  - torchfde/paper/lotka_volterra/train_lotka_volterra.py"
  echo "  - paper/lotka_volterra/train_lotka_volterra.py"
  exit 1
fi

export PYTHONPATH="$pythonpath_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


NITERS="${NITERS:-500}"
SOLVER="${SOLVER:-rampde_fp16}"

case "$SOLVER" in
  torchfde_fp32|rampde_fp16)
    ;;
  *)
    echo "ERROR: SOLVER must be torchfde_fp32 or rampde_fp16"
    exit 1
    ;;
esac


# echo ""
# echo "Job ID: $SLURM_JOB_ID"
# echo "Running on nodes: $SLURM_NODELIST"
# echo "Submit dir: $project_root"
# echo "Conda env: $ENV_NAME"
# echo "Driver: $driver_script"
# echo "Iterations: $NITERS"
# echo "Save root: $SAVE_ROOT"
# echo ""

echo "--- Running ${SOLVER} ---"
conda run -n "$ENV_NAME" python -u "$driver_script" \
  --solver "$SOLVER" \
  --niters "$NITERS" \
  --save "exp_LV/${SOLVER}_niters${NITERS}"
