#!/bin/bash
#SBATCH --job-name=mp-fde-mnist
#SBATCH --output=slurm_logs/mp_fde_mnist_%j.out
#SBATCH --error=slurm_logs/mp_fde_mnist_%j.err
#SBATCH --partition=work1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1

set -euo pipefail

# Match your known-good Palmetto style setup.
module load cuda/12.3.0
module load anaconda3/2023.09-0
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$project_root/slurm_logs"
cd "$project_root"

# Choose conda env (override by exporting ENV_NAME before sbatch).
ENV_NAME="${ENV_NAME:-torch28}"

# Run controls (override with: sbatch --export=ALL,MODE=direct,EPOCHS=5 ...)
MODE="${MODE:-all}"                # direct | adjoint | adjoint-mixed | adjoint-mixed-bfloat | all
EPOCHS="${EPOCHS:-160}"
SAVE_ROOT="${SAVE_ROOT:-exp_mp_mnist}"
ADJOINT_METHOD="${ADJOINT_METHOD:-predictor-f}"
MP_DTYPE="${MP_DTYPE:-float16}"
MP_LOSS_SCALER="${MP_LOSS_SCALER:-auto}"
DATA_ROOT="${DATA_ROOT:-data/mnist}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"   # 1 to pre-download in this job, 0 to skip

# Optional cuDNN path injection (same pattern as your previous H200 jobs).
ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"
if [ -d "${ENV_PATH}/lib" ]; then
  export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
CUDNN_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

# Resolve driver path for either repo root submit or torchfde root submit.
driver_script="torchfde/examples/mp_fde_mnist.py"
pythonpath_root="$project_root/torchfde"
if [ ! -f "$driver_script" ]; then
  driver_script="examples/mp_fde_mnist.py"
  pythonpath_root="$project_root"
fi

if [ ! -f "$driver_script" ]; then
  echo "ERROR: Could not find mp_fde_mnist.py from submit dir: $project_root"
  echo "Expected either:"
  echo "  - torchfde/examples/mp_fde_mnist.py"
  echo "  - examples/mp_fde_mnist.py"
  exit 1
fi

case "$MODE" in
  direct|adjoint|adjoint-mixed|adjoint-mixed-bfloat)
    MODES=("$MODE")
    ;;
  all)
    MODES=(direct adjoint adjoint-mixed adjoint-mixed-bfloat)
    ;;
  *)
    echo "ERROR: MODE must be one of: direct, adjoint, adjoint-mixed, adjoint-mixed-bfloat, all"
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
echo "PYTHONPATH: $PYTHONPATH"

if [ "$DOWNLOAD_DATA" = "1" ]; then
  echo "Preparing MNIST dataset at: $DATA_ROOT"
  conda run -n "$ENV_NAME" python - <<'PY'
import os
from torchvision import datasets

data_root = os.environ["DATA_ROOT"]
os.makedirs(data_root, exist_ok=True)

lock_path = os.path.join(data_root, ".mnist_download.lock")
try:
    import fcntl
except ImportError:
    fcntl = None

if fcntl is None:
    datasets.MNIST(root=data_root, train=True, download=True)
    datasets.MNIST(root=data_root, train=False, download=True)
else:
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            datasets.MNIST(root=data_root, train=True, download=True)
            datasets.MNIST(root=data_root, train=False, download=True)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

print(f"MNIST ready at {data_root}")
PY
fi

download_arg="--download-data"
if [ "$DOWNLOAD_DATA" != "1" ]; then
  download_arg="--no-download-data"
fi

conda run -n "$ENV_NAME" python -u "$driver_script" \
  --modes "${MODES[@]}" \
  --nepochs "$EPOCHS" \
  --save "$SAVE_ROOT" \
  --data-root "$DATA_ROOT" \
  "$download_arg" \
  --adjoint-method "$ADJOINT_METHOD" \
  --mp-dtype "$MP_DTYPE" \
  --mp-loss-scaler "$MP_LOSS_SCALER" \
  --beta 0.5 \
  --T 5 \
  --step_size 0.1 \
  --batch_size 128 \
  --test_batch_size 128 \
  --downsampling-method conv
