#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_mnist.sbatch"

if [ ! -f "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found: $sbatch_script"
  exit 1
fi

mode="${1:-full}"   # full | pilot

if [ "$mode" = "pilot" ]; then
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_mnist_pilot}"
  echo "Submitting PILOT jobs with epochs=$epochs"
else
  epochs="${FULL_EPOCHS:-100}"
  save_root="${SAVE_ROOT:-exp_mp_mnist}"
  echo "Submitting FULL jobs with epochs=$epochs"
fi

mkdir -p slurm_logs

echo "Submitting direct..."
sbatch --job-name=mp-fde-direct \
  --export=ALL,MODE=direct,EPOCHS="$epochs",SAVE_ROOT="$save_root/direct",DOWNLOAD_DATA=0,DATA_ROOT=data/mnist \
  "$sbatch_script"

echo "Submitting adjoint..."
sbatch --job-name=mp-fde-adjoint \
  --export=ALL,MODE=adjoint,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint",DOWNLOAD_DATA=0,DATA_ROOT=data/mnist \
  "$sbatch_script"

echo "Submitting adjoint-mixed..."
sbatch --job-name=mp-fde-adjmix \
  --export=ALL,MODE=adjoint-mixed,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed",DOWNLOAD_DATA=0,DATA_ROOT=data/mnist \
  "$sbatch_script"
sbatch --job-name=mp-fde-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat",DOWNLOAD_DATA=0,DATA_ROOT=data/mnist \
  "$sbatch_script"
echo "Submitted 4 jobs (direct, adjoint, adjoint-mixed, adjoint-mixed-bfloat)."

