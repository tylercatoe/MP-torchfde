#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_mnist.sbatch"

if [ ! -f "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found: $sbatch_script"
  exit 1
fi

run_size="${1:-full}"   # full | pilot

if [ "$run_size" = "pilot" ]; then
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
job_direct=$(sbatch --parsable --job-name=mp-mnist-direct \
  --export=ALL,MODE=direct,EPOCHS="$epochs",SAVE_ROOT="$save_root/direct" \
  "$sbatch_script")
echo "  job_id=$job_direct"

echo "Submitting adjoint..."
job_adj=$(sbatch --parsable --job-name=mp-mnist-adjoint \
  --export=ALL,MODE=adjoint,DTYPE_HI=float32,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint" \
  "$sbatch_script")
echo "  job_id=$job_adj"

echo "Submitting adjoint-mixed..."
job_adjmix=$(sbatch --parsable --job-name=mp-mnist-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE=float16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed" \
  "$sbatch_script")
echo "  job_id=$job_adjmix"

echo "Submitting adjoint-mixed-bfloat..."
job_adjmix_bf16=$(sbatch --parsable --job-name=mp-mnist-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE=bfloat16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat" \
  "$sbatch_script")
echo "  job_id=$job_adjmix_bf16"

echo "Submitted 4 jobs in parallel."
