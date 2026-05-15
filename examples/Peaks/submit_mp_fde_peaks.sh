#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"

sbatch_script=""
for candidate in \
  "$script_dir/train_mp_fde_peaks.sbatch" \
  "$submit_dir/train_mp_fde_peaks.sbatch" \
  "$submit_dir/torchfde/examples/Peaks/train_mp_fde_peaks.sbatch" \
  "$submit_dir/examples/Peaks/train_mp_fde_peaks.sbatch"
do
  if [ -f "$candidate" ]; then
    sbatch_script="$candidate"
    break
  fi
done

if [ -z "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found. Tried:"
  echo "  $script_dir/train_mp_fde_peaks.sbatch"
  echo "  $submit_dir/train_mp_fde_peaks.sbatch"
  echo "  $submit_dir/torchfde/examples/Peaks/train_mp_fde_peaks.sbatch"
  echo "  $submit_dir/examples/Peaks/train_mp_fde_peaks.sbatch"
  exit 1
fi

run_size="${1:-pilot}"  # pilot | full

if [ "$run_size" = "full" ]; then
  epochs="${FULL_EPOCHS:-160}"
  save_root="${SAVE_ROOT:-exp_mp_peaks}"
  echo "Submitting FULL run (epochs=$epochs)"
else
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_peaks_pilot}"
  echo "Submitting PILOT run (epochs=$epochs)"
fi

mkdir -p slurm_logs

echo "Submitting direct..."
job_direct=$(sbatch --parsable --job-name=mp-peaks-direct \
  --export=ALL,MODE=direct,EPOCHS="$epochs",SAVE_ROOT="$save_root/direct" \
  "$sbatch_script")
echo "  job_id=$job_direct"

echo "Submitting adjoint..."
job_adjoint=$(sbatch --parsable --job-name=mp-peaks-adjoint \
  --export=ALL,MODE=adjoint,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint" \
  "$sbatch_script")
echo "  job_id=$job_adjoint"

echo "Submitting adjoint-mixed..."
job_adjmix=$(sbatch --parsable --job-name=mp-peaks-adjmix \
  --export=ALL,MODE=adjoint-mixed,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed" \
  "$sbatch_script")
echo "  job_id=$job_adjmix"

echo "Submitting adjoint-mixed-bfloat..."
job_adjmix_bf=$(sbatch --parsable --job-name=mp-peaks-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat" \
  "$sbatch_script")
echo "  job_id=$job_adjmix_bf"

echo "Submitted 4 Peaks jobs in parallel."
