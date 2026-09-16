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
  epochs="${FULL_EPOCHS:-5000}"
  save_root="${SAVE_ROOT:-exp_mp_peaks}"
  echo "Submitting FULL run (epochs=$epochs)"
else
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_peaks_pilot}"
  echo "Submitting PILOT run (epochs=$epochs)"
fi

mkdir -p slurm_logs
predictor_corrector="${PREDICTOR_CORRECTOR:-FALSE}"
echo "Predictor-corrector for adjoint jobs: $predictor_corrector"

echo "Submitting direct..."
job_direct=$(sbatch --parsable --job-name=mp-peaks-direct \
  --export=ALL,MODE=direct,MP_DTYPE='float32',EPOCHS="$epochs",SAVE_ROOT="$save_root/direct",GRADED_TIME=False,PREDICTOR_CORRECTOR=False \
  "$sbatch_script")
echo "  job_id=$job_direct"

echo "Submitting adjoint..."
job_adjoint=$(sbatch --parsable --job-name=mp-peaks-adjoint \
  --export=ALL,MODE=adjoint,MP_DTYPE='float32',EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adjoint"

echo "Submitting adjoint-mixed..."
job_adjmix=$(sbatch --parsable --job-name=mp-peaks-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE='float16',EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adjmix"

echo "Submitting adjoint-mixed-bfloat..."
job_adjmix_bf=$(sbatch --parsable --job-name=mp-peaks-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE='bfloat16',EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adjmix_bf"

echo "Submitting adjoint-graded..."
job_graded_adj=$(sbatch --parsable --job-name=mp-peaks-graded-adj \
  --export=ALL,MODE=adjoint,MP_DTYPE='float32',EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adj"

echo "Submitting adjoint-mixed-graded..."
job_graded_adjmix=$(sbatch --parsable --job-name=mp-peaks-graded-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE='float16',EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint-mixed",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adjmix"

echo "Submitting adjoint-mixed-bfloat-graded..."
job_graded_adjmix_bf=$(sbatch --parsable --job-name=mp-peaks-graded-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE='bfloat16',EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint-mixed-bfloat",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adjmix_bf"

echo "Submitted 7 Peaks jobs in parallel."

dependency="${job_direct}:${job_adjoint}:${job_adjmix}:${job_adjmix_bf}:${job_graded_adj}:${job_graded_adjmix}:${job_graded_adjmix_bf}"

analysis_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=peaks-analysis \
  --partition=work1 \
  --time=00:15:00 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --output=slurm_logs/peaks_analysis_%j.out \
  --wrap="bash -lc 'module load anaconda3/2023.09-0 && eval \"\$(conda shell.bash hook)\" && conda run -n torch28 python -u \"/home/tcatoe/home_FDNN/MP-torchfde/examples/Peaks/Evaluate Logs/plot_peaks_training_logs.py\" --logs-dir \"/home/tcatoe/home_FDNN/MP-torchfde/examples/Peaks/${save_root}\"'")
echo "Analysis job ID: $analysis_job"
