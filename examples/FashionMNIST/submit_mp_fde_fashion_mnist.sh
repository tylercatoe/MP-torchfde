#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_fashion_mnist.sbatch"

if [ ! -f "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found: $sbatch_script"
  exit 1
fi

run_size="${1:-full}"   # full | pilot

if [ "$run_size" = "pilot" ]; then
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_fashion_mnist_pilot}"
  echo "Submitting PILOT jobs with epochs=$epochs"
else
  epochs="${FULL_EPOCHS:-160}"
  save_root="${SAVE_ROOT:-exp_mp_fashion_mnist}"
  echo "Submitting FULL jobs with epochs=$epochs"
fi

mkdir -p slurm_logs "$script_dir/slurm_logs"
predictor_corrector="${PREDICTOR_CORRECTOR:-FALSE}"
echo "Predictor-corrector for adjoint jobs: $predictor_corrector"

echo "Submitting direct..."
job_direct=$(sbatch --parsable --job-name=mp-fashion-mnist-direct \
  --export=ALL,MODE=direct,EPOCHS="$epochs",SAVE_ROOT="$save_root/direct",GRADED_TIME=False,PREDICTOR_CORRECTOR=False \
  "$sbatch_script")
echo "  job_id=$job_direct"

echo "Submitting adjoint..."
job_adj=$(sbatch --parsable --job-name=mp-fashion-mnist-adjoint \
  --export=ALL,MODE=adjoint,DTYPE_HI=float32,MP_DTYPE=float32,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adj"

echo "Submitting adjoint-mixed..."
job_adjmix=$(sbatch --parsable --job-name=mp-fashion-mnist-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE=float16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adjmix"

echo "Submitting adjoint-mixed-bfloat..."
job_adjmix_bf16=$(sbatch --parsable --job-name=mp-fashion-mnist-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE=bfloat16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat",GRADED_TIME=False,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_adjmix_bf16"

echo "Submitting adjoint-graded..."
job_graded_adj=$(sbatch --parsable --job-name=mp-fashion-mnist-graded-adj \
  --export=ALL,MODE=adjoint,DTYPE_HI=float32,MP_DTYPE=float32,EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adj"

echo "Submitting adjoint-mixed-graded..."
job_graded_adjmix=$(sbatch --parsable --job-name=mp-fashion-mnist-graded-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE=float16,EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint-mixed",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adjmix"

echo "Submitting adjoint-mixed-bfloat-graded..."
job_graded_adjmix_bf16=$(sbatch --parsable --job-name=mp-fashion-mnist-graded-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE=bfloat16,EPOCHS="$epochs",SAVE_ROOT="$save_root/graded-adjoint-mixed-bfloat",GRADED_TIME=True,PREDICTOR_CORRECTOR="$predictor_corrector" \
  "$sbatch_script")
echo "  job_id=$job_graded_adjmix_bf16"

echo "Submitted 7 jobs in parallel."

dependency="${job_direct}:${job_adj}:${job_adjmix}:${job_adjmix_bf16}:${job_graded_adj}:${job_graded_adjmix}:${job_graded_adjmix_bf16}"
analysis_script="$script_dir/Evaluate Full Training Logs/plot_fashion_mnist_training_logs.py"
analysis_env="${ENV_NAME:-torch28}"
if [[ "$save_root" = /* ]]; then
  analysis_logs_dir="$save_root"
else
  analysis_logs_dir="$script_dir/$save_root"
fi

if [ ! -f "$analysis_script" ]; then
  echo "ERROR: analysis script not found: $analysis_script"
  exit 1
fi

analysis_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=fashion-analysis \
  --partition=work1 \
  --time=00:15:00 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --output="$script_dir/slurm_logs/fashion_analysis_%j.out" \
  --wrap="bash -lc 'module load anaconda3/2023.09-0 && eval \"\$(conda shell.bash hook)\" && conda run -n \"$analysis_env\" python -u \"$analysis_script\" --logs-dir \"$analysis_logs_dir\"'")
echo "Analysis job ID: $analysis_job"
