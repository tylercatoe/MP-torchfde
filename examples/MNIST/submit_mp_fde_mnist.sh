#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_mnist.sbatch"
analysis_script="$script_dir/Evaluate Full Training Logs/plot_mnist_training_logs.py"
for required in "$sbatch_script" "$analysis_script"; do
  [ -f "$required" ] || { echo "ERROR: required file not found: $required"; exit 1; }
done

run_size="${1:-full}"
if [ "$run_size" = "full" ]; then
  epochs="${FULL_EPOCHS:-60}"
  save_root="${SAVE_ROOT:-exp_mp_mnist}"
  analysis_output="$script_dir/Evaluate Full Training Logs"
elif [ "$run_size" = "pilot" ]; then
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_mnist_pilot}"
  analysis_output="$script_dir/$save_root/analysis"
else
  echo "ERROR: run size must be 'pilot' or 'full'."
  exit 1
fi

mkdir -p "$script_dir/slurm_logs"
if [[ "$save_root" = /* ]]; then analysis_logs="$save_root"; else analysis_logs="$script_dir/$save_root"; fi

job_ids=()
submit_run() {
  local label="$1" mode="$2" dtype="$3" mesh="$4" corrector="$5" output="$6"
  local graded=False
  [ "$mesh" = "graded" ] && graded=True
  local job_id
  job_id=$(sbatch --parsable --job-name="mnist-${label}" \
    --export=ALL,MODE="$mode",DTYPE_HI=float32,MP_DTYPE="$dtype",EPOCHS="$epochs",SAVE_ROOT="$output",GRADED_TIME="$graded",PREDICTOR_CORRECTOR="$corrector" \
    "$sbatch_script")
  job_ids+=("$job_id")
  echo "  $label: $job_id"
}

echo "Submitting the canonical 13-run MNIST matrix (epochs=$epochs)..."
submit_run direct direct float32 uniform False "$save_root/direct"
for solver in predictor predictor-corrector; do
  corrector=False
  [ "$solver" = "predictor-corrector" ] && corrector=True
  for mesh in uniform graded; do
    submit_run "${solver}-${mesh}-fp32" adjoint float32 "$mesh" "$corrector" "$save_root/$solver/$mesh-adjoint"
    submit_run "${solver}-${mesh}-fp16" adjoint-mixed float16 "$mesh" "$corrector" "$save_root/$solver/$mesh-adjoint-mixed"
    submit_run "${solver}-${mesh}-bf16" adjoint-mixed-bfloat bfloat16 "$mesh" "$corrector" "$save_root/$solver/$mesh-adjoint-mixed-bfloat"
  done
done

dependency=$(IFS=:; echo "${job_ids[*]}")
analysis_env="${ENV_NAME:-torch28}"
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" --job-name=mnist-analysis \
  --partition=work1 --time=00:15:00 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --output="$script_dir/slurm_logs/mnist_analysis_%j.out" \
  --wrap="bash -lc 'module load anaconda3/2023.09-0 && conda run -n \"$analysis_env\" python -u \"$analysis_script\" --logs-dir \"$analysis_logs\" --output-dir \"$analysis_output\"'")

echo "Submitted ${#job_ids[@]} training jobs."
echo "Analysis job (after all training succeeds): $analysis_job"
