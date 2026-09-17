#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_stl10.sbatch"
analysis_script="$script_dir/Evaluate Logs/plot_stl10_training_logs.py"
for required in "$sbatch_script" "$analysis_script"; do
  [ -f "$required" ] || { echo "ERROR: required file not found: $required"; exit 1; }
done

run_size="${1:-full}"
if [ "$run_size" = "full" ]; then
  epochs="${FULL_EPOCHS:-160}"
  save_root="${SAVE_ROOT:-exp_mp_stl10}"
  analysis_output="$script_dir/Evaluate Logs"
elif [ "$run_size" = "pilot" ]; then
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_stl10_pilot}"
  analysis_output="$script_dir/$save_root/analysis"
else
  echo "ERROR: run size must be 'pilot' or 'full'."
  exit 1
fi

mkdir -p "$script_dir/slurm_logs"
if [[ "$save_root" = /* ]]; then analysis_logs="$save_root"; else analysis_logs="$script_dir/$save_root"; fi
data_root="${DATA_ROOT:-.data/stl10}"
if [[ "$data_root" = /* ]]; then data_root_abs="$data_root"; else data_root_abs="$script_dir/$data_root"; fi
analysis_env="${ENV_NAME:-torch28}"

submit_options=(--parsable)
if [ "${DOWNLOAD_DATA:-0}" = "1" ]; then
  echo "Submitting a data-preparation job before the training matrix..."
  prep_job=$(sbatch --parsable --job-name=stl10-data \
    --partition=work1 --time=01:00:00 --ntasks=1 --cpus-per-task=2 --mem=8G \
    --output="$script_dir/slurm_logs/stl10_data_%j.out" \
    --wrap="bash -lc 'module load anaconda3/2023.09-0 && conda run -n \"$analysis_env\" python -c \"from torchvision.datasets import STL10; STL10(root=\\\"$data_root_abs\\\", split=\\\"train\\\", download=True)\"'")
  submit_options+=(--dependency="afterok:$prep_job")
  echo "  data preparation: $prep_job"
fi

job_ids=()
submit_run() {
  local label="$1" mode="$2" dtype="$3" mesh="$4" corrector="$5" output="$6"
  local graded=False
  [ "$mesh" = "graded" ] && graded=True
  local job_id
  job_id=$(sbatch "${submit_options[@]}" --job-name="stl10-${label}" \
    --export=ALL,MODE="$mode",MP_DTYPE="$dtype",EPOCHS="$epochs",SAVE_ROOT="$output",DATA_ROOT="$data_root_abs",DOWNLOAD_DATA=0,GRADED_TIME="$graded",PREDICTOR_CORRECTOR="$corrector" \
    "$sbatch_script")
  job_ids+=("$job_id")
  echo "  $label: $job_id"
}

echo "Submitting the canonical 13-run STL10 matrix (epochs=$epochs)..."
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
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" --job-name=stl10-analysis \
  --partition=work1 --time=00:15:00 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --output="$script_dir/slurm_logs/stl10_analysis_%j.out" \
  --wrap="bash -lc 'module load anaconda3/2023.09-0 && conda run -n \"$analysis_env\" python -u \"$analysis_script\" --logs-dir \"$analysis_logs\" --output-dir \"$analysis_output\"'")

echo "Submitted ${#job_ids[@]} training jobs."
echo "Analysis job (after all training succeeds): $analysis_job"
