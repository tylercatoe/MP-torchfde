#!/bin/bash
set -euo pipefail

# Submit a matched integer-order ODE vs. fractional-order FDE experiment.
# Each training configuration is an independent Slurm job. A CPU-only analysis
# job is submitted after all training jobs finish.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}"
ode_dir="$project_root/rampfde/paper/stl10"
ode_script="$ode_dir/ode_stl10.py"
fde_sbatch="$script_dir/train_mp_fde_stl10.sbatch"
analysis_script="$script_dir/analyze_stl10_order_comparison.py"

output_root="${SAVE_ROOT:-$script_dir/exp_stl10_order_comparison}"
manifest_path="$output_root/comparison_manifest.csv"
env_name="${ENV_NAME:-torch28}"
seed="${SEED:-25}"
epochs="${EPOCHS:-160}"
width="${WIDTH:-64}"
batch_size="${BATCH_SIZE:-16}"
test_batch_size="${TEST_BATCH_SIZE:-16}"
learning_rate="${LR:-0.05}"
weight_decay="${WEIGHT_DECAY:-5e-4}"
step_size="${STEP_SIZE:-0.1}"
t_final="${T_FINAL:-1.0}"
time_bins="${TIME_BINS:-4}"
train_size="${TRAIN_SIZE:-4000}"
download_data="${DOWNLOAD_DATA:-0}"
dry_run="${DRY_RUN:-0}"

beta_values=(0.1 0.2 0.4 0.6 0.8 1.0)
job_ids=()

if [ ! -f "$ode_script" ]; then
  echo "ERROR: ODE script not found: $ode_script" >&2
  exit 1
fi
if [ ! -f "$fde_sbatch" ]; then
  echo "ERROR: FDE sbatch script not found: $fde_sbatch" >&2
  exit 1
fi
if [ ! -f "$analysis_script" ]; then
  echo "ERROR: analysis script not found: $analysis_script" >&2
  exit 1
fi

mkdir -p "$output_root" "$output_root/slurm_logs"
output_root="$(cd "$output_root" && pwd)"
manifest_path="$output_root/comparison_manifest.csv"
printf 'equation,beta,method,precision,job_id,result_root,log_path\n' > "$manifest_path"

echo "Submitting matched STL10 ODE/FDE comparison"
echo "  project_root=$project_root"
echo "  output_root=$output_root"
echo "  seed=$seed epochs=$epochs width=$width batch_size=$batch_size"
echo "  manifest=$manifest_path"

submit_ode_job() {
  local precision="$1"
  local result_root="$2"
  local label="$3"
  local job_name="stl10-ode-${label}"
  local job_id

  if [ "$dry_run" = "1" ]; then
    job_id="DRY_RUN"
    echo "dry-run: $job_name"
  else
    job_id=$(sbatch --parsable \
      --job-name="$job_name" \
      --chdir="$script_dir" \
      --partition=work1 \
      --time=48:00:00 \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --mem=64G \
      --gres=gpu:h200:1 \
      --output="$output_root/slurm_logs/ode_${label}_%j.out" \
      --error="$output_root/slurm_logs/ode_${label}_%j.err" <<EOF
#!/bin/bash
set -euo pipefail
module load cuda/12.3.0
module load anaconda3/2023.09-0
eval "\$(conda shell.bash hook)"
conda activate "$env_name"
export PYTHONPATH="$project_root/rampfde:\${PYTHONPATH:-}"
export MPLBACKEND=Agg

mkdir -p "$result_root"
conda run -n "$env_name" python -u "$ode_script" \\
  --odeint rampde \\
  --method rk4 \\
  --precision "$precision" \\
  --no_grad_scaler \\
  --no_dynamic_scaler \\
  --seed "$seed" \\
  --nepochs "$epochs" \\
  --lr "$learning_rate" \\
  --momentum 0.9 \\
  --batch_size "$batch_size" \\
  --test_batch_size "$test_batch_size" \\
  --weight_decay "$weight_decay" \\
  --width "$width" \\
  --test_freq 1 \\
  --results_dir "$result_root"
EOF
    )
    echo "submitted: equation=ode precision=$precision job_id=$job_id"
  fi

  job_ids+=("$job_id")
  printf 'ode,,%s,%s,%s,%s,\n' "$label" "$precision" "$job_id" "$result_root" >> "$manifest_path"
}

submit_fde_job() {
  local beta="$1"
  local mode="$2"
  local precision="$3"
  local run_root="$output_root/fde/Beta_${beta}/${mode}"
  local log_path="$run_root/$mode/training.log"
  local job_name="stl10-fde-${mode}-b${beta}"
  local export_vars
  local job_id

  export_vars="ALL,ENV_NAME=${env_name},MODE=${mode},EPOCHS=${epochs},SAVE_ROOT=${run_root},BETA=${beta},DTYPE_HI=float32,MP_DTYPE=${precision},MP_LOSS_SCALER=false,DOWNLOAD_DATA=${download_data},WIDTH=${width},BATCH_SIZE=${batch_size},TEST_BATCH_SIZE=${test_batch_size},STEP_SIZE=${step_size},T_FINAL=${t_final},TRAIN_SIZE=${train_size},TIME_BINS=${time_bins}"

  if [ "$dry_run" = "1" ]; then
    job_id="DRY_RUN"
    echo "dry-run: $job_name"
  else
    job_id=$(sbatch --parsable \
      --job-name="$job_name" \
      --export="$export_vars" \
      "$fde_sbatch")
    echo "submitted: equation=fde beta=$beta mode=$mode job_id=$job_id"
  fi

  job_ids+=("$job_id")
  printf 'fde,%s,%s,%s,%s,%s,%s\n' "$beta" "$mode" "$precision" "$job_id" "$run_root" "$log_path" >> "$manifest_path"
}

# Integer-order baselines using rampde's fixed-grid ODE solver.
submit_ode_job "float32" "$output_root/ode/float32" "fp32"
submit_ode_job "bfloat16" "$output_root/ode/bfloat16" "bf16"

# Fractional runs: adjoint FP32 and unscaled adjoint BF16.
for beta in "${beta_values[@]}"; do
  submit_fde_job "$beta" "adjoint" "float32"
  submit_fde_job "$beta" "adjoint-mixed-bfloat" "bfloat16"
done

echo "Manifest written to: $manifest_path"
echo "Training configurations: ${#job_ids[@]}"

if [ "$dry_run" = "1" ]; then
  echo "DRY_RUN=1: no Slurm jobs or analysis dependency submitted."
  exit 0
fi

dependency_ids="$(IFS=:; echo "${job_ids[*]}")"
analysis_job_id=$(sbatch --parsable \
  --job-name=stl10-order-analysis \
  --chdir="$script_dir" \
  --partition=work1 \
  --time=00:30:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --dependency="afterany:${dependency_ids}" \
  --output="$output_root/slurm_logs/analysis_%j.out" \
  --error="$output_root/slurm_logs/analysis_%j.err" <<EOF
#!/bin/bash
set -euo pipefail
module load anaconda3/2023.09-0
eval "\$(conda shell.bash hook)"
conda activate "$env_name"
export MPLBACKEND=Agg
python "$analysis_script" \\
  --manifest "$manifest_path" \\
  --expected-epoch "$epochs"
EOF
)

echo "Submitted analysis job: $analysis_job_id"
echo "Analysis depends on training jobs: $dependency_ids"
