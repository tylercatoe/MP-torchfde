#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_stl10.sbatch"

if [ ! -f "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found: $sbatch_script"
  exit 1
fi

run_size="${1:-full}"   # full | pilot

if [ "$run_size" = "pilot" ]; then
  epochs="${PILOT_EPOCHS:-5}"
  save_root="${SAVE_ROOT:-exp_mp_stl10_beta_sweep_pilot}"
  echo "Submitting PILOT jobs with epochs=$epochs"
else
  epochs="${FULL_EPOCHS:-160}"
  save_root="${SAVE_ROOT:-exp_mp_beta_sweep_stl10}"
  echo "Submitting FULL jobs with epochs=$epochs"
fi

output_root="$save_root"
# output_root="${SAVE_ROOT:-exp_mp_stl10_beta_sweep}"
# beta_values=(0.1 0.2 0.4 0.6 0.8 1.0)
beta_values=(1.0)
modes=(direct adjoint adjoint-mixed adjoint-mixed-bfloat)
env_name="${ENV_NAME:-torch28}"
job_ids=()

mkdir -p slurm_logs
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
manifest_path="${output_root}/sweep_manifest.csv"
mkdir -p slurm_logs
echo "Beta,mode,job_id,run_log,save_root" > "$manifest_path"

echo "Submitting STL10 Beta sweep"
echo "  epochs=${epochs}"
echo "  output_root=${output_root}"
echo "  manifest=${manifest_path}"

# Optional: allow first job to download STL10, keep others off to avoid races.
download_data="${DOWNLOAD_DATA:-0}"

for beta_val in "${beta_values[@]}"; do
  for mode in "${modes[@]}"; do
    dtype_hi="float32"
    mp_dtype="float32"

    case "$mode" in
      direct)
        mp_dtype="float32"
        ;;
      adjoint)
        mp_dtype="float32"
        ;;
      adjoint-mixed)
        mp_dtype="float16"
        ;;
      adjoint-mixed-bfloat)
        mp_dtype="bfloat16"
        ;;
      *)
        echo "ERROR: unexpected mode '$mode'"
        exit 1
        ;;
    esac

    run_save="${output_root}/Beta_${beta_val}/${mode}"
    run_log="${run_save}/logs"
    job_name="stl10-${mode}-Beta_${beta_val}"

    export_vars="ALL,MODE=${mode},EPOCHS=${epochs},SAVE_ROOT=${run_save},BETA=${beta_val},DTYPE_HI=${dtype_hi},MP_DTYPE=${mp_dtype},BENCHMARK_ONLY=1"
    if [ -n "${BATCH_SIZE:-}" ]; then
      export_vars="${export_vars},BATCH_SIZE=${BATCH_SIZE}"
    fi
    if [ -n "${TEST_BATCH_SIZE:-}" ]; then
      export_vars="${export_vars},TEST_BATCH_SIZE=${TEST_BATCH_SIZE}"
    fi
    if [ -n "${STEP_SIZE:-}" ]; then
      export_vars="${export_vars},STEP_SIZE=${STEP_SIZE}"
    fi
    if [ -n "${T:-}" ]; then
      export_vars="${export_vars},T_value=${T}"
    fi
    if [ -n "${MEMORY:-}" ]; then
      export_vars="${export_vars},MEMORY=${MEMORY}"
    fi

    job_id=$(sbatch --parsable --job-name="$job_name" --export="$export_vars" "$sbatch_script")
    job_ids+=("$job_id")
    echo "${beta_val},${mode},${job_id},${run_log},${run_save}" >> "$manifest_path"
    echo "submitted: mode=${mode} Beta=${beta_val} job_id=${job_id}"
  done
done


if [ "${#job_ids[@]}" -eq 0 ]; then
  echo "ERROR: no benchmark jobs were submitted, cannot submit summary job."
  exit 1
fi

dependency_ids="$(IFS=:; echo "${job_ids[*]}")"
summary_job_name="stl10-beta-sweep-summary"
summary_out="slurm_logs/stl10_beta_sweep_summary_%j.out"
summary_err="slurm_logs/stl10_beta_sweep_summary_%j.err"

summary_job_id=$(sbatch --parsable \
  --job-name="$summary_job_name" \
  --output="$summary_out" \
  --error="$summary_err" \
  --dependency="afterany:${dependency_ids}" <<EOF
#!/bin/bash
#SBATCH --partition=work1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
module load anaconda3/2023.09-0
eval "\$(conda shell.bash hook)"
conda activate "$env_name"
conda run -n "$env_name" python "$script_dir/summarize_beta_sweep_stl10.py" --manifest "$manifest_path" --epoch "$epochs"
EOF
)

echo "Done. Manifest written to ${manifest_path}"
echo "Submitted dependent summary job: job_id=${summary_job_id}"
echo "Summary logs:"
echo "  stdout: ${summary_out}"
echo "  stderr: ${summary_err}"