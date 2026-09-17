#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="$script_dir/train_mp_fde_fashion_mnist.sbatch"

if [ ! -f "$sbatch_script" ]; then
  echo "ERROR: sbatch script not found: $sbatch_script"
  exit 1
fi

epochs="${EPOCHS:-3}"
t_values=(1 2 4 8 16 32 64 128)
modes=(direct adjoint adjoint-mixed adjoint-mixed-bfloat)
env_name="${ENV_NAME:-torch28}"
adjoint_method="${ADJOINT_METHOD:-predictor-f}"
direct_method="${DIRECT_METHOD:-predictor}"
mp_loss_scaler="${MP_LOSS_SCALER:-auto}"
job_ids=()

case "${GRADED_TIME:-FALSE}" in
  true|TRUE|True|1|yes|YES|Yes)
    graded_time="True"
    ;;
  false|FALSE|False|0|no|NO|No|"")
    graded_time="False"
    ;;
  *)
    echo "ERROR: Invalid GRADED_TIME=${GRADED_TIME}"
    exit 1
    ;;
esac

case "${PREDICTOR_CORRECTOR:-FALSE}" in
  true|TRUE|True|1|yes|YES|Yes)
    predictor_corrector="True"
    ;;
  false|FALSE|False|0|no|NO|No|"")
    predictor_corrector="False"
    ;;
  *)
    echo "ERROR: Invalid PREDICTOR_CORRECTOR=${PREDICTOR_CORRECTOR}"
    exit 1
    ;;
esac

case "$mp_loss_scaler" in
  auto|dynamic|false)
    ;;
  *)
    echo "ERROR: MP_LOSS_SCALER must be auto, dynamic, or false"
    exit 1
    ;;
esac

if { [ "$graded_time" = "True" ] || [ "$predictor_corrector" = "True" ]; } \
  && [ "$adjoint_method" != "predictor-f" ]; then
  echo "ERROR: graded time and predictor-corrector require ADJOINT_METHOD=predictor-f"
  exit 1
fi

mesh_label="uniform"
if [ "$graded_time" = "True" ]; then
  mesh_label="graded"
fi
solver_label="$adjoint_method"
if [ "$adjoint_method" = "predictor-f" ]; then
  solver_label="predictor"
  if [ "$predictor_corrector" = "True" ]; then
    solver_label="predictor_corrector"
  fi
fi
solver_slug="${solver_label//-/_}"
variant_label="${mesh_label}_${solver_slug}"
output_root="${SAVE_ROOT:-exp_mp_fashion_mnist_t_sweep_${variant_label}}"

mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
manifest_path="${output_root}/sweep_manifest.csv"
mkdir -p slurm_logs
echo "T,mode,job_id,run_log,save_root,adjoint_method,direct_method,mesh,solver,predictor_corrector,mp_dtype,loss_scaler" > "$manifest_path"

echo "Submitting FashionMNIST T sweep"
echo "  epochs=${epochs}"
echo "  adjoint_method=${adjoint_method}"
echo "  mesh=${mesh_label}"
echo "  solver=${solver_label}"
echo "  mp_loss_scaler=${mp_loss_scaler}"
echo "  output_root=${output_root}"
echo "  manifest=${manifest_path}"

for t_val in "${t_values[@]}"; do
  for mode in "${modes[@]}"; do
    dtype_hi="float32"
    mp_dtype="float32"
    run_graded_time="$graded_time"
    run_predictor_corrector="$predictor_corrector"
    run_mesh="$mesh_label"
    run_solver="$solver_label"
    effective_loss_scaler="false"

    case "$mode" in
      direct)
        mp_dtype="float32"
        run_graded_time="False"
        run_predictor_corrector="False"
        run_mesh="uniform"
        run_solver="direct"
        ;;
      adjoint)
        mp_dtype="float32"
        ;;
      adjoint-mixed)
        mp_dtype="float16"
        if [ "$mp_loss_scaler" = "auto" ] || [ "$mp_loss_scaler" = "dynamic" ]; then
          effective_loss_scaler="dynamic"
        fi
        ;;
      adjoint-mixed-bfloat)
        mp_dtype="bfloat16"
        ;;
      *)
        echo "ERROR: unexpected mode '$mode'"
        exit 1
        ;;
    esac

    run_save="${output_root}/T_${t_val}/${mode}"
    run_log="${run_save}/logs"
    job_name="fmnist-${mode}-T${t_val}-${variant_label}"

    export_vars="ALL,MODE=${mode},EPOCHS=${epochs},SAVE_ROOT=${run_save},T_FINAL=${t_val},DTYPE_HI=${dtype_hi},MP_DTYPE=${mp_dtype},MP_LOSS_SCALER=${mp_loss_scaler},ADJOINT_METHOD=${adjoint_method},DIRECT_METHOD=${direct_method},GRADED_TIME=${run_graded_time},PREDICTOR_CORRECTOR=${run_predictor_corrector},BENCHMARK_ONLY=1"
    if [ -n "${BATCH_SIZE:-}" ]; then
      export_vars="${export_vars},BATCH_SIZE=${BATCH_SIZE}"
    fi
    if [ -n "${TEST_BATCH_SIZE:-}" ]; then
      export_vars="${export_vars},TEST_BATCH_SIZE=${TEST_BATCH_SIZE}"
    fi
    if [ -n "${STEP_SIZE:-}" ]; then
      export_vars="${export_vars},STEP_SIZE=${STEP_SIZE}"
    fi
    if [ -n "${BETA:-}" ]; then
      export_vars="${export_vars},BETA=${BETA}"
    fi
    if [ -n "${MEMORY:-}" ]; then
      export_vars="${export_vars},MEMORY=${MEMORY}"
    fi

    job_id=$(sbatch --parsable --job-name="$job_name" --export="$export_vars" "$sbatch_script")
    job_ids+=("$job_id")
    echo "${t_val},${mode},${job_id},${run_log},${run_save},${adjoint_method},${direct_method},${run_mesh},${run_solver},${run_predictor_corrector},${mp_dtype},${effective_loss_scaler}" >> "$manifest_path"
    echo "submitted: mode=${mode} T=${t_val} mesh=${run_mesh} solver=${run_solver} job_id=${job_id}"
  done
done

if [ "${#job_ids[@]}" -eq 0 ]; then
  echo "ERROR: no benchmark jobs were submitted, cannot submit summary job."
  exit 1
fi

dependency_ids="$(IFS=:; echo "${job_ids[*]}")"
summary_job_name="fmnist-t-sweep-summary"
summary_out="slurm_logs/fmnist_t_sweep_summary_%j.out"
summary_err="slurm_logs/fmnist_t_sweep_summary_%j.err"

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
conda run -n "$env_name" python "$script_dir/summarize_t_sweep_fashion_mnist.py" --manifest "$manifest_path" --epoch "$epochs"
EOF
)

echo "Done. Manifest written to ${manifest_path}"
echo "Submitted dependent summary job: job_id=${summary_job_id}"
echo "Summary logs:"
echo "  stdout: ${summary_out}"
echo "  stderr: ${summary_err}"
