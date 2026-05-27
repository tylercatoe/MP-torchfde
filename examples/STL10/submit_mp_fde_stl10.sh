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
  save_root="${SAVE_ROOT:-exp_mp_stl10_pilot}"
  echo "Submitting PILOT jobs with epochs=$epochs"
else
  epochs="${FULL_EPOCHS:-160}"
  save_root="${SAVE_ROOT:-exp_mp_stl10}"
  echo "Submitting FULL jobs with epochs=$epochs"
fi

mkdir -p slurm_logs

# Optional: allow first job to download STL10, keep others off to avoid races.
download_data="${DOWNLOAD_DATA:-0}"

echo "Submitting direct..."
job_direct=$(sbatch --parsable --job-name=mp-stl10-direct \
  --export=ALL,MODE=direct,EPOCHS="$epochs",SAVE_ROOT="$save_root/direct",DOWNLOAD_DATA="$download_data" \
  "$sbatch_script")
echo "  job_id=$job_direct"

echo "Submitting adjoint..."
job_adj=$(sbatch --parsable --job-name=mp-stl10-adjoint \
  --export=ALL,MODE=adjoint,MP_DTYPE=float32,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint",DOWNLOAD_DATA=0 \
  "$sbatch_script")
echo "  job_id=$job_adj"

echo "Submitting adjoint-mixed..."
job_adjmix=$(sbatch --parsable --job-name=mp-stl10-adjmix \
  --export=ALL,MODE=adjoint-mixed,MP_DTYPE=float16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed",DOWNLOAD_DATA=0 \
  "$sbatch_script")
echo "  job_id=$job_adjmix"

echo "Submitting adjoint-mixed-bfloat..."
job_adjmix_bf16=$(sbatch --parsable --job-name=mp-stl10-adjmix-bf16 \
  --export=ALL,MODE=adjoint-mixed-bfloat,MP_DTYPE=bfloat16,EPOCHS="$epochs",SAVE_ROOT="$save_root/adjoint-mixed-bfloat",DOWNLOAD_DATA=0 \
  "$sbatch_script")
echo "  job_id=$job_adjmix_bf16"

echo "Submitted 4 jobs in parallel."

merge_summary="${MERGE_SUMMARY:-1}"
if [ "$merge_summary" = "1" ]; then
  submit_dir="$(pwd)"
  dep="afterany:${job_direct}:${job_adj}:${job_adjmix}:${job_adjmix_bf16}"
  merge_job_script="$submit_dir/slurm_logs/mp_fde_stl10_summary_merge_job.sh"
  cat > "$merge_job_script" <<'SLURM'
#!/bin/bash
set -euo pipefail
cd "$SUBMIT_DIR"

if command -v module >/dev/null 2>&1; then
  module load anaconda3/2023.09-0 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  env_name="${ENV_NAME:-torch28}"
  PY_RUNNER=(conda run -n "$env_name" python)
elif command -v python3 >/dev/null 2>&1; then
  PY_RUNNER=(python3)
elif command -v python >/dev/null 2>&1; then
  PY_RUNNER=(python)
else
  echo "ERROR: no python interpreter found for summary merge job."
  exit 1
fi

"${PY_RUNNER[@]}" - <<'PY'
import os
import time

save_root = os.environ["SAVE_ROOT"]
modes = ["direct", "adjoint", "adjoint-mixed", "adjoint-mixed-bfloat"]
headers = [
    "Mode",
    "Val Error",
    "Train GPU Mem (MB)",
    "Train Time (s)",
    "Infer GPU Mem (MB)",
    "Infer Time (s)",
]

def parse_latest_row(path: str):
    if not os.path.isfile(path):
        print(f"[merge] missing summary input: {path}")
        return None
    latest = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if "|" not in line:
                continue
            if line.startswith("Mode"):
                continue
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 6:
                continue
            latest = cells[:6]
    return latest

rows = []
for mode in modes:
    summary_path = os.path.join(save_root, mode, "summary.log")
    parsed = parse_latest_row(summary_path)
    if parsed is None:
        rows.append([mode, "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"])
    else:
        parsed[0] = mode
        rows.append(parsed)

widths = [len(h) for h in headers]
for row in rows:
    for i, cell in enumerate(row):
        widths[i] = max(widths[i], len(str(cell)))

def fmt(cells):
    return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

line = "-+-".join("-" * w for w in widths)
table = "\n".join([fmt(headers), line] + [fmt(r) for r in rows])

out_path = os.path.join(save_root, "summary_all.log")
print(f"[merge] cwd={os.getcwd()}")
print(f"[merge] SAVE_ROOT={save_root}")
print(f"[merge] writing to={out_path}")
with open(out_path, "a", encoding="utf-8") as f:
    f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    f.write(table)
    f.write("\n")

print("Combined summary table written to:", out_path)
print(table)
PY
SLURM
  chmod +x "$merge_job_script"
  echo "Submitting summary-merge job (dependency: $dep)..."
  job_merge=$(sbatch --parsable --job-name=mp-stl10-summary-merge \
    --dependency="$dep" \
    --output=slurm_logs/mp_fde_stl10_summary_%j.out \
    --error=slurm_logs/mp_fde_stl10_summary_%j.err \
    --export=ALL,SAVE_ROOT="$save_root",SUBMIT_DIR="$submit_dir" \
    "$merge_job_script")
  echo "  merge_job_id=$job_merge"
else
  echo "Skipping summary merge job because MERGE_SUMMARY=$merge_summary"
fi
