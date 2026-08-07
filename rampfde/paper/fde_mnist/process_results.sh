#!/bin/bash
#
# FDE MNIST Results Processing Script
#
# Locates the torchfde_fp32 / rampde_fp16 / rampde_predictor_fp16 result
# directories for a given (beta, T) configuration and writes a comparison
# summary to outputs/.
#
# Usage:
#   ./process_results.sh [--beta 0.7] [--T 2.0]

set -e
cd "$(dirname "$0")"

BETA="0.7"
T="2.0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --beta) BETA="$2"; shift 2 ;;
        --T)    T="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p outputs

RUN1="results/torchfde_fp32_b${BETA}_T${T}/results.json"
RUN2="results/rampde_fp16_b${BETA}_T${T}/results.json"
RUN3="results/rampde_predictor_fp16_b${BETA}_T${T}/results.json"

RUNS=()
for r in "$RUN1" "$RUN2" "$RUN3"; do
    if [ -f "$r" ]; then
        RUNS+=("$r")
    else
        echo "Warning: missing $r (skipping)"
    fi
done

if [ ${#RUNS[@]} -lt 2 ]; then
    echo "Error: need at least 2 result files to compare; found ${#RUNS[@]}."
    echo "Run run_experiment.sh (or run_test.sh) first."
    exit 1
fi

OUT="outputs/comparison_b${BETA}_T${T}.txt"
echo "Comparing: ${RUNS[*]}"
python compare_results.py "${RUNS[@]}" | tee "$OUT"

echo ""
echo "Summary written to $OUT"
