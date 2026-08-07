#!/usr/bin/env python
"""
Compare results from two or more fractional Lotka-Volterra solver runs
(torchfde FP32 / rampde L1 FP16 / rampde predictor FP16).

Usage:
    python compare_results.py results/torchfde_fp32/results.json results/rampde_fp16/results.json
    python compare_results.py results/torchfde_fp32/results.json results/rampde_fp16/results.json results/rampde_predictor_fp16/results.json
"""

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize(data: dict) -> dict:
    iters = data["iterations"]
    final = iters[-1]
    best_err = min(r["param_err"] for r in iters)
    avg_mem = sum(r["peak_mem_mb"] for r in iters) / len(iters)
    peak_mem = max(r["peak_mem_mb"] for r in iters)
    return {
        "solver": data["solver"],
        "beta": data["beta"], "T": data["T"],
        "true_params": data["true_params"],
        "n_records": len(iters),
        "final_iter": final["iter"],
        "final_loss": final["loss"],
        "final_param_err": final["param_err"],
        "best_param_err": best_err,
        "final_params": final["params"],
        "avg_peak_mem_mb": avg_mem, "max_peak_mem_mb": peak_mem,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <run1.json> <run2.json> [<run3.json> ...]")
        sys.exit(1)

    runs = [summarize(load(p)) for p in sys.argv[1:]]
    reference = runs[0]

    print("\n" + "=" * 88)
    print("  Fractional Lotka-Volterra parameter estimation — solver comparison")
    print("=" * 88)
    print(f"  β={reference['beta']}  T={reference['T']}  "
          f"true_params={reference['true_params']}  iters={reference['final_iter']}")
    print()

    label_w = 24
    col_w = 16
    header = f"  {'Metric':<{label_w}}" + "".join(
        f"{r['solver']:>{col_w}}" for r in runs
    )
    print(header)
    print("  " + "-" * (label_w + col_w * len(runs)))

    def row(label, key, fmt=".6f"):
        cells = "".join(f"{r[key]:>{col_w}{fmt}}" for r in runs)
        print(f"  {label:<{label_w}}{cells}")

    row("Final loss", "final_loss")
    row("Final param err", "final_param_err")
    row("Best param err", "best_param_err")
    row("Avg peak mem (MB)", "avg_peak_mem_mb", fmt=".1f")
    row("Max peak mem (MB)", "max_peak_mem_mb", fmt=".1f")

    print()
    for r in runs:
        print(f"  {r['solver']:<{label_w}}final params = "
              f"{['%.4f' % p for p in r['final_params']]}")

    # Relative comparisons against the first run (typically torchfde_fp32)
    if len(runs) > 1:
        print(f"\n  Relative to {reference['solver']!r}:")
        for r in runs[1:]:
            mem_saving = 100.0 * (reference["max_peak_mem_mb"] - r["max_peak_mem_mb"]) / reference["max_peak_mem_mb"]
            err_delta = 100.0 * (r["final_param_err"] - reference["final_param_err"]) / (reference["final_param_err"] + 1e-12)
            print(f"    {r['solver']:<24} memory {'saving' if mem_saving >= 0 else 'increase'}: "
                  f"{abs(mem_saving):5.1f}%   param err change: {err_delta:+6.1f}%")

    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
