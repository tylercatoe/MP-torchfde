#!/usr/bin/env python
"""
Compare results from torchfde FP32 vs rampde FP16 runs.

Usage:
    python compare_results.py results/torchfde_fp32/results.json results/rampde_fp16/results.json
"""

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize(data: dict) -> dict:
    epochs = data["epochs"]
    best_acc = max(e["test_acc"] for e in epochs)
    last_acc = epochs[-1]["test_acc"]
    avg_mem = sum(e["peak_mem_mb"] for e in epochs) / len(epochs)
    peak_mem = max(e["peak_mem_mb"] for e in epochs)
    return {
        "solver": data["solver"],
        "beta": data["beta"], "T": data["T"], "step_size": data["step_size"],
        "n_params": data["n_params"],
        "n_epochs": len(epochs),
        "best_acc": best_acc, "last_acc": last_acc,
        "avg_peak_mem_mb": avg_mem, "max_peak_mem_mb": peak_mem,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <fp32.json> <fp16.json>")
        sys.exit(1)

    fp32 = summarize(load(sys.argv[1]))
    fp16 = summarize(load(sys.argv[2]))

    print("\n" + "=" * 70)
    print("  Neural FDE MNIST: torchfde FP32 vs rampde FP16")
    print("=" * 70)
    print(f"  β={fp32['beta']}  T={fp32['T']}  h={fp32['step_size']}  "
          f"N={int(fp32['T'] / fp32['step_size']) + 1}  "
          f"params={fp32['n_params']:,}  epochs={fp32['n_epochs']}")
    print()

    w = 22
    print(f"  {'Metric':<{w}} {'torchfde FP32':>14}  {'rampde FP16':>12}  {'Δ':>10}")
    print("  " + "-" * 62)

    def row(label, k32, k16, fmt=".4f", pct=False):
        v32, v16 = fp32[k32], fp16[k16]
        delta = v16 - v32 if not pct else 100.0 * (v32 - v16) / v32
        sign = "+" if delta >= 0 else ""
        dsym = "%" if pct else ""
        print(f"  {label:<{w}} {v32:>14{fmt}}  {v16:>12{fmt}}  "
              f"{sign}{delta:>9.2f}{dsym}")

    row("Best test acc",    "best_acc",        "best_acc")
    row("Final test acc",   "last_acc",         "last_acc")
    row("Avg peak mem (MB)", "avg_peak_mem_mb", "avg_peak_mem_mb", fmt=".1f")
    row("Max peak mem (MB)", "max_peak_mem_mb", "max_peak_mem_mb", fmt=".1f")

    # Memory saving
    mem_saving = 100.0 * (fp32["max_peak_mem_mb"] - fp16["max_peak_mem_mb"]) / fp32["max_peak_mem_mb"]
    print(f"\n  Memory saving: {mem_saving:.1f}%  "
          f"({'rampde uses less' if mem_saving > 0 else 'rampde uses more'})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
