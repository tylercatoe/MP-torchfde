#!/usr/bin/env python
"""
Compare results from torchfde FP32 vs rampde FP16 STL10 experiment.

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

def plot_acc(data1: dict, data2: dict):
    import matplotlib.pyplot as plt

    epochs1 = data1["epochs"]
    beta = data1["beta"]
    T = data1["T"]
    step = data1["step_size"]
    acc1 = [e["test_acc"] for e in epochs1]
    epochs2 = data2["epochs"]
    acc2 = [e["test_acc"] for e in epochs2]
    plt.plot(acc1, marker="o", label="torchfde FP32")
    plt.plot(acc2, marker="s", label="rampde FP16")
    plt.title(f"STL10 Test Accuracy over Epochs (β={data1['beta']}, T={data1['T']}, step={data1['step_size']})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.xlim(0, max(len(acc1), len(acc2)) - 1)
    plt.ylim(0, 1.0)
    plt.grid()
    plt.legend()
    plt.savefig(f"test_accuracy_plot_b{beta}_T{T}_h{step}.png")
    plt.show()


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <fp32.json> <fp16.json>")
        sys.exit(1)

    fp32_raw = load(sys.argv[1])
    fp16_raw = load(sys.argv[2])
    fp32 = summarize(fp32_raw)
    fp16 = summarize(fp16_raw)

    print("\n" + "=" * 70)
    print("  Neural FDE STL10: torchfde FP32 vs rampde FP16")
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

    # Plot accuracy curves
    plot_acc(fp32_raw, fp16_raw)


if __name__ == "__main__":
    main()
