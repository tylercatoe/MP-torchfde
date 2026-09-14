#!/usr/bin/env python
"""Create a CSV and Markdown summary from Lotka--Volterra result files."""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


def load_summary(path: str, threshold: float) -> Dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["iterations"]
    final = records[-1]
    best = min(records, key=lambda item: item["val_loss"])
    threshold_records = [
        item for item in records if item["val_loss"] <= threshold
    ]
    return {
        "init_regime": data["init_regime"],
        "mesh": data["mesh"],
        "precision": data["precision"],
        "beta": data["beta"],
        "step_size": data["step_size"],
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "seed": data["seed"],
        "final_iter": final["iter"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_iter": best["iter"],
        "final_param_err": final["param_err"],
        "peak_mem_mb": max(item["peak_mem_mb"] for item in records),
        "mean_iter_time_s": sum(item["iter_time_s"] for item in records) / len(records),
        "total_logged_time_s": sum(item["iter_time_s"] for item in records),
        "iter_to_val_threshold": (
            threshold_records[0]["iter"] if threshold_records else "NA"
        ),
        "final_params": ", ".join(f"{value:.6f}" for value in final["params"]),
        "initialization_params": ", ".join(
            f"{value:.6f}" for value in data["initialization_params"]
        ),
    }


def load_run(path: str) -> Dict:
    """Load one complete run, including its logged training history."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _plotting_modules():
    """Import matplotlib lazily so CSV/Markdown work without plotting deps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; skipping plots.")
        return None
    return plt


def _estimated_time(records: List[Dict]) -> List[float]:
    """Estimate elapsed time from sparse logged iteration timings."""
    if not records:
        return []
    mean_time = sum(item["iter_time_s"] for item in records) / len(records)
    return [item["iter"] * mean_time for item in records]


def write_convergence_plot(paths: List[str], output_dir: str, x_mode: str) -> bool:
    """Write faceted validation-loss curves versus iteration or time."""
    plt = _plotting_modules()
    if plt is None:
        return False

    runs = [load_run(path) for path in paths]
    initializations = sorted({run["init_regime"] for run in runs})
    precisions = [
        precision for precision in ("fp32", "fp16")
        if precision in {run["precision"] for run in runs}
    ]
    colors = {"uniform": "tab:blue", "graded": "tab:orange"}

    fig, axes = plt.subplots(
        len(initializations), len(precisions),
        squeeze=False,
        figsize=(5.2 * len(precisions), 3.8 * len(initializations)),
        sharey=True,
    )
    for row_index, init_regime in enumerate(initializations):
        for column_index, precision in enumerate(precisions):
            ax = axes[row_index][column_index]
            matching = [
                run for run in runs
                if run["init_regime"] == init_regime
                and run["precision"] == precision
            ]
            for run in sorted(matching, key=lambda item: item["mesh"]):
                records = run["iterations"]
                if x_mode == "iteration":
                    x_values = [item["iter"] for item in records]
                    xlabel = "Training iteration"
                else:
                    x_values = _estimated_time(records)
                    xlabel = "Estimated elapsed time (s)"
                y_values = [max(item["val_loss"], 1e-16) for item in records]
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    color=colors.get(run["mesh"], "black"),
                    label=run["mesh"],
                )
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(f"{init_regime} / {precision.upper()}")
            ax.grid(True, which="both", alpha=0.25)
            if column_index == 0:
                ax.set_ylabel("Validation loss")
            ax.legend()

    fig.suptitle("Lotka--Volterra validation convergence", y=1.02)
    fig.tight_layout()
    filename = (
        "validation_loss_vs_iteration.png"
        if x_mode == "iteration"
        else "validation_loss_vs_time.png"
    )
    fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_efficiency_plot(paths: List[str], output_dir: str, threshold: float) -> bool:
    """Write best-validation-loss versus runtime and peak memory."""
    plt = _plotting_modules()
    if plt is None:
        return False

    runs = [load_run(path) for path in paths]
    colors = {"near_true": "tab:blue", "worse": "tab:orange"}
    markers = {"uniform": "o", "graded": "s"}
    fig, axes_grid = plt.subplots(
        1, 2, squeeze=False, figsize=(12, 4.8), sharey=True
    )
    axes = axes_grid[0]

    plotted_labels = set()
    for ax, x_key, xlabel in (
        (axes[0], "runtime", "Estimated total training time (s)"),
        (axes[1], "peak_mem_mb", "Peak GPU memory (MB)"),
    ):
        for run in runs:
            records = run["iterations"]
            best_val_loss = min(item["val_loss"] for item in records)
            mean_iter_time = sum(item["iter_time_s"] for item in records) / len(records)
            x_value = (
                run["iterations"][-1]["iter"] * mean_iter_time
                if x_key == "runtime"
                else max(item["peak_mem_mb"] for item in records)
            )
            label = (
                f"{run['precision'].upper()} / {run['init_regime']} / "
                f"{run['mesh']}"
            )
            ax.scatter(
                x_value,
                max(best_val_loss, 1e-16),
                s=75,
                color=colors.get(run["init_regime"], "black"),
                marker=markers.get(run["mesh"], "o"),
                edgecolors="black",
                linewidths=0.5,
                label=label if label not in plotted_labels else None,
            )
            plotted_labels.add(label)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Best validation loss")

    axes[0].set_title("Runtime tradeoff")
    axes[1].set_title("Memory tradeoff")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        f"Accuracy--cost tradeoff (best validation loss; threshold={threshold:g})",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(
        os.path.join(output_dir, "accuracy_cost_tradeoff.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return True


def write_csv(rows: List[Dict], path: str) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict], path: str, threshold: float) -> None:
    columns = [
        ("Initialization", "init_regime"),
        ("Initial parameters [a,b,c,d]", "initialization_params"),
        ("Mesh", "mesh"),
        ("Precision", "precision"),
        ("Final train loss", "final_train_loss"),
        ("Final val loss", "final_val_loss"),
        ("Best val loss", "best_val_loss"),
        ("Final parameter error", "final_param_err"),
        ("Peak memory (MB)", "peak_mem_mb"),
        ("Mean logged iteration time (s)", "mean_iter_time_s"),
        (f"Iteration to val loss ≤ {threshold:g}", "iter_to_val_threshold"),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Lotka--Volterra Uniform versus Graded Mesh\n\n")
        handle.write(
            "Both mesh conditions use the predictor--corrector recurrence. "
            "The data, initialization, beta, optimizer, and step size are shared.\n\n"
        )
        handle.write("| " + " | ".join(label for label, _ in columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = []
            for _, key in columns:
                value = row[key]
                if isinstance(value, float):
                    value = f"{value:.6g}"
                values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")

        handle.write("\nFinal learned parameters:\n\n")
        for row in rows:
            handle.write(
                f"- `{row['init_regime']}/{row['mesh']}_{row['precision']}`: "
                f"[{row['final_params']}]\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", help="paths to results.json files")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--val-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    paths = args.results or sorted(
        glob.glob(os.path.join(args.output_dir, "*", "*", "results.json"))
    )
    if len(paths) < 2:
        parser.error("provide at least two results.json files")

    rows = [load_summary(path, args.val_threshold) for path in paths]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    markdown_path = os.path.join(args.output_dir, "comparison.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args.val_threshold)
    plots_written = [
        write_convergence_plot(paths, args.output_dir, "iteration"),
        write_convergence_plot(paths, args.output_dir, "time"),
        write_efficiency_plot(paths, args.output_dir, args.val_threshold),
    ]

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if all(plots_written):
        print(f"Wrote {os.path.join(args.output_dir, 'validation_loss_vs_iteration.png')}")
        print(f"Wrote {os.path.join(args.output_dir, 'validation_loss_vs_time.png')}")
        print(f"Wrote {os.path.join(args.output_dir, 'accuracy_cost_tradeoff.png')}")
    else:
        print("Plot files were skipped because matplotlib is unavailable.")
    print("\nMesh/precision comparison:")
    for row in rows:
        print(
            f"  {row['init_regime']:>9} {row['mesh']:>7} {row['precision']:>4} | "
            f"val={row['final_val_loss']:.6e} | "
            f"param_err={row['final_param_err']:.6e} | "
            f"peak_MB={row['peak_mem_mb']:.1f}"
        )


if __name__ == "__main__":
    main()
