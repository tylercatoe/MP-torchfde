#!/usr/bin/env python
"""Create a CSV and Markdown summary from Lotka--Volterra result files."""

import argparse
import csv
import glob
import json
import math
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
        "configuration": (
            f"{data['init_regime']} · {data.get('method', 'predictor-corrector')} · "
            f"{data['mesh']} · {data['precision'].upper()}"
        ),
        "init_regime": data["init_regime"],
        "mesh": data["mesh"],
        "method": data.get("method", "predictor-corrector"),
        "precision": data["precision"],
        "beta": data["beta"],
        "t_end": data.get("t_end"),
        "step_size": data["step_size"],
        "data_step_size": data.get("data_step_size"),
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "noise_std": data.get("noise_std"),
        "learning_rate": data.get("learning_rate"),
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
        "estimated_train_time_s": (
            final["iter"]
            * sum(item["iter_time_s"] for item in records)
            / len(records)
        ),
        "iter_to_val_threshold": (
            threshold_records[0]["iter"] if threshold_records else "NA"
        ),
        "final_params": ", ".join(f"{value:.6f}" for value in final["params"]),
        "initialization_params": ", ".join(
            f"{value:.6f}" for value in data["initialization_params"]
        ),
        "true_params": ", ".join(f"{value:.6f}" for value in data["true_params"]),
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


def _parameter_history(run: Dict) -> tuple[List[int], List[List[float]]]:
    """Return parameter history, including the saved initialization at step 0."""
    records = run["iterations"]
    iterations: List[int] = []
    parameters: List[List[float]] = []

    initial = run.get("initial_params", run.get("initialization_params"))
    if initial is not None and (not records or records[0]["iter"] != 0):
        iterations.append(0)
        parameters.append(initial)

    iterations.extend(record["iter"] for record in records)
    parameters.extend(record["params"] for record in records)
    return iterations, parameters


def _line_label(run: Dict, include_method: bool) -> str:
    """Build one consistent legend label for a solver configuration."""
    label = f"{run['mesh']} / {run['precision'].upper()}"
    if include_method:
        label += f" / {run.get('method', 'predictor-corrector')}"
    return label


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
    line_styles = {"predictor": "-", "predictor-corrector": "--"}

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
            for run in sorted(matching, key=lambda item: (item.get("method", ""), item["mesh"])):
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
                    linestyle=line_styles.get(run.get("method", "predictor-corrector"), "-"),
                    label=f"{run['mesh']} / {run.get('method', 'predictor-corrector')}",
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


def write_parameter_trajectories_plot(paths: List[str], output_dir: str) -> bool:
    """Plot each learned parameter against iteration and its true value."""
    plt = _plotting_modules()
    if plt is None:
        return False

    runs = [load_run(path) for path in paths]
    preferred_order = ["near_true", "worse"]
    available = {run["init_regime"] for run in runs}
    initializations = [name for name in preferred_order if name in available]
    initializations.extend(sorted(available - set(initializations)))
    parameter_names = ("a", "b", "c", "d")
    colors = {"uniform": "tab:blue", "graded": "tab:orange"}
    line_styles = {"predictor": "-", "predictor-corrector": "--"}
    markers = {"fp32": None, "fp16": "o"}
    methods = {run.get("method", "predictor-corrector") for run in runs}
    include_method = len(methods) > 1

    fig, axes = plt.subplots(
        len(initializations),
        len(parameter_names),
        squeeze=False,
        figsize=(4.0 * len(parameter_names), 3.5 * len(initializations)),
        sharex=True,
    )

    for row_index, init_regime in enumerate(initializations):
        matching = [run for run in runs if run["init_regime"] == init_regime]
        for parameter_index, parameter_name in enumerate(parameter_names):
            ax = axes[row_index][parameter_index]
            for run in sorted(
                matching, key=lambda item: (item.get("method", ""), item["mesh"], item["precision"])
            ):
                iterations, parameter_history = _parameter_history(run)
                values = [params[parameter_index] for params in parameter_history]
                ax.plot(
                    iterations,
                    values,
                    color=colors.get(run["mesh"], "black"),
                    linestyle=line_styles.get(run.get("method", "predictor-corrector"), "-"),
                    marker=markers.get(run["precision"]),
                    markevery=max(1, len(iterations) // 8),
                    markersize=3,
                    linewidth=1.8,
                    label=_line_label(run, include_method),
                )

            true_value = matching[0]["true_params"][parameter_index]
            ax.axhline(
                true_value,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="true value",
            )
            ax.set_title(f"{parameter_name} (true={true_value:g})")
            ax.set_xlabel("Training iteration")
            ax.grid(True, alpha=0.25)
            if parameter_index == 0:
                ax.set_ylabel(f"{init_regime}\nParameter value")

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=min(5, len(unique)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Lotka--Volterra parameter convergence", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(
        os.path.join(output_dir, "parameter_trajectories.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return True


def write_relative_parameter_error_plot(paths: List[str], output_dir: str) -> bool:
    """Plot relative L2 parameter error against training iteration."""
    plt = _plotting_modules()
    if plt is None:
        return False

    runs = [load_run(path) for path in paths]
    preferred_order = ["near_true", "worse"]
    available = {run["init_regime"] for run in runs}
    initializations = [name for name in preferred_order if name in available]
    initializations.extend(sorted(available - set(initializations)))
    colors = {"uniform": "tab:blue", "graded": "tab:orange"}
    line_styles = {"predictor": "-", "predictor-corrector": "--"}
    markers = {"fp32": None, "fp16": "o"}
    methods = {run.get("method", "predictor-corrector") for run in runs}
    include_method = len(methods) > 1

    fig, axes_grid = plt.subplots(
        1,
        len(initializations),
        squeeze=False,
        figsize=(5.5 * len(initializations), 4.2),
        sharey=True,
    )
    axes = axes_grid[0]

    for column_index, init_regime in enumerate(initializations):
        ax = axes[column_index]
        matching = [run for run in runs if run["init_regime"] == init_regime]
        for run in sorted(
            matching, key=lambda item: (item.get("method", ""), item["mesh"], item["precision"])
        ):
            iterations, parameter_history = _parameter_history(run)
            true_params = run["true_params"]
            true_norm = math.sqrt(sum(value * value for value in true_params))
            relative_errors = [
                max(
                    math.sqrt(
                        sum(
                            (value - truth) ** 2
                            for value, truth in zip(params, true_params)
                        )
                    )
                    / true_norm,
                    1e-16,
                )
                for params in parameter_history
            ]
            ax.plot(
                iterations,
                relative_errors,
                color=colors.get(run["mesh"], "black"),
                linestyle=line_styles.get(run.get("method", "predictor-corrector"), "-"),
                marker=markers.get(run["precision"]),
                markevery=max(1, len(iterations) // 8),
                markersize=3,
                linewidth=1.8,
                label=_line_label(run, include_method),
            )

        ax.set_yscale("log")
        ax.set_xlabel("Training iteration")
        ax.set_title(init_regime)
        ax.grid(True, which="both", alpha=0.25)
        if column_index == 0:
            ax.set_ylabel(r"Relative parameter error $\|\theta-\theta^*\|_2/\|\theta^*\|_2$")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=min(4, len(unique)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle("Lotka--Volterra relative parameter error", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(
        os.path.join(output_dir, "relative_parameter_error_vs_iteration.png"),
        dpi=180,
        bbox_inches="tight",
    )
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
                f"{run['mesh']} / {run.get('method', 'predictor-corrector')}"
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
        ("configuration", "configuration"),
        ("final_train", "final_train_loss"),
        ("final_val", "final_val_loss"),
        ("best_val", "best_val_loss"),
        ("best_iter", "best_val_iter"),
        ("param_err", "final_param_err"),
        ("peak_mem_mb", "peak_mem_mb"),
        ("mean_iter_s", "mean_iter_time_s"),
        ("est_train_s", "estimated_train_time_s"),
        (f"iter_to_{threshold:g}", "iter_to_val_threshold"),
    ]

    def formatted(value) -> str:
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    table_rows = [[formatted(row[key]) for _, key in columns] for row in rows]
    widths = [len(label) for label, _ in columns]
    for table_row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, table_row)]

    def render(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    table = "\n".join(
        [
            render([label for label, _ in columns]),
            "-+-".join("-" * width for width in widths),
            *(render(table_row) for table_row in table_rows),
        ]
    )

    first = rows[0]
    lines = [
        "# Lotka--Volterra Final Metrics Summary",
        "",
        "## Full Training Metrics",
        "",
        "```text",
        table,
        "```",
        "",
        "FP16 memory savings compared with FP32:",
    ]
    for init_regime in ("near_true", "worse"):
        for method in ("predictor", "predictor-corrector"):
            for mesh in ("uniform", "graded"):
                matching = [
                    row for row in rows
                    if row["init_regime"] == init_regime
                    and row["method"] == method and row["mesh"] == mesh
                ]
                fp32 = next((row for row in matching if row["precision"] == "fp32"), None)
                fp16 = next((row for row in matching if row["precision"] == "fp16"), None)
                if fp32 and fp16 and fp32["peak_mem_mb"]:
                    saving = 100.0 * (fp32["peak_mem_mb"] - fp16["peak_mem_mb"]) / fp32["peak_mem_mb"]
                    lines.append(f"- {init_regime}, {method}, {mesh}: ${saving:.1f}\\%$")

    lines.extend(
        [
            "",
            "Experiment Parameters:",
            "- Fractional Lotka--Volterra system:",
            "    - $D^\\beta x = x(a-cy)$",
            "    - $D^\\beta y = -y(b-dx)$",
            f"    - True parameters $[a,b,c,d]$: [{first['true_params']}]",
            f"    - Beta: {first['beta']}",
            f"    - T: {first['t_end']}",
            f"    - Training step size: {first['step_size']}",
            f"    - Synthetic-data step size: {first['data_step_size']}",
            "- Training Arguments:",
            f"    - Iterations: {first['final_iter']}",
            f"    - Training trajectories: {first['n_train']}",
            f"    - Validation trajectories: {first['n_val']}",
            f"    - Noise standard deviation: {first['noise_std']}",
            f"    - Learning rate: {first['learning_rate']}",
            f"    - Seed: {first['seed']}",
            "",
            "Initialization and Final Learned Parameters:",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['configuration']}`: initial [{row['initialization_params']}], "
            f"final [{row['final_params']}]"
        )

    lines.extend(
        [
            "",
            "Note:",
            "- All configurations use the same seeded train/validation data and initialization within each initialization regime.",
            "- Predictor and predictor-corrector are tested independently on uniform and double-graded meshes.",
            "- FP16 uses the safe mixed-precision custom adjoint; FP32 is the full-precision custom adjoint.",
            "- `param_err` is the final mean absolute error in the four learned parameters.",
            "- `est_train_s` extrapolates mean measured iteration time across all training iterations.",
            "",
            "Validation Loss versus Iteration:",
            "![Validation loss versus iteration](./validation_loss_vs_iteration.png)",
            "",
            "Validation Loss versus Estimated Time:",
            "![Validation loss versus time](./validation_loss_vs_time.png)",
            "",
            "Parameter Trajectories:",
            "![Parameter trajectories](./parameter_trajectories.png)",
            "",
            "Relative Parameter Error:",
            "![Relative parameter error](./relative_parameter_error_vs_iteration.png)",
            "",
            "Accuracy--Cost Tradeoff:",
            "![Accuracy cost tradeoff](./accuracy_cost_tradeoff.png)",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", help="paths to results.json files")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--val-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    # Match only the current init/method/config layout. This deliberately
    # ignores stale result files from the former init/config layout.
    paths = args.results or sorted(
        glob.glob(os.path.join(args.output_dir, "*", "*", "*", "results.json"))
    )
    if len(paths) < 2:
        parser.error("provide at least two results.json files")

    rows = [load_summary(path, args.val_threshold) for path in paths]
    init_order = {"near_true": 0, "worse": 1}
    method_order = {"predictor": 0, "predictor-corrector": 1}
    mesh_order = {"uniform": 0, "graded": 1}
    precision_order = {"fp32": 0, "fp16": 1}
    rows.sort(key=lambda row: (
        init_order.get(row["init_regime"], 9), method_order.get(row["method"], 9),
        mesh_order.get(row["mesh"], 9), precision_order.get(row["precision"], 9),
    ))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    markdown_path = os.path.join(args.output_dir, "comparison.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args.val_threshold)
    plots_written = [
        write_convergence_plot(paths, args.output_dir, "iteration"),
        write_convergence_plot(paths, args.output_dir, "time"),
        write_parameter_trajectories_plot(paths, args.output_dir),
        write_relative_parameter_error_plot(paths, args.output_dir),
        write_efficiency_plot(paths, args.output_dir, args.val_threshold),
    ]

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if all(plots_written):
        print(f"Wrote {os.path.join(args.output_dir, 'validation_loss_vs_iteration.png')}")
        print(f"Wrote {os.path.join(args.output_dir, 'validation_loss_vs_time.png')}")
        print(f"Wrote {os.path.join(args.output_dir, 'parameter_trajectories.png')}")
        print(
            f"Wrote {os.path.join(args.output_dir, 'relative_parameter_error_vs_iteration.png')}"
        )
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
