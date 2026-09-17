#!/usr/bin/env python3
"""
Parse one or more Peaks experiments, compare predictor and
predictor-corrector runs on uniform and graded meshes, plot Train and Test MSE
across epochs, and write CSV/Markdown final-metrics summaries.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|.*?Train MSE\s+([0-9.eE+-]+)\s+\|.*?"
    r"Test MSE\s+([0-9.eE+-]+)"
)
MODE_RE = re.compile(r"ModeConfig\(name='([^']+)'")
METHOD_RE = re.compile(r"method='([^']+)'")
GRADED_RE = re.compile(r"graded_time=(True|False)", re.IGNORECASE)
PREDICTOR_CORRECTOR_RE = re.compile(r"predictor_corrector=(True|False)", re.IGNORECASE)
MP_DTYPE_RE = re.compile(r"mp_dtype=(?:torch\.)?([A-Za-z0-9_]+)", re.IGNORECASE)
METRIC_RE = r"(?:[0-9.eE+-]+|nan|inf|-inf)"
FINAL_RE = re.compile(
    r"Final Results\s+\|\s+"
    r"Final Test MSE\s+(" + METRIC_RE + r")\s+\|\s+"
    r"Best Test MSE\s+(" + METRIC_RE + r")\s+\|\s+"
    r"Max Train Mem\s+(" + METRIC_RE + r")\s+MB\s+\|\s+"
    r"Train Time\s+(" + METRIC_RE + r")\s+s\s+\|\s+"
    r"Inference Time\s+(" + METRIC_RE + r")s\s+\|\s+"
    r"Inference Peak Mem\s+(" + METRIC_RE + r")\s+MB",
    re.IGNORECASE,
)


def _logged_bool(pattern: re.Pattern, text: str, default: bool = False) -> bool:
    match = pattern.search(text)
    if match is None:
        return default
    return match.group(1).lower() == "true"


def _precision_name(mode: str, text: str) -> str:
    # Direct and ordinary adjoint runs are intentionally full float32. Mixed
    # modes use the logged adjoint-storage dtype.
    if mode in {"direct", "adjoint"}:
        return "float32"

    match = MP_DTYPE_RE.search(text)
    if match:
        return match.group(1).lower()
    if mode == "adjoint-mixed-bfloat":
        return "bfloat16"
    if mode == "adjoint-mixed":
        return "float16"
    return "unknown"


def _configuration_label(mode: str, method: str, mesh: str, solver: str, precision: str) -> str:
    precision_label = {
        "float16": "FP16",
        "bfloat16": "BF16",
        "float32": "FP32",
        "float64": "FP64",
    }.get(precision, precision)

    if mode == "direct":
        method_label = method.replace("-", " ").title()
        return f"Direct {method_label} · {precision_label}"

    solver_label = {
        "predictor": "Predictor",
        "predictor-corrector": "Predictor-Corrector",
    }.get(solver, solver)
    return f"{mesh.title()} {solver_label} · {precision_label}"


def parse_log(log_path: Path, experiment: str) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    mode_name = None
    for line in lines:
        match = MODE_RE.search(line)
        if match:
            mode_name = match.group(1)
            break
    if mode_name is None:
        mode_name = log_path.stem.replace("_training", "")

    method_match = METHOD_RE.search(text)
    method = method_match.group(1) if method_match else "unknown"
    graded_time = _logged_bool(
        GRADED_RE,
        text,
        default=log_path.parent.name.startswith("graded-"),
    )
    predictor_corrector = _logged_bool(PREDICTOR_CORRECTOR_RE, text)
    mesh = "graded" if graded_time else "uniform"
    if mode_name == "direct":
        solver = "direct"
    elif method == "predictor-f":
        solver = "predictor-corrector" if predictor_corrector else "predictor"
    else:
        solver = method
    precision = _precision_name(mode_name, text)
    configuration = _configuration_label(mode_name, method, mesh, solver, precision)

    epochs = []
    train_mse = []
    test_mse = []
    for line in lines:
        match = EPOCH_RE.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_mse.append(float(match.group(2)))
            test_mse.append(float(match.group(3)))

    final_match = None
    for line in reversed(lines):
        match = FINAL_RE.search(line)
        if match:
            final_match = match
            break

    if final_match is None:
        raise ValueError(f"Could not parse final metrics from {log_path}")

    metrics = {
        "mode": mode_name,
        "method": method,
        "mesh": mesh,
        "solver": solver,
        "precision": precision,
        "predictor_corrector": predictor_corrector,
        "configuration": configuration,
        "plot_label": configuration,
        "experiment": experiment,
        "log_file": str(log_path),
        "final_test_mse": float(final_match.group(1)),
        "best_test_mse": float(final_match.group(2)),
        "train_memory_mb": float(final_match.group(3)),
        "train_time_s": float(final_match.group(4)),
        "inference_time_s": float(final_match.group(5)),
        "inference_peak_mem_mb": float(final_match.group(6)),
        "epochs": epochs,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
    return metrics


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "experiment",
        "configuration",
        "mode",
        "method",
        "mesh",
        "solver",
        "precision",
        "predictor_corrector",
        "log_file",
        "final_test_mse",
        "best_test_mse",
        "train_memory_mb",
        "train_time_s",
        "inference_time_s",
        "inference_peak_mem_mb",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def build_fixed_width_summary_table(rows: list[dict]) -> str:
    headers = [
        "experiment",
        "configuration",
        "mode",
        "mesh",
        "solver",
        "precision",
        "final_mse",
        "best_mse",
        "train_mem_mb",
        "train_time_s",
        "infer_time_s",
        "infer_mem_mb",
    ]

    body = []
    for row in rows:
        body.append(
            [
                row["experiment"],
                row["configuration"],
                row["mode"],
                row["mesh"],
                row["solver"],
                row["precision"],
                f'{row["final_test_mse"]:.6g}',
                f'{row["best_test_mse"]:.6g}',
                f'{row["train_memory_mb"]:.2f}',
                f'{row["train_time_s"]:.2f}',
                f'{row["inference_time_s"]:.4f}',
                f'{row["inference_peak_mem_mb"]:.2f}',
            ]
        )

    widths = [len(h) for h in headers]
    for r in body:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(r) for r in body)
    return "\n".join(lines)


def write_markdown(rows: list[dict], out_path: Path) -> None:
    table_text = build_fixed_width_summary_table(rows)
    lines = [
        "# Peaks Final Metrics Summary",
        "",
        "```text",
        table_text,
        "```",
        "",
        "Log files:",
    ]
    for row in rows:
        lines.append(
            f"- {row['experiment']} / {row['configuration']}: {row['log_file']}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def downsample_series(epochs: list[int], values: list[float], stride: int) -> tuple[list[int], list[float]]:
    if stride <= 1 or len(epochs) <= 1:
        return epochs, values

    sampled_epochs = epochs[::stride]
    sampled_values = values[::stride]
    if sampled_epochs[-1] != epochs[-1]:
        sampled_epochs.append(epochs[-1])
        sampled_values.append(values[-1])
    return sampled_epochs, sampled_values


def make_plot(
    rows: list[dict],
    out_path: Path,
    title: str,
    ylabel: str,
    metric_key: str,
    plot_every: int,
) -> None:
    plt.figure(figsize=(12, 7))

    colors = {
        "float32": "tab:blue",
        "float16": "tab:orange",
        "bfloat16": "tab:green",
    }
    seen_direct = False

    for row in rows:
        if row["mode"] == "direct":
            if seen_direct:
                continue
            seen_direct = True
        epochs = row["epochs"]
        values = row[metric_key]
        if not epochs:
            continue
        epochs_ds, values_ds = downsample_series(epochs, values, plot_every)
        is_corrector = row["predictor_corrector"]
        plt.plot(
            epochs_ds,
            values_ds,
            color="black" if row["mode"] == "direct" else colors.get(row["precision"]),
            linestyle="--" if row["mesh"] == "graded" else "-",
            linewidth=1.8 if is_corrector else 1.25,
            marker="o" if is_corrector else None,
            markevery=max(1, len(epochs_ds) // 10) if is_corrector else None,
            markersize=3,
            label=row["plot_label"],
        )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    evaluate_logs_dir = Path(__file__).resolve().parent
    default_logs_dir = evaluate_logs_dir.parent / "exp_mp_peaks"
    parser = argparse.ArgumentParser(description="Plot Peaks training logs and summarize final metrics.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory containing training logs. Repeat this option to compare "
            "multiple experiments. Defaults to examples/Peaks/exp_mp_peaks."
        ),
    )
    parser.add_argument(
        "--log-glob",
        type=str,
        default="*/training.log",
        help="Glob pattern for selecting log files inside --logs-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated plots and summaries. By default, a single "
            "experiment writes to its analysis/ subdirectory; a multi-experiment "
            "comparison writes under Evaluate Logs/peaks_combined_analysis."
        ),
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional output filename or path for the training-MSE plot",
    )
    parser.add_argument(
        "--test-plot-out",
        type=Path,
        default=None,
        help="Optional output filename or path for the test-MSE plot",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output filename or path for the CSV summary",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Optional output filename or path for the Markdown summary",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Peaks Training MSE Across Epochs",
        help="Plot title",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=50,
        help="Plot every Nth epoch point (1 means plot all points)",
    )
    args = parser.parse_args()

    logs_dirs = [path.resolve() for path in (args.logs_dir or [default_logs_dir])]
    rows = []
    matched_paths = set()
    for logs_dir in logs_dirs:
        for log_path in sorted(logs_dir.glob(args.log_glob)):
            resolved_path = log_path.resolve()
            if resolved_path in matched_paths:
                continue
            matched_paths.add(resolved_path)
            rows.append(parse_log(resolved_path, logs_dir.name))

    if not rows:
        searched = ", ".join(str(path) for path in logs_dirs)
        raise FileNotFoundError(
            f"No logs matched '{args.log_glob}' in: {searched}. "
            f"Try passing --logs-dir explicitly, e.g. --logs-dir '{default_logs_dir}'."
        )

    precision_order = {"float32": 0, "float16": 1, "bfloat16": 2}
    mesh_order = {"uniform": 0, "graded": 1}
    solver_order = {"direct": 0, "predictor": 1, "predictor-corrector": 2}
    rows.sort(
        key=lambda row: (
            solver_order.get(row["solver"], 3),
            mesh_order.get(row["mesh"], 2),
            precision_order.get(row["precision"], 3),
            row["experiment"],
        )
    )

    # If multiple roots contain the same non-direct configuration, include the
    # experiment directory in its legend entry. Repeated direct baselines are
    # retained in the tables but drawn only once by make_plot().
    label_counts = {}
    for row in rows:
        label_counts[row["configuration"]] = label_counts.get(row["configuration"], 0) + 1
    for row in rows:
        if label_counts[row["configuration"]] > 1 and row["mode"] != "direct":
            row["plot_label"] = f'{row["configuration"]} [{row["experiment"]}]'

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif len(logs_dirs) == 1:
        output_dir = logs_dirs[0] / "analysis"
    else:
        output_dir = evaluate_logs_dir / "peaks_combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(option: Path | None, default_name: str) -> Path:
        if option is None:
            return output_dir / default_name
        return option if option.is_absolute() else output_dir / option

    plot_out = output_path(args.plot_out, "peaks_train_mse_logscale.png")
    test_plot_out = output_path(args.test_plot_out, "peaks_test_mse_logscale.png")
    csv_out = output_path(args.csv_out, "peaks_final_metrics_summary.csv")
    md_out = output_path(args.md_out, "peaks_final_metrics_summary.md")

    make_plot(
        rows,
        plot_out,
        args.title,
        "Train MSE (log scale)",
        "train_mse",
        args.plot_every,
    )
    make_plot(
        rows,
        test_plot_out,
        "Peaks Test MSE Across Epochs",
        "Test MSE (log scale)",
        "test_mse",
        args.plot_every,
    )
    write_csv(rows, csv_out)
    write_markdown(rows, md_out)

    print(f"Parsed {len(rows)} logs from:")
    for logs_dir in logs_dirs:
        print(f"  - {logs_dir}")
    print(f"Plot written to: {plot_out}")
    print(f"Test plot written to: {test_plot_out}")
    print(f"CSV summary written to: {csv_out}")
    print(f"Markdown summary written to: {md_out}")


if __name__ == "__main__":
    main()
