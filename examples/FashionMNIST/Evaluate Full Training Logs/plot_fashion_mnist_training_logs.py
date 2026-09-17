#!/usr/bin/env python3
"""Compare FashionMNIST predictor and predictor-corrector training logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_RE = r"(?:[0-9.eE+-]+|nan|inf|-inf)"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|.*?Train Acc\s+(" + METRIC_RE + r")\s+\|.*?"
    r"Val Acc\s+(" + METRIC_RE + r")\s+\|.*?Best\s+(" + METRIC_RE + r")",
    re.IGNORECASE,
)
FINAL_RE = re.compile(
    r"Final metrics\s+\|\s+"
    r"Final Val Error\s+(" + METRIC_RE + r")\s+\|\s+"
    r"Best Val Error\s+(" + METRIC_RE + r")\s+\|\s+"
    r"Train Mem\s+(" + METRIC_RE + r")\s+MB\s+\|\s+"
    r"Train Time\s+(" + METRIC_RE + r")s\s+\|\s*"
    r"Infer Time\s+(" + METRIC_RE + r")s\s+\|\s*"
    r"Infer Peak Mem\s+(" + METRIC_RE + r")\s+MB",
    re.IGNORECASE,
)
MODE_RE = re.compile(r"ModeConfig\(name='([^']+)'")
METHOD_RE = re.compile(r"(?<![A-Za-z0-9_])method='([^']+)'")
GRADED_RE = re.compile(r"graded_time=(True|False)", re.IGNORECASE)
PREDICTOR_CORRECTOR_RE = re.compile(r"predictor_corrector=(True|False)", re.IGNORECASE)
MP_DTYPE_RE = re.compile(r"mp_dtype=(?:torch\.)?([A-Za-z0-9_]+)", re.IGNORECASE)


def logged_bool(pattern: re.Pattern, text: str, default: bool = False) -> bool:
    match = pattern.search(text)
    return default if match is None else match.group(1).lower() == "true"


def precision_name(mode: str, text: str) -> str:
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


def configuration_label(mode: str, method: str, mesh: str, solver: str, precision: str) -> str:
    precision_label = {
        "float16": "FP16",
        "bfloat16": "BF16",
        "float32": "FP32",
        "float64": "FP64",
    }.get(precision, precision)
    if mode == "direct":
        return f"Direct {method.replace('-', ' ').title()} · {precision_label}"
    solver_label = {
        "predictor": "Predictor",
        "predictor-corrector": "Predictor-Corrector",
    }.get(solver, solver)
    return f"{mesh.title()} {solver_label} · {precision_label}"


def parse_log(log_path: Path, experiment: str) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    mode_match = MODE_RE.search(text)
    mode = mode_match.group(1) if mode_match else log_path.parent.name
    method_match = METHOD_RE.search(text)
    method = method_match.group(1) if method_match else "unknown"
    graded_time = logged_bool(
        GRADED_RE,
        text,
        default=log_path.parent.name.startswith("graded-"),
    )
    predictor_corrector = logged_bool(PREDICTOR_CORRECTOR_RE, text)
    mesh = "graded" if graded_time else "uniform"
    if mode == "direct":
        solver = "direct"
    elif method == "predictor-f":
        solver = "predictor-corrector" if predictor_corrector else "predictor"
    else:
        solver = method
    precision = precision_name(mode, text)
    configuration = configuration_label(mode, method, mesh, solver, precision)

    epochs = []
    train_acc = []
    val_acc = []
    best_acc = []
    for match in EPOCH_RE.finditer(text):
        epochs.append(int(match.group(1)))
        train_acc.append(float(match.group(2)))
        val_acc.append(float(match.group(3)))
        best_acc.append(float(match.group(4)))

    final_matches = list(FINAL_RE.finditer(text))
    if not final_matches:
        raise ValueError(f"Could not parse final metrics from {log_path}")
    final_match = final_matches[-1]
    final_val_error = float(final_match.group(1))
    best_val_error = float(final_match.group(2))

    return {
        "experiment": experiment,
        "configuration": configuration,
        "plot_label": configuration,
        "mode": mode,
        "method": method,
        "mesh": mesh,
        "solver": solver,
        "precision": precision,
        "predictor_corrector": predictor_corrector,
        "log_file": str(log_path),
        "final_val_accuracy": 1.0 - final_val_error,
        "final_val_error": final_val_error,
        "best_val_accuracy": 1.0 - best_val_error,
        "best_val_error": best_val_error,
        "train_memory_mb": float(final_match.group(3)),
        "train_time_s": float(final_match.group(4)),
        "inference_time_s": float(final_match.group(5)),
        "inference_peak_mem_mb": float(final_match.group(6)),
        "epochs": epochs,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_acc": best_acc,
    }


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
        "final_val_accuracy",
        "final_val_error",
        "best_val_accuracy",
        "best_val_error",
        "train_memory_mb",
        "train_time_s",
        "inference_time_s",
        "inference_peak_mem_mb",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def fixed_width_table(rows: list[dict]) -> str:
    headers = [
        "experiment",
        "configuration",
        "final_acc",
        "best_acc",
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
                f'{row["final_val_accuracy"]:.4f}',
                f'{row["best_val_accuracy"]:.4f}',
                f'{row["train_memory_mb"]:.2f}',
                f'{row["train_time_s"]:.2f}',
                f'{row["inference_time_s"]:.4f}',
                f'{row["inference_peak_mem_mb"]:.2f}',
            ]
        )

    widths = [len(header) for header in headers]
    for row in body:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *[format_row(row) for row in body]])


def write_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# FashionMNIST Final Metrics Summary",
        "",
        "```text",
        fixed_width_table(rows),
        "```",
        "",
        "Log files:",
    ]
    for row in rows:
        lines.append(f"- {row['experiment']} / {row['configuration']}: {row['log_file']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def downsample(epochs: list[int], values: list[float], stride: int) -> tuple[list[int], list[float]]:
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
    metric: str,
    plot_every: int,
) -> None:
    plt.figure(figsize=(12, 7))
    colors = {"float32": "tab:blue", "float16": "tab:orange", "bfloat16": "tab:green"}
    seen_direct = False
    for row in rows:
        if row["mode"] == "direct":
            if seen_direct:
                continue
            seen_direct = True
        if not row["epochs"]:
            continue
        epochs, values = downsample(row["epochs"], row[metric], plot_every)
        is_corrector = row["predictor_corrector"]
        plt.plot(
            epochs,
            values,
            color="black" if row["mode"] == "direct" else colors.get(row["precision"]),
            linestyle="--" if row["mesh"] == "graded" else "-",
            linewidth=1.8 if is_corrector else 1.25,
            marker="o" if is_corrector else None,
            markevery=max(1, len(epochs) // 10) if is_corrector else None,
            markersize=3,
            label=row["plot_label"],
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.01)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_logs_dir = script_dir.parent / "exp_mp_fashion_mnist"
    parser = argparse.ArgumentParser(description="Plot and summarize FashionMNIST training logs.")
    parser.add_argument(
        "--logs-dir",
        action="append",
        type=Path,
        default=None,
        help="Experiment directory. Repeat to compare multiple experiments.",
    )
    parser.add_argument(
        "--log-glob",
        action="append",
        default=None,
        help=(
            "Recursive glob used to find training logs. Repeat for multiple patterns. "
            "By default both **/logs and **/logs.log are searched."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-plot-out", type=Path, default=None)
    parser.add_argument("--val-plot-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--plot-every", type=int, default=1)
    args = parser.parse_args()

    logs_dirs = [path.resolve() for path in (args.logs_dir or [default_logs_dir])]
    log_globs = args.log_glob or ["**/logs", "**/logs.log"]
    rows = []
    matched_paths = set()
    for logs_dir in logs_dirs:
        for log_glob in log_globs:
            for log_path in sorted(logs_dir.glob(log_glob)):
                resolved_path = log_path.resolve()
                if resolved_path in matched_paths or not resolved_path.is_file():
                    continue
                matched_paths.add(resolved_path)
                rows.append(parse_log(resolved_path, logs_dir.name))

    if not rows:
        searched = ", ".join(str(path) for path in logs_dirs)
        patterns = ", ".join(log_globs)
        raise FileNotFoundError(f"No logs matched [{patterns}] in: {searched}")

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
        output_dir = script_dir / "fashion_mnist_combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(option: Path | None, default_name: str) -> Path:
        if option is None:
            return output_dir / default_name
        return option if option.is_absolute() else output_dir / option

    train_plot = output_path(args.train_plot_out, "fashion_mnist_train_accuracy.png")
    val_plot = output_path(args.val_plot_out, "fashion_mnist_validation_accuracy.png")
    csv_out = output_path(args.csv_out, "fashion_mnist_final_metrics.csv")
    md_out = output_path(args.md_out, "fashion_mnist_final_metrics.md")

    make_plot(rows, train_plot, "FashionMNIST Training Accuracy", "Training accuracy", "train_acc", args.plot_every)
    make_plot(rows, val_plot, "FashionMNIST Validation Accuracy", "Validation accuracy", "val_acc", args.plot_every)
    write_csv(rows, csv_out)
    write_markdown(rows, md_out)

    print(f"Parsed {len(rows)} logs from:")
    for logs_dir in logs_dirs:
        print(f"  - {logs_dir}")
    print(f"Training plot written to: {train_plot}")
    print(f"Validation plot written to: {val_plot}")
    print(f"CSV summary written to: {csv_out}")
    print(f"Markdown summary written to: {md_out}")


if __name__ == "__main__":
    main()
