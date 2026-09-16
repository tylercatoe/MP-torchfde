#!/usr/bin/env python3
"""
Parse Peaks training logs, plot Train and Test MSE (log scale) across epochs,
and write a final-metrics summary table.
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
GRADED_RE = re.compile(r"graded_time=(True|False)", re.IGNORECASE)
PREDICTOR_CORRECTOR_RE = re.compile(r"predictor_corrector=(True|False)", re.IGNORECASE)
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


def parse_log(log_path: Path) -> dict:
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

    # Training stores each run in its own directory, e.g. ``adjoint`` or
    # ``graded-adjoint``.  Use that directory name so uniform and graded runs
    # remain distinct when plotted together.  Fall back to the logged
    # configuration when a log is supplied from a different layout.
    run_name = log_path.parent.name
    if run_name == mode_name or run_name == f"graded-{mode_name}":
        mode_name = run_name
    else:
        graded_match = GRADED_RE.search(text)
        if graded_match and graded_match.group(1).lower() == "true":
            mode_name = f"graded-{mode_name}"

    corrector_match = PREDICTOR_CORRECTOR_RE.search(text)
    if corrector_match and corrector_match.group(1).lower() == "true":
        mode_name = f"pc-{mode_name}"

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
        "log_file": str(log_path.name),
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
        "mode",
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
        "mode",
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
                row["mode"],
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
        lines.append(f"- {row['mode']}: {row['log_file']}")

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
    plt.figure(figsize=(10, 6))

    for row in rows:
        epochs = row["epochs"]
        values = row[metric_key]
        if not epochs:
            continue
        epochs_ds, values_ds = downsample_series(epochs, values, plot_every)
        plt.plot(epochs_ds, values_ds, linewidth=1.2, label=row["mode"])

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
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
        default=default_logs_dir,
        help="Directory containing training logs (default: this script's directory)",
    )
    parser.add_argument(
        "--log-glob",
        type=str,
        default="*/training.log",
        help="Glob pattern for selecting log files inside --logs-dir",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=evaluate_logs_dir / "peaks_train_mse_logscale.png",
        help="Output PNG path for training-MSE plot",
    )
    parser.add_argument(
        "--test-plot-out",
        type=Path,
        default=evaluate_logs_dir / "peaks_test_mse_logscale.png",
        help="Output PNG path for test-MSE plot",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=evaluate_logs_dir / "peaks_final_metrics_summary.csv",
        help="Output CSV path for final metrics summary",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=evaluate_logs_dir / "peaks_final_metrics_summary.md",
        help="Output Markdown path for final metrics summary",
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

    logs_dir = args.logs_dir.resolve()
    log_paths = sorted(logs_dir.glob(args.log_glob))
    if not log_paths:
        raise FileNotFoundError(
            f"No logs matched '{args.log_glob}' in {logs_dir}. "
            f"Try passing --logs-dir explicitly, e.g. --logs-dir '{default_logs_dir}'."
        )

    rows = [parse_log(path) for path in log_paths]
    rows.sort(key=lambda x: x["mode"])

    plot_out = args.plot_out if args.plot_out.is_absolute() else (logs_dir / args.plot_out)
    test_plot_out = (
        args.test_plot_out
        if args.test_plot_out.is_absolute()
        else (logs_dir / args.test_plot_out)
    )
    csv_out = args.csv_out if args.csv_out.is_absolute() else (logs_dir / args.csv_out)
    md_out = args.md_out if args.md_out.is_absolute() else (logs_dir / args.md_out)

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

    print(f"Parsed {len(rows)} logs from: {logs_dir}")
    print(f"Plot written to: {plot_out}")
    print(f"Test plot written to: {test_plot_out}")
    print(f"CSV summary written to: {csv_out}")
    print(f"Markdown summary written to: {md_out}")


if __name__ == "__main__":
    main()
