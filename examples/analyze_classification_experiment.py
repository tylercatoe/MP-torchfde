#!/usr/bin/env python3
"""Create consistent MNIST, FashionMNIST, and STL10 experiment reports."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC = r"(?:[0-9.eE+-]+|nan|inf|-inf)"
EPOCH_RE = re.compile(
    rf"Epoch\s+(\d+)\s+\|.*?Train Acc\s+({METRIC})\s+\|.*?"
    rf"Val Acc\s+({METRIC})\s+\|.*?Best\s+({METRIC})",
    re.IGNORECASE,
)
FINAL_RE = re.compile(
    rf"Final metrics\s+\|\s+Final Val Error\s+({METRIC})\s+\|\s+"
    rf"Best Val Error\s+({METRIC})\s+\|\s+(?:Max\s+)?Train Mem\s+({METRIC})\s+MB\s+\|\s+"
    rf"Train Time\s+({METRIC})s\s+\|\s*Infer Time\s+({METRIC})s\s+\|\s*"
    rf"Infer Peak Mem\s+({METRIC})\s+MB",
    re.IGNORECASE,
)
MODE_RE = re.compile(r"ModeConfig\(name='([^']+)'")
METHOD_RE = re.compile(r"(?<![A-Za-z0-9_])method='([^']+)'")
GRADED_RE = re.compile(r"graded_time=(True|False)", re.IGNORECASE)
CORRECTOR_RE = re.compile(r"predictor_corrector=(True|False)", re.IGNORECASE)
MP_DTYPE_RE = re.compile(r"mp_dtype=(?:torch\.)?([A-Za-z0-9_]+)", re.IGNORECASE)
PARAMETER_RE = re.compile(r"(?:Number of parameters|Total parameters):\s*([0-9,]+)")


@dataclass(frozen=True)
class Profile:
    name: str
    slug: str
    default_experiment: str
    log_globs: tuple[str, ...]
    report_subdir: str
    architecture: tuple[str, ...]
    beta: str
    final_time: str
    step_size: str
    f_description: str
    epochs: str
    batch_size: str
    initial_lr: str
    parameter_count: str
    train_plot_name: str
    val_plot_name: str
    preserve_heading: str | None = None


PROFILES = {
    "fashion-mnist": Profile(
        name="Fashion MNIST",
        slug="fashion_mnist",
        default_experiment="exp_mp_fashion_mnist",
        log_globs=("**/logs", "**/logs.log"),
        report_subdir="Evaluate Full Training Logs",
        architecture=("Same as the torchfde Neural FDE paper (and the MNIST example)",),
        beta="0.3",
        final_time="1.0",
        step_size="0.1",
        f_description="Convolution module",
        epochs="160",
        batch_size="128",
        initial_lr="0.1, decayed at the specified boundary epochs",
        parameter_count="208,266",
        train_plot_name="fashion_mnist_train_accuracy.png",
        val_plot_name="fashion_mnist_validation_accuracy.png",
        preserve_heading="## Fashion MNIST Final Time, T, Sweep Comparisions",
    ),
    "mnist": Profile(
        name="MNIST",
        slug="mnist",
        default_experiment="exp_mp_mnist",
        log_globs=("**/logs", "**/logs.log"),
        report_subdir="Evaluate Full Training Logs",
        architecture=("Same as the torchfde Neural FDE paper",),
        beta="0.5",
        final_time="20.0",
        step_size="0.1",
        f_description="Convolution module",
        epochs="60",
        batch_size="128",
        initial_lr="0.1, decayed at the specified boundary epochs",
        parameter_count="208,266",
        train_plot_name="mnist_train_acc.png",
        val_plot_name="mnist_test_acc.png",
    ),
    "stl10": Profile(
        name="STL10",
        slug="stl10",
        default_experiment="exp_mp_stl10",
        log_globs=("**/training.log",),
        report_subdir="Evaluate Logs",
        architecture=("STL10 convolutional Neural FDE classifier", "Width: 128"),
        beta="0.6",
        final_time="1.0",
        step_size="0.1",
        f_description="Convolution module",
        epochs="160",
        batch_size="16",
        initial_lr="0.05, decayed by the training schedule",
        parameter_count="3,144,970",
        train_plot_name="stl10_train_acc.png",
        val_plot_name="stl10_test_acc.png",
    ),
}


def logged_bool(pattern: re.Pattern[str], text: str, default: bool = False) -> bool:
    match = pattern.search(text)
    return default if match is None else match.group(1).lower() == "true"


def precision_name(mode: str, text: str) -> str:
    if mode in {"direct", "adjoint"}:
        return "float32"
    match = MP_DTYPE_RE.search(text)
    if match:
        return match.group(1).lower()
    return "bfloat16" if mode == "adjoint-mixed-bfloat" else "float16"


def configuration_label(mode: str, mesh: str, solver: str, precision: str) -> str:
    precision_label = {
        "float32": "FP32",
        "float16": "FP16",
        "bfloat16": "BF16",
    }.get(precision, precision)
    if mode == "direct":
        return f"Direct Predictor · {precision_label}"
    solver_label = "Predictor-Corrector" if solver == "predictor-corrector" else "Predictor"
    return f"{mesh.title()} {solver_label} · {precision_label}"


def parse_log(log_path: Path, experiment: str) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    path_text = "/".join(part.lower() for part in log_path.parts)
    mode_match = MODE_RE.search(text)
    if mode_match:
        mode = mode_match.group(1)
    else:
        mode = next(
            (name for name in ("adjoint-mixed-bfloat", "adjoint-mixed", "adjoint", "direct") if name in path_text),
            "unknown",
        )
    method_match = METHOD_RE.search(text)
    method = method_match.group(1) if method_match else "predictor-f"
    graded = logged_bool(GRADED_RE, text, "graded" in path_text)
    corrector = logged_bool(CORRECTOR_RE, text, "predictor-corrector" in path_text)
    mesh = "graded" if graded else "uniform"
    solver = "predictor-corrector" if corrector and mode != "direct" else "predictor"
    precision = precision_name(mode, text)

    epochs: list[int] = []
    train_acc: list[float] = []
    val_acc: list[float] = []
    best_acc: list[float] = []
    for match in EPOCH_RE.finditer(text):
        epochs.append(int(match.group(1)))
        train_acc.append(float(match.group(2)))
        val_acc.append(float(match.group(3)))
        best_acc.append(float(match.group(4)))

    parameter_match = PARAMETER_RE.search(text)
    row = {
        "experiment": experiment,
        "configuration": configuration_label(mode, mesh, solver, precision),
        "mode": mode,
        "method": method,
        "mesh": mesh,
        "solver": solver,
        "precision": precision,
        "predictor_corrector": corrector,
        "log_file": str(log_path),
        "parameter_count": parameter_match.group(1) if parameter_match else None,
        "epochs": epochs,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "plot_label": configuration_label(mode, mesh, solver, precision),
    }
    matches = list(FINAL_RE.finditer(text))
    if not matches:
        row.update(
            status="FAIL",
            failure_reason="training log has no Final metrics line",
            final_val_accuracy=None,
            final_val_error=None,
            best_val_accuracy=None,
            best_val_error=None,
            train_memory_mb=None,
            train_time_s=None,
            inference_time_s=None,
            inference_peak_mem_mb=None,
        )
        return row

    final = matches[-1]
    row.update(
        status="ok",
        failure_reason="",
        final_val_accuracy=1.0 - float(final.group(1)),
        final_val_error=float(final.group(1)),
        best_val_accuracy=1.0 - float(final.group(2)),
        best_val_error=float(final.group(2)),
        train_memory_mb=float(final.group(3)),
        train_time_s=float(final.group(4)),
        inference_time_s=float(final.group(5)),
        inference_peak_mem_mb=float(final.group(6)),
    )
    return row


def fixed_width_table(rows: list[dict]) -> str:
    headers = [
        "configuration", "backward mode", "mesh", "precision", "status", "final_acc",
        "best_acc", "train_mem_mb", "train_time_s", "inf_time_s", "inf_mem_mb",
    ]
    body = [
        [
            row["configuration"],
            "direct AG" if row["mode"] == "direct" else row["mode"],
            row["mesh"], row["precision"], row["status"],
            *( ["F"] * 6 if row["status"] != "ok" else [
                f'{row["final_val_accuracy"]:.4f}', f'{row["best_val_accuracy"]:.4f}',
                f'{row["train_memory_mb"]:.2f}', f'{row["train_time_s"]:.2f}',
                f'{row["inference_time_s"]:.2f}', f'{row["inference_peak_mem_mb"]:.2f}',
            ]),
        ]
        for row in rows
    ]
    widths = [len(value) for value in headers]
    for row in body:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    render = lambda values: " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))
    return "\n".join([render(headers), "-+-".join("-" * width for width in widths), *(render(row) for row in body)])


def memory_saving(reference: dict | None, candidates: list[dict]) -> str:
    candidates = [row for row in candidates if row["status"] == "ok"]
    if reference is None or reference["status"] != "ok" or not candidates or reference["train_memory_mb"] == 0:
        return "N/A"
    mixed = min(row["train_memory_mb"] for row in candidates)
    return f'{100.0 * (reference["train_memory_mb"] - mixed) / reference["train_memory_mb"]:.1f}\\%'


def find_row(rows: list[dict], *, mode: str | None = None, solver: str, mesh: str) -> dict | None:
    return next(
        (row for row in rows if row["status"] == "ok" and row["solver"] == solver and row["mesh"] == mesh and (mode is None or row["mode"] == mode)),
        None,
    )


def mixed_rows(rows: list[dict], *, solver: str, mesh: str) -> list[dict]:
    return [
        row for row in rows
        if row["solver"] == solver and row["mesh"] == mesh
        and row["mode"] in {"adjoint-mixed", "adjoint-mixed-bfloat"}
    ]


def write_markdown(rows: list[dict], out_path: Path, profile: Profile, train_plot: Path, val_plot: Path) -> None:
    suffix = ""
    if profile.preserve_heading and out_path.exists():
        old_text = out_path.read_text(encoding="utf-8")
        position = old_text.find(profile.preserve_heading)
        if position >= 0:
            suffix = "\n\n" + old_text[position:].rstrip() + "\n"

    direct = find_row(rows, mode="direct", solver="predictor", mesh="uniform")
    predictor_fp32 = find_row(rows, mode="adjoint", solver="predictor", mesh="uniform")
    corrector_fp32 = find_row(rows, mode="adjoint", solver="predictor-corrector", mesh="uniform")
    predictor_mixed = mixed_rows(rows, solver="predictor", mesh="uniform")
    corrector_mixed = mixed_rows(rows, solver="predictor-corrector", mesh="uniform")
    parsed_counts = {row["parameter_count"] for row in rows if row["parameter_count"]}
    parameter_count = parsed_counts.pop() if len(parsed_counts) == 1 else profile.parameter_count

    lines = [
        f"# {profile.name} Final Metrics Summary",
        "",
        "## Full Training Metrics",
        "",
        "```text",
        fixed_width_table(rows),
        "```",
        "",
        "Predictor:",
        f"- Adjoint MP memory savings compared to direct AG: ${memory_saving(direct, predictor_mixed)}$",
        f"- Adjoint MP memory savings compared to full precision adjoint: ${memory_saving(predictor_fp32, predictor_mixed)}$",
        "",
        "Predictor-Corrector:",
        f"- Adjoint MP memory savings compared to full precision adjoint: ${memory_saving(corrector_fp32, corrector_mixed)}$",
        "",
        "Experiment Parameters:",
        "- Network Architecture:",
        *(f"    - {item}" for item in profile.architecture),
        f"    - Model parameter count: {parameter_count}",
        "- FDE_Block:",
        f"    - Beta: {profile.beta}",
        f"    - T: {profile.final_time}",
        f"    - step_size: {profile.step_size}",
        f"    - $f$ in $D^\\beta z = f$: {profile.f_description}",
        "- Training Arguments:",
        f"    - Epochs: {profile.epochs}",
        f"    - Batch size: {profile.batch_size}",
        f"    - Initial LR: {profile.initial_lr}",
        "    - Momentum: 0.9",
        "    - Weight decay: 5e-4",
        "    - GPU: NVIDIA H200 (Palmetto)",
        "",
        f"Parameter count: {parameter_count}",
        "",
        "Note:",
        "- adjoint mode uses the custom adjoint in float32 throughout",
        "- adjoint-mixed mode uses float16 adjoint storage and the DynamicScaler",
        "- adjoint-mixed-bfloat uses bfloat16 adjoint storage without dynamic scaling",
        "- direct mode uses standard backpropagation in float32",
        "- graded and uniform specify the shared forward/backward time mesh",
    ]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    if failed_rows:
        lines.extend(["", "Failed Configurations:"])
        for row in failed_rows:
            lines.append(f'- {row["configuration"]}: {row["failure_reason"]}')
    lines.extend([
        "",
        "Training Plot (every logged epoch):",
        f'![Training plot for {profile.name}](./{train_plot.name} "{profile.name} training curves")',
        "",
        "Validation Accuracy Plot (every logged epoch):",
        f'![Validation plot for {profile.name}](./{val_plot.name} "{profile.name} validation curves")',
    ])
    out_path.write_text("\n".join(lines).rstrip() + "\n" + suffix, encoding="utf-8")


def write_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "experiment", "configuration", "mode", "method", "mesh", "solver", "precision",
        "predictor_corrector", "status", "failure_reason", "log_file", "final_val_accuracy", "final_val_error",
        "best_val_accuracy", "best_val_error", "train_memory_mb", "train_time_s",
        "inference_time_s", "inference_peak_mem_mb", "parameter_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def downsample(x: list[int], y: list[float], stride: int) -> tuple[list[int], list[float]]:
    if stride <= 1 or len(x) <= 1:
        return x, y
    out_x, out_y = x[::stride], y[::stride]
    if out_x[-1] != x[-1]:
        out_x.append(x[-1])
        out_y.append(y[-1])
    return out_x, out_y


def make_plot(rows: list[dict], out_path: Path, profile: Profile, metric: str, ylabel: str, stride: int) -> None:
    plt.figure(figsize=(12, 7))
    colors = {"float32": "tab:blue", "float16": "tab:orange", "bfloat16": "tab:green"}
    for row in rows:
        if row["status"] != "ok":
            continue
        if not row["epochs"]:
            continue
        x, y = downsample(row["epochs"], row[metric], stride)
        plt.plot(
            x, y,
            color="black" if row["mode"] == "direct" else colors.get(row["precision"], "gray"),
            linestyle="--" if row["mesh"] == "graded" else "-",
            linewidth=1.8 if row["predictor_corrector"] else 1.25,
            marker="o" if row["predictor_corrector"] else None,
            markevery=max(1, len(x) // 10) if row["predictor_corrector"] else None,
            markersize=3,
            label=row["plot_label"],
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{profile.name} {ylabel}")
    plt.ylim(0.0, 1.01)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(PROFILES), required=True)
    parser.add_argument("--logs-dir", action="append", type=Path, default=None)
    parser.add_argument("--log-glob", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-plot-out", type=Path, default=None)
    parser.add_argument("--val-plot-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--plot-every", type=int, default=1)
    args = parser.parse_args()

    profile = PROFILES[args.dataset]
    examples_dir = Path(__file__).resolve().parent
    dataset_dir = {
        "fashion-mnist": examples_dir / "FashionMNIST",
        "mnist": examples_dir / "MNIST",
        "stl10": examples_dir / "STL10",
    }[args.dataset]
    logs_dirs = [path.resolve() for path in (args.logs_dir or [dataset_dir / profile.default_experiment])]
    patterns = args.log_glob or list(profile.log_globs)
    rows: list[dict] = []
    seen: set[Path] = set()
    for logs_dir in logs_dirs:
        for pattern in patterns:
            for log_path in sorted(logs_dir.glob(pattern)):
                log_path = log_path.resolve()
                if log_path.is_file() and log_path not in seen:
                    seen.add(log_path)
                    rows.append(parse_log(log_path, logs_dir.name))
    if not rows:
        parser.error(f"no logs matching {patterns} under {', '.join(map(str, logs_dirs))}")

    solver_order = {"predictor": 0, "predictor-corrector": 1}
    mesh_order = {"uniform": 0, "graded": 1}
    precision_order = {"float32": 0, "float16": 1, "bfloat16": 2}
    rows.sort(key=lambda row: (
        -1 if row["mode"] == "direct" else solver_order.get(row["solver"], 9),
        mesh_order.get(row["mesh"], 9), precision_order.get(row["precision"], 9), row["experiment"],
    ))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["configuration"]] = counts.get(row["configuration"], 0) + 1
    for row in rows:
        if counts[row["configuration"]] > 1:
            row["plot_label"] = f'{row["configuration"]} [{row["experiment"]}]'

    output_dir = (args.output_dir or dataset_dir / profile.report_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(option: Path | None, default_name: str) -> Path:
        if option is None:
            return output_dir / default_name
        return option.resolve() if option.is_absolute() else output_dir / option

    train_plot = output_path(args.train_plot_out, profile.train_plot_name)
    val_plot = output_path(args.val_plot_out, profile.val_plot_name)
    csv_out = output_path(args.csv_out, f"{profile.slug}_final_metrics.csv")
    md_out = output_path(args.md_out, f"{profile.slug}_final_metrics.md")
    make_plot(rows, train_plot, profile, "train_acc", "Training Accuracy", args.plot_every)
    make_plot(rows, val_plot, profile, "val_acc", "Validation Accuracy", args.plot_every)
    write_csv(rows, csv_out)
    write_markdown(rows, md_out, profile, train_plot, val_plot)
    print(f"Parsed {len(rows)} logs")
    print(f"Wrote {train_plot}")
    print(f"Wrote {val_plot}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
