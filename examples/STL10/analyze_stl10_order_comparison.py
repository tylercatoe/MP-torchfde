#!/usr/bin/env python3
"""Analyze matched integer-order ODE and fractional-order FDE STL10 runs."""

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Time\s+(?P<time>[-+0-9.eE]+)s\s*\|\s*"
    r"Peak Mem\s+(?P<mem>[-+0-9.eE]+) MB\s*\|.*?"
    r"Train Acc\s+(?P<train_acc>[-+0-9.eE]+)\s*\|\s*"
    r"Val Acc\s+(?P<val_acc>[-+0-9.eE]+)\s*\|\s*"
    r"Best\s+(?P<best_acc>[-+0-9.eE]+)"
    r"(?:\s*\|\s*Wall Time\s+(?P<wall_time>[-+0-9.eE]+)s)?"
)
FINAL_RE = re.compile(
    r"Final metrics\s*\|.*?"
    r"Max Train Mem\s+(?P<train_mem>[-+0-9.eE]+) MB\s*\|\s*"
    r"Train Time\s+(?P<train_time>[-+0-9.eE]+)s\s*\|\s*"
    r"Infer Time\s+(?P<infer_time>[-+0-9.eE]+)s\s*\|\s*"
    r"Infer Peak Mem\s+(?P<infer_mem>[-+0-9.eE]+) MB"
)


@dataclass
class Run:
    equation: str
    beta: Optional[float]
    method: str
    precision: str
    job_id: str
    result_path: Path
    source: str
    epochs: List[Dict[str, float]] = field(default_factory=list)
    final: Dict[str, float] = field(default_factory=dict)
    status: str = "missing"

    @property
    def final_train_acc(self) -> Optional[float]:
        return self.epochs[-1].get("train_acc") if self.epochs else None

    @property
    def final_val_acc(self) -> Optional[float]:
        return self.epochs[-1].get("val_acc") if self.epochs else None

    @property
    def best_val_acc(self) -> Optional[float]:
        return max((row["val_acc"] for row in self.epochs), default=None)

    @property
    def best_epoch(self) -> Optional[int]:
        if not self.epochs:
            return None
        return int(max(self.epochs, key=lambda row: row["val_acc"])["epoch"])

    @property
    def final_val_error(self) -> Optional[float]:
        return 1.0 - self.final_val_acc if self.final_val_acc is not None else None

    @property
    def best_val_error(self) -> Optional[float]:
        return 1.0 - self.best_val_acc if self.best_val_acc is not None else None

    @property
    def generalization_gap(self) -> Optional[float]:
        if self.final_train_acc is None or self.final_val_acc is None:
            return None
        return self.final_train_acc - self.final_val_acc

    @property
    def train_memory(self) -> Optional[float]:
        if self.equation == "ode":
            values = [row.get("max_memory_mb") for row in self.epochs]
            values = [value for value in values if value is not None]
            return max(values, default=None)
        return self.final.get("train_mem")

    @property
    def train_time(self) -> Optional[float]:
        if self.epochs:
            # Prefer the common wall-clock measurement; retain legacy
            # fallbacks for result files produced before this field existed.
            wall_time = self.epochs[-1].get("wall_time_s")
            if wall_time is not None and math.isfinite(wall_time):
                return wall_time
        if self.equation == "ode":
            return self.epochs[-1].get("compute_time_s") if self.epochs else None
        return self.final.get("train_time")

    @property
    def infer_time(self) -> Optional[float]:
        return self.final.get("infer_time")

    @property
    def infer_memory(self) -> Optional[float]:
        return self.final.get("infer_mem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Comparison manifest CSV")
    parser.add_argument("--expected-epoch", type=int, default=160)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="stl10_order_comparison")
    parser.add_argument("--targets", default="0.40,0.50")
    return parser.parse_args()


def parse_targets(raw: str) -> List[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def float_or_none(value: object) -> Optional[float]:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_fde_log(run: Run, expected_epoch: int) -> None:
    if not run.result_path.exists():
        return
    try:
        lines = run.result_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return

    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            row = {}
            for key, value in epoch_match.groupdict().items():
                parsed = float_or_none(value)
                if parsed is not None:
                    row[key] = parsed
            if "wall_time" in row:
                row["wall_time_s"] = row.pop("wall_time")
            run.epochs.append(row)
        final_match = FINAL_RE.search(line)
        if final_match:
            run.final = {key: float(value) for key, value in final_match.groupdict().items()}

    if run.final and run.epochs and int(run.epochs[-1]["epoch"]) >= expected_epoch:
        run.status = "complete"
    elif run.epochs:
        run.status = "incomplete"
    else:
        run.status = "failed"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            return list(csv.DictReader(input_file))
    except (OSError, csv.Error):
        return []


def parse_ode_result(run: Run, expected_epoch: int) -> None:
    args_candidates = list(run.result_path.rglob("args.csv")) if run.result_path.exists() else []
    selected_dir: Optional[Path] = None
    selected_args: Optional[Dict[str, str]] = None

    for args_path in args_candidates:
        rows = read_csv_rows(args_path)
        if not rows:
            continue
        args_row = rows[0]
        if run.job_id and str(args_row.get("job_id", "")) == run.job_id:
            selected_dir = args_path.parent
            selected_args = args_row
            break

    if selected_dir is None and len(args_candidates) == 1:
        selected_dir = args_candidates[0].parent
        rows = read_csv_rows(args_candidates[0])
        selected_args = rows[0] if rows else None

    if selected_dir is None:
        return

    metric_csv: Optional[Path] = None
    for candidate in selected_dir.glob("*.csv"):
        if candidate.name == "args.csv":
            continue
        rows = read_csv_rows(candidate)
        if rows and {"epoch", "val_acc"}.issubset(rows[0]):
            metric_csv = candidate
            break
    if metric_csv is None:
        return

    for row in read_csv_rows(metric_csv):
        epoch = float_or_none(row.get("epoch"))
        val_acc = float_or_none(row.get("val_acc"))
        train_acc = float_or_none(row.get("train_acc"))
        memory = float_or_none(row.get("max_memory_mb"))
        fwd_sum = float_or_none(row.get("time_fwd_sum")) or 0.0
        bwd_sum = float_or_none(row.get("time_bwd_sum")) or 0.0
        wall_time = float_or_none(row.get("wall_time_s"))
        if epoch is None or val_acc is None:
            continue
        run.epochs.append({
            "epoch": epoch,
            "val_acc": val_acc,
            "train_acc": train_acc if train_acc is not None else float("nan"),
            "max_memory_mb": memory if memory is not None else float("nan"),
            "compute_time_s": fwd_sum + bwd_sum,
        })
        if wall_time is not None:
            run.epochs[-1]["wall_time_s"] = wall_time

    if selected_args:
        run.precision = selected_args.get("precision_str", run.precision)
    if run.epochs and int(max(row["epoch"] for row in run.epochs)) >= expected_epoch:
        run.status = "complete"
    elif run.epochs:
        run.status = "incomplete"
    else:
        run.status = "failed"


def load_runs(manifest_path: Path, expected_epoch: int) -> List[Run]:
    runs: List[Run] = []
    manifest_dir = manifest_path.resolve().parent
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            equation = row.get("equation", "").strip()
            if equation not in {"ode", "fde"}:
                continue
            result_path = Path(row.get("result_root", "").strip())
            if not result_path.is_absolute():
                result_path = manifest_dir / result_path
            log_path = Path(row.get("log_path", "").strip())
            if not log_path.is_absolute():
                log_path = manifest_dir / log_path
            beta_text = row.get("beta", "").strip()
            beta = float(beta_text) if beta_text else None
            run = Run(
                equation=equation,
                beta=beta,
                method=row.get("method", "").strip(),
                precision=row.get("precision", "").strip(),
                job_id=row.get("job_id", "").strip(),
                result_path=log_path if equation == "fde" else result_path,
                source="fde-log" if equation == "fde" else "ode-csv",
            )
            if equation == "fde":
                parse_fde_log(run, expected_epoch)
            else:
                parse_ode_result(run, expected_epoch)
            runs.append(run)
    return runs


def time_to_target(run: Run, target: float) -> Optional[float]:
    elapsed = 0.0
    for row in run.epochs:
        if row["val_acc"] >= target:
            wall_time = row.get("wall_time_s")
            if wall_time is not None and math.isfinite(wall_time):
                return wall_time
            if run.equation == "ode":
                return row.get("compute_time_s")
            return elapsed + row.get("time", 0.0)
        if run.equation != "ode":
            elapsed += row.get("time", 0.0)
    return None


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "F"
    return f"{value:.{digits}f}"


def write_run_csv(path: Path, runs: List[Run], targets: List[float]) -> None:
    fields = [
        "equation", "beta", "method", "precision", "job_id", "status", "source", "result_path",
        "final_train_acc", "final_val_acc", "final_val_error", "best_val_acc", "best_val_error",
        "best_epoch", "generalization_gap", "train_memory_mb", "train_time_s", "infer_time_s", "infer_memory_mb",
    ] + [f"time_to_acc_{target:g}" for target in targets]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "equation": run.equation,
                "beta": "" if run.beta is None else f"{run.beta:g}",
                "method": run.method,
                "precision": run.precision,
                "job_id": run.job_id,
                "status": run.status,
                "source": run.source,
                "result_path": str(run.result_path),
                "final_train_acc": fmt(run.final_train_acc, 6),
                "final_val_acc": fmt(run.final_val_acc, 6),
                "final_val_error": fmt(run.final_val_error, 6),
                "best_val_acc": fmt(run.best_val_acc, 6),
                "best_val_error": fmt(run.best_val_error, 6),
                "best_epoch": fmt(float(run.best_epoch) if run.best_epoch is not None else None, 0),
                "generalization_gap": fmt(run.generalization_gap, 6),
                "train_memory_mb": fmt(run.train_memory, 6),
                "train_time_s": fmt(run.train_time, 6),
                "infer_time_s": fmt(run.infer_time, 6),
                "infer_memory_mb": fmt(run.infer_memory, 6),
            }
            for target in targets:
                row[f"time_to_acc_{target:g}"] = fmt(time_to_target(run, target), 6)
            writer.writerow(row)


def table_for_metric(runs: List[Run], metric_name: str, title: str, digits: int = 4) -> str:
    fde_betas = sorted({run.beta for run in runs if run.equation == "fde" and run.beta is not None})
    columns = ["ODE"] + fde_betas
    lines = [f"## {title}", "", "| configuration | " + " | ".join(str(value) if value == "ODE" else f"beta={value:g}" for value in columns) + " |", "|---|" + "---|" * len(columns)]
    configs = [
        ("ODE-rampde-float32", lambda run: run.equation == "ode" and run.precision == "float32"),
        ("ODE-rampde-bfloat16", lambda run: run.equation == "ode" and run.precision == "bfloat16"),
        ("FDE-adjoint-fp32", lambda run: run.equation == "fde" and run.method == "adjoint"),
        ("FDE-adjoint-bfloat16", lambda run: run.equation == "fde" and run.method == "adjoint-mixed-bfloat"),
    ]
    for label, predicate in configs:
        matching = [run for run in runs if predicate(run)]
        ode_value = next((getattr(run, metric_name) for run in matching if run.equation == "ode"), None)
        values = [ode_value]
        for beta in fde_betas:
            values.append(next((getattr(run, metric_name) for run in matching if run.beta == beta), None))
        lines.append("| " + label + " | " + " | ".join(fmt(value, digits) for value in values) + " |")
    return "\n".join(lines) + "\n"


def make_plots(output_dir: Path, prefix: str, runs: List[Run], targets: List[float]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; skipping plots.")
        return []

    plots: List[Path] = []

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        plots.append(path)

    groups = [
        ("float32_adjoint", "FDE adjoint FP32 vs. integer ODE FP32", "adjoint", "float32"),
        ("bfloat16_adjoint", "FDE adjoint BF16 vs. integer ODE BF16", "adjoint-mixed-bfloat", "bfloat16"),
    ]
    for filename, title, fde_method, precision in groups:
        plt.figure(figsize=(9, 5))
        ode_runs = [run for run in runs if run.equation == "ode" and run.precision == precision]
        for run in ode_runs:
            if run.epochs:
                plt.plot([row["epoch"] for row in run.epochs], [row["val_acc"] for row in run.epochs], linewidth=2.5, label="integer ODE")
        fde_runs = [run for run in runs if run.equation == "fde" and run.method == fde_method]
        for run in sorted(fde_runs, key=lambda item: item.beta or -1):
            if run.epochs:
                plt.plot([row["epoch"] for row in run.epochs], [row["val_acc"] for row in run.epochs], label=f"FDE β={run.beta:g}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(fontsize="small", ncol=2)
        save(output_dir / f"{prefix}_{filename}_learning_curve.png")

    for metric_name, title, filename, ylabel in [
        ("best_val_acc", "Best Validation Accuracy", "best_validation_accuracy", "Best validation accuracy"),
        ("final_val_error", "Final Validation Error", "final_validation_error", "Final validation error"),
        ("train_memory", "Peak Training Memory", "training_memory", "Peak memory (MB)"),
        ("train_time", "Recorded Training Time", "training_time", "Time (s)"),
    ]:
        plt.figure(figsize=(9, 5))
        for label, predicate in [
            ("FDE adjoint FP32", lambda run: run.equation == "fde" and run.method == "adjoint"),
            ("FDE adjoint BF16", lambda run: run.equation == "fde" and run.method == "adjoint-mixed-bfloat"),
        ]:
            values = [(run.beta, getattr(run, metric_name)) for run in runs if predicate(run) and run.beta is not None]
            values = [(beta, value) for beta, value in values if value is not None]
            if values:
                values.sort()
                plt.plot([beta for beta, _ in values], [value for _, value in values], marker="o", label=label)
        for precision, label in [("float32", "ODE FP32"), ("bfloat16", "ODE BF16")]:
            ode_values = [
                getattr(run, metric_name)
                for run in runs
                if run.equation == "ode"
                and run.precision == precision
                and getattr(run, metric_name) is not None
            ]
            for value in ode_values:
                plt.axhline(value, linestyle="--", alpha=0.7, label=label)
        plt.xlabel("Beta")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(fontsize="small", ncol=2)
        save(output_dir / f"{prefix}_{filename}_by_beta.png")

    for target in targets:
        plt.figure(figsize=(9, 5))
        for label, predicate in [
            ("FDE adjoint FP32", lambda run: run.equation == "fde" and run.method == "adjoint"),
            ("FDE adjoint BF16", lambda run: run.equation == "fde" and run.method == "adjoint-mixed-bfloat"),
        ]:
            values = [(run.beta, time_to_target(run, target)) for run in runs if predicate(run) and run.beta is not None]
            values = [(beta, value) for beta, value in values if value is not None]
            if values:
                values.sort()
                plt.plot([beta for beta, _ in values], [value for _, value in values], marker="o", label=label)
        plt.xlabel("Beta")
        plt.ylabel("Recorded time to target (s)")
        plt.title(f"Time to Validation Accuracy ≥ {target:g}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize="small")
        save(output_dir / f"{prefix}_time_to_acc_{target:g}.png")

    return plots


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.targets)
    runs = load_runs(manifest, args.expected_epoch)
    if not runs:
        raise SystemExit("No ODE or FDE runs found in manifest")

    report = ["# STL10 Integer ODE vs. Fractional FDE Comparison", "", f"Manifest: `{manifest}`", ""]
    report.append("## Run status\n\n| equation | beta | method | precision | status | path |\n|---|---|---|---|---|---|")
    for run in runs:
        beta = "integer" if run.beta is None else f"{run.beta:g}"
        report.append(f"| {run.equation} | {beta} | {run.method} | {run.precision} | {run.status} | `{run.result_path}` |")
    report.append("")
    report.append(table_for_metric(runs, "best_val_acc", "Best Validation Accuracy"))
    report.append(table_for_metric(runs, "final_val_error", "Final Validation Error"))
    report.append(table_for_metric(runs, "best_epoch", "Epoch of Best Validation Accuracy", digits=0))
    report.append(table_for_metric(runs, "generalization_gap", "Final Generalization Gap"))
    report.append(table_for_metric(runs, "train_memory", "Peak Training Memory (MB)", digits=2))
    report.append(table_for_metric(runs, "train_time", "Recorded Training Time (s)", digits=2))
    for target in targets:
        report.append(f"## Time to Validation Accuracy ≥ {target:g}\n")
        beta_values = sorted({run.beta for run in runs if run.beta is not None})
        report.append("| configuration | " + " | ".join(f"beta={beta:g}" for beta in beta_values) + " |\n|---|" + "---|" * len(beta_values))
        for label, predicate in [
            ("ODE-rampde-float32", lambda run: run.equation == "ode" and run.precision == "float32"),
            ("ODE-rampde-bfloat16", lambda run: run.equation == "ode" and run.precision == "bfloat16"),
            ("FDE-adjoint-fp32", lambda run: run.equation == "fde" and run.method == "adjoint"),
            ("FDE-adjoint-bfloat16", lambda run: run.equation == "fde" and run.method == "adjoint-mixed-bfloat"),
        ]:
            matching = [run for run in runs if predicate(run)]
            ode_run = next((run for run in matching if run.equation == "ode"), None)
            values = []
            for beta in beta_values:
                beta_run = next((run for run in matching if run.beta == beta), ode_run)
                values.append(time_to_target(beta_run, target) if beta_run is not None else None)
            report.append("| " + label + " | " + " | ".join(fmt(value, 2) for value in values) + " |")
        report.append("")

    plot_paths = make_plots(output_dir, args.prefix, runs, targets)
    report.extend(["## Generated plots", ""])
    report.extend(f"- [{path.name}]({path.name})" for path in plot_paths)
    report.extend([
        "", "## Interpretation notes", "",
        "- The primary equation comparison is integer ODE rampde versus fractional FDE at matched settings.",
        "- ODE rampde uses its custom reverse-mode backward implementation; it is not the same API as torchdiffeq.odeint_adjoint.",
        "- BF16 runs use no gradient or dynamic loss scaling.",
        "- Training time and time-to-target use the common wall-clock timer recorded by both runners; older result files without wall-clock fields fall back to their legacy timing fields.",
        "- One seed is exploratory evidence. Use multiple seeds for paper-level means and uncertainty.",
        "- These runs report validation accuracy on the held-out STL10 training split, not official STL10 test accuracy.",
    ])

    report_path = output_dir / f"{args.prefix}.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    write_run_csv(output_dir / f"{args.prefix}.csv", runs, targets)
    print(f"Wrote: {report_path}")
    print(f"Wrote: {output_dir / f'{args.prefix}.csv'}")
    for path in plot_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
