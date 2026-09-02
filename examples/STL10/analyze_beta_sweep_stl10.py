#!/usr/bin/env python3
"""Create a comprehensive STL10 beta-sweep report from training logs."""

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MODE_ORDER = [
    ("direct", "dir"),
    ("adjoint", "adj"),
    ("adjoint-mixed", "adj_fl16"),
    ("adjoint-mixed-bfloat", "adj_bfl16"),
]

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Time\s+(?P<time>[-+0-9.eE]+)s\s*\|\s*"
    r"Peak Mem\s+(?P<mem>[-+0-9.eE]+) MB\s*\|.*?"
    r"Train Acc\s+(?P<train_acc>[-+0-9.eE]+)\s*\|\s*"
    r"Val Acc\s+(?P<val_acc>[-+0-9.eE]+)\s*\|\s*"
    r"Best\s+(?P<best_acc>[-+0-9.eE]+)"
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
    beta: float
    mode: str
    log_path: Path
    epochs: List[Dict[str, float]] = field(default_factory=list)
    final: Dict[str, float] = field(default_factory=dict)
    status: str = "missing"

    @property
    def completed(self) -> bool:
        return bool(self.final) and self.status == "complete"

    @property
    def final_val_acc(self) -> Optional[float]:
        return self.epochs[-1]["val_acc"] if self.epochs else None

    @property
    def best_epoch(self) -> Optional[int]:
        if not self.epochs:
            return None
        return int(max(self.epochs, key=lambda row: row["val_acc"])["epoch"])

    @property
    def best_val_acc(self) -> Optional[float]:
        if not self.epochs:
            return None
        return max(row["val_acc"] for row in self.epochs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Beta sweep manifest CSV")
    parser.add_argument(
        "--expected-epoch",
        type=int,
        default=None,
        help="Expected final epoch; omit to treat any Final metrics line as complete",
    )
    parser.add_argument("--output-dir", default=None, help="Report output directory")
    parser.add_argument("--prefix", default="stl10_beta_sweep_analysis")
    parser.add_argument(
        "--targets",
        default="0.40,0.50",
        help="Validation-accuracy targets for time-to-quality tables",
    )
    return parser.parse_args()


def parse_targets(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def resolve_log_path(manifest_dir: Path, row: Dict[str, str]) -> Path:
    save_root = Path(row.get("save_root", "").strip())
    if not save_root.is_absolute():
        save_root = manifest_dir / save_root
    mode_log = save_root / row["mode"].strip() / "training.log"
    if mode_log.exists():
        return mode_log

    recorded = Path(row.get("run_log", "").strip())
    if not recorded.is_absolute():
        recorded = manifest_dir / recorded
    return recorded


def parse_log(path: Path, expected_epoch: Optional[int]) -> Run:
    run = Run(beta=0.0, mode="", log_path=path)
    if not path.exists():
        return run

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return run

    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            run.epochs.append({key: float(value) for key, value in epoch_match.groupdict().items()})
        final_match = FINAL_RE.search(line)
        if final_match:
            run.final = {key: float(value) for key, value in final_match.groupdict().items()}

    if run.final and (expected_epoch is None or (run.epochs and run.epochs[-1]["epoch"] >= expected_epoch)):
        run.status = "complete"
    elif run.epochs:
        run.status = "incomplete"
    else:
        run.status = "failed"
    return run


def load_runs(manifest_path: Path, expected_epoch: Optional[int]) -> List[Run]:
    runs: List[Run] = []
    manifest_dir = manifest_path.resolve().parent
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            mode = row.get("mode", "").strip()
            if mode not in {name for name, _ in MODE_ORDER}:
                continue
            log_path = resolve_log_path(manifest_dir, row)
            run = parse_log(log_path, expected_epoch)
            run.beta = float(row["Beta"])
            run.mode = mode
            runs.append(run)
    return sorted(runs, key=lambda run: (run.beta, run.mode))


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "F"
    return f"{value:.{digits}f}"


def metric(run: Run, name: str) -> Optional[float]:
    if name == "final_val_acc":
        return run.final_val_acc
    if name == "best_val_acc":
        return run.best_val_acc
    if name == "final_val_error":
        return 1.0 - run.final_val_acc if run.final_val_acc is not None else None
    if name == "best_val_error":
        return 1.0 - run.best_val_acc if run.best_val_acc is not None else None
    if name == "generalization_gap":
        if run.final_val_acc is None or not run.epochs:
            return None
        return run.epochs[-1]["train_acc"] - run.final_val_acc
    if name == "best_epoch":
        return float(run.best_epoch) if run.best_epoch is not None else None
    return run.final.get(name)


def markdown_table(runs: List[Run], name: str, title: str, digits: int = 4) -> str:
    betas = sorted({run.beta for run in runs})
    lines = [f"## {title}", "", "| method | " + " | ".join(f"{beta:g}" for beta in betas) + " |", "|---|" + "---|" * len(betas)]
    for mode, label in MODE_ORDER:
        values = {run.beta: metric(run, name) for run in runs if run.mode == mode}
        lines.append("| " + label + " | " + " | ".join(fmt(values.get(beta), digits) for beta in betas) + " |")
    return "\n".join(lines) + "\n"


def time_to_target(run: Run, target: float) -> Optional[float]:
    elapsed = 0.0
    for row in run.epochs:
        elapsed += row["time"]
        if row["val_acc"] >= target:
            return elapsed
    return None


def write_csv(path: Path, runs: List[Run], targets: List[float]) -> None:
    fields = [
        "beta", "mode", "status", "log_path", "final_train_acc", "final_val_acc",
        "final_val_error", "best_val_acc", "best_val_error", "best_epoch",
        "generalization_gap", "train_mem", "train_time", "infer_time", "infer_mem",
    ] + [f"time_to_acc_{target:g}" for target in targets]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {"beta": f"{run.beta:g}", "mode": run.mode, "status": run.status, "log_path": str(run.log_path)}
            row.update({field: fmt(metric(run, field), 6) for field in fields if field not in row and not field.startswith("time_to_acc_")})
            for target in targets:
                row[f"time_to_acc_{target:g}"] = fmt(time_to_target(run, target), 2)
            writer.writerow(row)


def make_plots(output_dir: Path, prefix: str, runs: List[Run]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; skipping plots.")
        return []

    plots: List[Path] = []
    complete_runs = [run for run in runs if run.epochs]

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        plots.append(path)

    for title, field, filename, ylabel in [
        ("STL10 Validation Accuracy by Beta", "val_acc", "validation_accuracy", "Validation accuracy"),
        ("STL10 Training Accuracy by Beta", "train_acc", "training_accuracy", "Training accuracy"),
    ]:
        fig, axes = plt.subplots(1, len(MODE_ORDER), figsize=(18, 4), sharey=True)
        for axis, (mode, label) in zip(axes, MODE_ORDER):
            mode_runs = [run for run in complete_runs if run.mode == mode]
            for run in mode_runs:
                axis.plot([row["epoch"] for row in run.epochs], [row[field] for row in run.epochs], label=f"β={run.beta:g}")
            axis.set_title(label)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize="small")
        fig.suptitle(title)
        save(output_dir / f"{prefix}_{filename}.png")

    for title, name, filename, ylabel in [
        ("Final Validation Error by Beta", "final_val_error", "final_validation_error", "Final validation error"),
        ("Best Validation Error by Beta", "best_val_error", "best_validation_error", "Best validation error"),
        ("Training Memory by Beta", "train_mem", "training_memory", "Peak memory (MB)"),
        ("Training Time by Beta", "train_time", "training_time", "Training time (s)"),
    ]:
        plt.figure(figsize=(8, 5))
        for mode, label in MODE_ORDER:
            mode_runs = [run for run in runs if run.mode == mode]
            values = [(run.beta, metric(run, name)) for run in mode_runs]
            values = [(beta, value) for beta, value in values if value is not None]
            if values:
                plt.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=label)
        plt.xlabel("Beta")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        save(output_dir / f"{prefix}_{filename}.png")

    plt.figure(figsize=(8, 5))
    for run in runs:
        if run.final.get("train_mem") is not None and metric(run, "best_val_error") is not None:
            plt.scatter(run.final["train_mem"], metric(run, "best_val_error"), label=f"{run.mode}, β={run.beta:g}")
    plt.xlabel("Peak training memory (MB)")
    plt.ylabel("Best validation error")
    plt.title("STL10 Memory/Accuracy Tradeoff")
    plt.grid(alpha=0.25)
    plt.legend(fontsize="small", ncol=2)
    save(output_dir / f"{prefix}_memory_accuracy_tradeoff.png")
    return plots


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.targets)
    runs = load_runs(manifest, args.expected_epoch)
    if not runs:
        raise SystemExit("No recognized runs found in manifest")

    report = [f"# STL10 Beta Sweep Analysis", "", f"Manifest: `{manifest}`", ""]
    report.append("## Run status\n\n| beta | mode | status | log |\n|---|---|---|---|")
    for run in runs:
        report.append(f"| {run.beta:g} | {run.mode} | {run.status} | `{run.log_path}` |")
    report.append("")
    report.append(markdown_table(runs, "final_val_error", "Final Validation Error by Beta"))
    report.append(markdown_table(runs, "best_val_error", "Best Validation Error by Beta"))
    report.append(markdown_table(runs, "final_val_acc", "Final Validation Accuracy by Beta"))
    report.append(markdown_table(runs, "best_epoch", "Epoch of Best Validation Accuracy", digits=0))
    report.append(markdown_table(runs, "generalization_gap", "Final Generalization Gap"))
    report.append(markdown_table(runs, "train_mem", "Peak Training Memory (MB)", digits=2))
    report.append(markdown_table(runs, "train_time", "Training Time (s)", digits=2))
    report.append(markdown_table(runs, "infer_time", "Inference Time (s)", digits=2))
    report.append(markdown_table(runs, "infer_mem", "Inference Peak Memory (MB)", digits=2))
    for target in targets:
        rows = []
        for mode, label in MODE_ORDER:
            values = {run.beta: time_to_target(run, target) for run in runs if run.mode == mode}
            rows.append((label, values))
        betas = sorted({run.beta for run in runs})
        report.extend([
            f"## Time to Validation Accuracy ≥ {target:g}", "",
            "| method | " + " | ".join(f"{beta:g}" for beta in betas) + " |",
            "|---|" + "---|" * len(betas),
        ])
        report.extend("| " + label + " | " + " | ".join(fmt(values.get(beta), 2) for beta in betas) + " |" for label, values in rows)
        report.append("")

    plot_paths = make_plots(output_dir, args.prefix, runs)
    report.extend(["## Generated plots", ""])
    report.extend(f"- [{path.name}]({path.name})" for path in plot_paths)
    report.extend([
        "", "## Interpretation notes", "",
        "- Lower validation error is better; higher validation accuracy is better.",
        "- Compare beta values within the same execution mode before comparing modes.",
        "- A single run per beta does not separate beta effects from random-seed variation.",
        "- `incomplete`, `failed`, and `F` entries should not be treated as numerical results.",
    ])

    report_path = output_dir / f"{args.prefix}.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    write_csv(output_dir / f"{args.prefix}.csv", runs, targets)
    print(f"Wrote: {report_path}")
    print(f"Wrote: {output_dir / f'{args.prefix}.csv'}")
    for path in plot_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
