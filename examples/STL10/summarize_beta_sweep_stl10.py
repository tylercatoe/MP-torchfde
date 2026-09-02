#!/usr/bin/env python3
"""Summarize STL10 beta-sweep training logs into memory/time tables."""

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, Union


MODE_ORDER: List[Tuple[str, str]] = [
    ("direct", "dir"),
    ("adjoint", "adj"),
    ("adjoint-mixed", "adj_fl16"),
    ("adjoint-mixed-bfloat", "adj_bfl16"),
]

RATIO_ROWS: List[Tuple[str, str, str]] = [
    ("dir/adj", "direct", "adjoint"),
    ("dir/adj_fl16", "direct", "adjoint-mixed"),
    ("dir/adj_bfl16", "direct", "adjoint-mixed-bfloat"),
    ("adj/adj_fl16", "adjoint", "adjoint-mixed"),
    ("adj/adj_bfl16", "adjoint", "adjoint-mixed-bfloat"),
]

FINAL_METRICS_RE = re.compile(
    r"Final metrics\s*\|"
    r"\s*Final Val Error (?P<final_val_error>[-+0-9.eE]+)\s*\|"
    r"\s*Best Val Error (?P<best_val_error>[-+0-9.eE]+)\s*\|"
    r"\s*Max Train Mem (?P<train_mem>[-+0-9.eE]+) MB\s*\|"
    r"\s*Train Time (?P<train_time>[-+0-9.eE]+)s\s*\|"
)

Metric = Union[float, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize STL10 beta-sweep final training metrics."
    )
    parser.add_argument("--manifest", required=True, help="Sweep manifest CSV")
    # Accepted for compatibility with submit_mp_fde_stl10_beta_sweep.sh.
    parser.add_argument("--epoch", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument(
        "--output_prefix", default="stl10_beta_sweep", help="Output filename prefix"
    )
    return parser.parse_args()


def parse_log(log_path: str) -> Dict[str, float]:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            for line in reversed(log_file.readlines()):
                match = FINAL_METRICS_RE.search(line)
                if match:
                    return {
                        "train_mem": float(match.group("train_mem")),
                        "train_time": float(match.group("train_time")),
                    }
    except OSError:
        pass
    return {}


def parse_manifest(manifest_path: str) -> Tuple[List[float], Dict[str, Dict[float, Dict[str, float]]]]:
    betas: set[float] = set()
    metrics: Dict[str, Dict[float, Dict[str, float]]] = {
        mode: {} for mode, _ in MODE_ORDER
    }
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    with open(manifest_path, "r", encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            mode = row.get("mode", "").strip()
            if mode not in metrics:
                continue
            beta = float(row["Beta"])
            betas.add(beta)

            save_root = row.get("save_root", "").strip()
            if not os.path.isabs(save_root):
                save_root = os.path.join(manifest_dir, save_root)
            log_path = os.path.join(save_root, mode, "training.log")

            # Fall back to the manifest path if a future submit script records it
            # correctly, while preferring the actual STL10 training-log location.
            if not os.path.exists(log_path):
                manifest_log = row.get("run_log", "").strip()
                if manifest_log:
                    log_path = manifest_log
                    if not os.path.isabs(log_path):
                        log_path = os.path.join(manifest_dir, log_path)

            parsed = parse_log(log_path)
            if parsed:
                metrics[mode][beta] = parsed

    return sorted(betas), metrics


def empty_table(betas: List[float]) -> Dict[str, Dict[float, Metric]]:
    return {mode: {beta: "F" for beta in betas} for mode, _ in MODE_ORDER}


def ratio_value(numerator: Metric, denominator: Metric) -> Metric:
    if numerator == "F" or denominator == "F" or denominator <= 0:  # type: ignore[operator]
        return "F"
    return float(numerator) / float(denominator)


def build_rows(table: Dict[str, Dict[float, Metric]], betas: List[float]) -> List[Tuple[str, Dict[float, Metric]]]:
    rows = [(label, {beta: table[mode][beta] for beta in betas}) for mode, label in MODE_ORDER]
    for label, numerator_mode, denominator_mode in RATIO_ROWS:
        rows.append(
            (
                label,
                {
                    beta: ratio_value(
                        table[numerator_mode][beta], table[denominator_mode][beta]
                    )
                    for beta in betas
                },
            )
        )
    return rows


def format_beta(beta: float) -> str:
    return f"{beta:g}"


def format_value(value: Metric, is_ratio: bool) -> str:
    if value == "F":
        return "F"
    return f"{float(value):.3f}" if is_ratio else f"{float(value):.2f}"


def write_tables(path: str, title: str, rows: List[Tuple[str, Dict[float, Metric]]], betas: List[float]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        output.write(f"# {title}\n\n")
        output.write("| method | " + " | ".join(format_beta(beta) for beta in betas) + " |\n")
        output.write("|---|" + "---|" * len(betas) + "\n")
        for label, values in rows:
            output.write(
                "| " + label + " | "
                + " | ".join(format_value(values[beta], "/" in label) for beta in betas)
                + " |\n"
            )


def write_csv(path: str, rows: List[Tuple[str, Dict[float, Metric]]], betas: List[float]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["method", *[format_beta(beta) for beta in betas]])
        for label, values in rows:
            writer.writerow([label, *[format_value(values[beta], "/" in label) for beta in betas]])


def main() -> None:
    args = parse_args()
    betas, parsed = parse_manifest(args.manifest)
    if not betas:
        raise SystemExit("No beta values found in manifest")

    memory = empty_table(betas)
    times = empty_table(betas)
    for mode in parsed:
        for beta, values in parsed[mode].items():
            memory[mode][beta] = values["train_mem"]
            times[mode][beta] = values["train_time"]

    memory_rows = build_rows(memory, betas)
    time_rows = build_rows(times, betas)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(output_dir, exist_ok=True)

    outputs = {
        "memory_md": os.path.join(output_dir, f"{args.output_prefix}_memory.md"),
        "time_md": os.path.join(output_dir, f"{args.output_prefix}_time.md"),
        "memory_csv": os.path.join(output_dir, f"{args.output_prefix}_memory.csv"),
        "time_csv": os.path.join(output_dir, f"{args.output_prefix}_time.csv"),
    }
    write_tables(outputs["memory_md"], "STL10 Beta Sweep - Peak Training Memory (MB)", memory_rows, betas)
    write_tables(outputs["time_md"], "STL10 Beta Sweep - Training Time (s)", time_rows, betas)
    write_csv(outputs["memory_csv"], memory_rows, betas)
    write_csv(outputs["time_csv"], time_rows, betas)
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
