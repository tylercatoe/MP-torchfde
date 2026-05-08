#!/usr/bin/env python3
"""
Driver for mixed-precision fractional sweeps.

This script orchestrates multiple runs of:
  examples/mixed_precision_experiment_matrix.py

Sweep groups:
  - scale
  - beta
  - discretization
  - seed

Outputs:
  - Per-run folder with stdout/stderr and raw JSON/CSV from each matrix run
  - consolidated_results.csv/json (one row per config per run)
  - best_by_run.csv/json (recommended config per run)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
MATRIX_SCRIPT = THIS_DIR / "mixed_precision_experiment_matrix.py"


@dataclass
class RunSpec:
    section: str
    name: str
    overrides: Dict[str, Any]

    @property
    def run_id(self) -> str:
        return f"{self.section}__{self.name}"


def _float_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _parse_discretization_case(case: str) -> Tuple[float, float]:
    # Expected format: "<t_final>:<step_size>"
    if ":" not in case:
        raise ValueError(
            f"Invalid discretization case '{case}'. Expected format '<t_final>:<step_size>', e.g. '2.0:0.05'."
        )
    t_str, step_str = case.split(":", 1)
    return float(t_str), float(step_str)


def _build_base_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "seed": args.base_seed,
        "train-samples": args.base_train_samples,
        "val-samples": args.base_val_samples,
        "batch-size": args.base_batch_size,
        "epochs": args.base_epochs,
        "dim": args.base_dim,
        "hidden": args.base_hidden,
        "out-dim": args.base_out_dim,
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "beta": args.base_beta,
        "t-final": args.base_t_final,
        "step-size": args.base_step_size,
        "adjoint-method": args.adjoint_method,
        "teacher-method": args.teacher_method,
        "targets-batch-size": args.targets_batch_size,
    }


def build_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    requested_sections = set(args.sections)
    base = _build_base_overrides(args)
    specs: List[RunSpec] = []

    if "scale" in requested_sections:
        scale_overrides = dict(base)
        scale_overrides.update(
            {
                "epochs": args.scale_epochs,
                "train-samples": args.scale_train_samples,
                "val-samples": args.scale_val_samples,
                "batch-size": args.scale_batch_size,
                "dim": args.scale_dim,
                "hidden": args.scale_hidden,
                "out-dim": args.scale_out_dim,
            }
        )
        specs.append(RunSpec(section="scale", name="large_workload", overrides=scale_overrides))

    if "beta" in requested_sections:
        for beta in args.beta_values:
            over = dict(base)
            over.update({"beta": beta, "epochs": args.beta_epochs})
            specs.append(RunSpec(section="beta", name=f"beta_{_float_label(beta)}", overrides=over))

    if "discretization" in requested_sections:
        for case in args.discretization_cases:
            t_final, step_size = _parse_discretization_case(case)
            over = dict(base)
            over.update(
                {
                    "t-final": t_final,
                    "step-size": step_size,
                    "epochs": args.discretization_epochs,
                }
            )
            specs.append(
                RunSpec(
                    section="discretization",
                    name=f"t_{_float_label(t_final)}__h_{_float_label(step_size)}",
                    overrides=over,
                )
            )

    if "seed" in requested_sections:
        for seed in args.seed_values:
            over = dict(base)
            over.update({"seed": seed, "epochs": args.seed_epochs})
            specs.append(RunSpec(section="seed", name=f"seed_{seed}", overrides=over))

    return specs


def _to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        args.extend([f"--{key}", str(value)])
    return args


def _run_once(
    spec: RunSpec,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    run_dir = output_root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_out = run_dir / "matrix_results.json"
    csv_out = run_dir / "matrix_results.csv"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    meta_json = run_dir / "run_meta.json"

    cmd = [
        sys.executable,
        str(MATRIX_SCRIPT),
        "--device",
        args.device,
        "--json-out",
        str(json_out),
        "--csv-out",
        str(csv_out),
    ] + _to_cli_args(spec.overrides)

    run_meta: Dict[str, Any] = {
        "run_id": spec.run_id,
        "section": spec.section,
        "name": spec.name,
        "overrides": spec.overrides,
        "command": cmd,
        "returncode": None,
        "wall_time_s": None,
        "json_out": str(json_out),
        "csv_out": str(csv_out),
    }

    print(f"\n=== Running {spec.run_id} ===")
    print(" ".join(cmd))

    if args.dry_run:
        run_meta["returncode"] = 0
        run_meta["wall_time_s"] = 0.0
        with open(meta_json, "w") as f:
            json.dump(run_meta, f, indent=2)
        return run_meta

    t0 = time.perf_counter()
    completed = subprocess.run(cmd, text=True, capture_output=True)
    wall = time.perf_counter() - t0

    with open(stdout_log, "w") as f:
        f.write(completed.stdout)
    with open(stderr_log, "w") as f:
        f.write(completed.stderr)

    run_meta["returncode"] = int(completed.returncode)
    run_meta["wall_time_s"] = float(wall)

    with open(meta_json, "w") as f:
        json.dump(run_meta, f, indent=2)

    status = "ok" if completed.returncode == 0 else "error"
    print(f"[{status}] {spec.run_id} in {wall:.2f}s")
    if completed.returncode != 0:
        print(f"See stderr: {stderr_log}")

    return run_meta


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _load_matrix_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}, got {type(rows)}")
    return rows


def _pick_best_row(run_rows: List[Dict[str, Any]], val_tol_pct: float) -> Dict[str, Any]:
    stable_rows = [
        r
        for r in run_rows
        if r.get("status") == "ok"
        and int(r.get("nan_inf_events", 1)) == 0
        and _is_finite_number(r.get("final_val_loss"))
        and _is_finite_number(r.get("mean_epoch_s"))
    ]

    baseline_rows = [r for r in stable_rows if r.get("config") == "fp32_unscaled"]
    baseline = baseline_rows[0]["final_val_loss"] if baseline_rows else None

    candidates = stable_rows
    if baseline is not None and _is_finite_number(baseline):
        tol = float(baseline) * (1.0 + val_tol_pct / 100.0)
        candidates = [r for r in stable_rows if float(r["final_val_loss"]) <= tol]
        if not candidates:
            candidates = stable_rows

    if not candidates:
        return {}

    best = min(candidates, key=lambda r: float(r["mean_epoch_s"]))
    return dict(best)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def _print_recommendations(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("No recommendation rows available.")
        return
    headers = [
        "run_id",
        "best_config",
        "best_val_loss",
        "best_mean_epoch_s",
        "best_train_samples_per_s",
        "best_peak_mem_mib",
        "baseline_fp32_val_loss",
        "best_vs_fp32_val_pct",
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if not math.isfinite(v):
                return "nan"
            return f"{v:.6f}"
        return str(v)

    widths = {h: max(len(h), *(len(fmt(r.get(h, ""))) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print("\nRecommended Config By Run")
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(fmt(row.get(h, "")).ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-sweep mixed-precision experiments and aggregate results.")
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--sections",
        nargs="+",
        default=["scale", "beta", "discretization", "seed"],
        choices=["scale", "beta", "discretization", "seed"],
        help="Which sweep groups to run.",
    )
    p.add_argument("--output-dir", type=str, default="examples/sweep_outputs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--val-tol-pct", type=float, default=0.1, help="Allowed val-loss increase vs fp32 for best pick.")

    # Shared base parameters.
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--base-train-samples", type=int, default=4096)
    p.add_argument("--base-val-samples", type=int, default=1024)
    p.add_argument("--base-batch-size", type=int, default=256)
    p.add_argument("--base-epochs", type=int, default=6)
    p.add_argument("--base-dim", type=int, default=32)
    p.add_argument("--base-hidden", type=int, default=64)
    p.add_argument("--base-out-dim", type=int, default=16)
    p.add_argument("--base-beta", type=float, default=0.8)
    p.add_argument("--base-t-final", type=float, default=1.0)
    p.add_argument("--base-step-size", type=float, default=0.05)
    p.add_argument("--adjoint-method", type=str, default="predictor-f")
    p.add_argument("--teacher-method", type=str, default="predictor")
    p.add_argument("--targets-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Scale sweep parameters.
    p.add_argument("--scale-epochs", type=int, default=10)
    p.add_argument("--scale-train-samples", type=int, default=32768)
    p.add_argument("--scale-val-samples", type=int, default=8192)
    p.add_argument("--scale-batch-size", type=int, default=1024)
    p.add_argument("--scale-dim", type=int, default=128)
    p.add_argument("--scale-hidden", type=int, default=256)
    p.add_argument("--scale-out-dim", type=int, default=64)

    # Beta sweep parameters.
    p.add_argument("--beta-values", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--beta-epochs", type=int, default=6)

    # Discretization sweep parameters.
    p.add_argument(
        "--discretization-cases",
        nargs="+",
        default=["1.0:0.1", "2.0:0.05", "2.0:0.025"],
        help="Cases of '<t_final>:<step_size>'.",
    )
    p.add_argument("--discretization-epochs", type=int, default=6)

    # Seed sweep parameters.
    p.add_argument("--seed-values", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--seed-epochs", type=int, default=6)

    return p.parse_args()


def main() -> int:
    if not MATRIX_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find matrix script: {MATRIX_SCRIPT}")

    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    specs = build_run_specs(args)
    if not specs:
        print("No run specs were generated.")
        return 1

    run_meta_list: List[Dict[str, Any]] = []
    consolidated_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for spec in specs:
        meta = _run_once(spec, args, output_root)
        run_meta_list.append(meta)

        if args.dry_run:
            continue

        if meta.get("returncode") != 0:
            continue
        matrix_json = Path(meta["json_out"])
        if not matrix_json.exists():
            print(f"[warn] missing matrix json for {spec.run_id}: {matrix_json}")
            continue

        try:
            rows = _load_matrix_rows(matrix_json)
        except Exception as exc:
            print(f"[warn] failed loading rows for {spec.run_id}: {type(exc).__name__}: {exc}")
            continue

        for row in rows:
            merged = {
                "run_id": spec.run_id,
                "section": spec.section,
                "name": spec.name,
                "wall_time_s": meta.get("wall_time_s", float("nan")),
            }
            merged.update(spec.overrides)
            merged.update(row)
            consolidated_rows.append(merged)

        best = _pick_best_row(rows, val_tol_pct=args.val_tol_pct)
        baseline = None
        for r in rows:
            if r.get("config") == "fp32_unscaled" and r.get("status") == "ok":
                baseline = r.get("final_val_loss")
                break

        best_row = {
            "run_id": spec.run_id,
            "section": spec.section,
            "name": spec.name,
            "baseline_fp32_val_loss": baseline if baseline is not None else float("nan"),
            "best_config": best.get("config", "none"),
            "best_val_loss": best.get("final_val_loss", float("nan")),
            "best_mean_epoch_s": best.get("mean_epoch_s", float("nan")),
            "best_train_samples_per_s": best.get("train_samples_per_s", float("nan")),
            "best_peak_mem_mib": best.get("peak_mem_mib", float("nan")),
            "best_nan_inf_events": best.get("nan_inf_events", -1),
        }
        if _is_finite_number(best_row["baseline_fp32_val_loss"]) and _is_finite_number(best_row["best_val_loss"]):
            b = float(best_row["baseline_fp32_val_loss"])
            if b != 0:
                best_row["best_vs_fp32_val_pct"] = 100.0 * (float(best_row["best_val_loss"]) - b) / b
            else:
                best_row["best_vs_fp32_val_pct"] = float("nan")
        else:
            best_row["best_vs_fp32_val_pct"] = float("nan")
        best_rows.append(best_row)

    # Write aggregate outputs.
    meta_path = output_root / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta_list, f, indent=2)

    if args.dry_run:
        print("\nDry run complete. Planned commands and metadata:")
        print(f"- {output_root}")
        print(f"- {meta_path}")
        return 0

    consolidated_json = output_root / "consolidated_results.json"
    consolidated_csv = output_root / "consolidated_results.csv"
    with open(consolidated_json, "w") as f:
        json.dump(consolidated_rows, f, indent=2)
    _write_csv(consolidated_csv, consolidated_rows)

    best_json = output_root / "best_by_run.json"
    best_csv = output_root / "best_by_run.csv"
    with open(best_json, "w") as f:
        json.dump(best_rows, f, indent=2)
    _write_csv(best_csv, best_rows)

    print("\nSweep outputs:")
    print(f"- {output_root}")
    print(f"- {consolidated_csv}")
    print(f"- {best_csv}")
    _print_recommendations(best_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
