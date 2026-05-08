#!/usr/bin/env python3
"""
Run a mixed-precision experiment matrix for fractional adjoint training.

This script compares:
1) fp32 + unscaled adjoint
2) bf16 + unscaled adjoint (if supported)
3) fp16 + safe adjoint
4) fp16 + dynamic-scaled adjoint

Metrics logged per configuration:
- final train loss
- final val loss
- NaN/Inf event count
- mean epoch time
- throughput (train samples/sec)
- peak GPU memory (MiB)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure local imports work when running from repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchfde import DynamicScaler, fdeint, fdeint_adjoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FractionalVectorField(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        return self.net(y)


class FractionalRegressor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        out_dim: int,
        beta: float,
        t_final: float,
        step_size: float,
        method: str = "predictor-f",
    ):
        super().__init__()
        self.func = FractionalVectorField(dim=dim, hidden=hidden)
        self.head = nn.Linear(dim, out_dim)
        self.method = method
        self.register_buffer("_beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("_t_final", torch.tensor(t_final, dtype=torch.float32))
        self.register_buffer("_step_size", torch.tensor(step_size, dtype=torch.float32))

    def _solver_scalars(self, x: torch.Tensor):
        beta = self._beta.to(device=x.device, dtype=x.dtype)
        t_final = self._t_final.to(device=x.device, dtype=x.dtype)
        step = self._step_size.to(device=x.device, dtype=x.dtype)
        return beta, t_final, step

    def forward(self, x: torch.Tensor, loss_scaler: Any = None):
        beta, t_final, step = self._solver_scalars(x)
        z = fdeint_adjoint(
            self.func,
            x,
            beta=beta,
            t=t_final,
            step_size=step,
            method=self.method,
            loss_scaler=loss_scaler,
        )
        return self.head(z)


class TeacherModel(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        out_dim: int,
        beta: float,
        t_final: float,
        step_size: float,
        method: str = "predictor",
    ):
        super().__init__()
        self.func = FractionalVectorField(dim=dim, hidden=hidden)
        self.head = nn.Linear(dim, out_dim)
        self.method = method
        self.register_buffer("_beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("_t_final", torch.tensor(t_final, dtype=torch.float32))
        self.register_buffer("_step_size", torch.tensor(step_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        beta = self._beta.to(device=x.device, dtype=x.dtype)
        t_final = self._t_final.to(device=x.device, dtype=x.dtype)
        step = self._step_size.to(device=x.device, dtype=x.dtype)
        z = fdeint(
            self.func,
            x,
            beta=beta,
            t=t_final,
            step_size=step,
            method=self.method,
        )
        return self.head(z)


@dataclass
class ConfigSpec:
    name: str
    autocast_dtype: Optional[torch.dtype]
    loss_scaler_mode: str  # "false" or "dynamic"


def build_configs(device: torch.device) -> List[ConfigSpec]:
    configs: List[ConfigSpec] = [
        ConfigSpec("fp32_unscaled", autocast_dtype=None, loss_scaler_mode="false"),
    ]

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        configs.append(ConfigSpec("bf16_unscaled", autocast_dtype=torch.bfloat16, loss_scaler_mode="false"))

    if device.type == "cuda":
        configs.append(ConfigSpec("fp16_safe", autocast_dtype=torch.float16, loss_scaler_mode="false"))
        configs.append(ConfigSpec("fp16_dynamic", autocast_dtype=torch.float16, loss_scaler_mode="dynamic"))

    return configs


def make_teacher_targets(
    teacher: TeacherModel,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    teacher.eval()
    targets = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(device)
            yb = teacher(xb)
            targets.append(yb.cpu())
    return torch.cat(targets, dim=0)


def make_autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_one_config(
    cfg: ConfigSpec,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict[str, Any]:
    set_seed(args.seed + 100)
    model = FractionalRegressor(
        dim=args.dim,
        hidden=args.hidden,
        out_dim=args.out_dim,
        beta=args.beta,
        t_final=args.t_final,
        step_size=args.step_size,
        method=args.adjoint_method,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    if cfg.loss_scaler_mode == "dynamic":
        loss_scaler: Any = DynamicScaler(dtype_low=torch.float16)
    else:
        loss_scaler = False

    epoch_times: List[float] = []
    nan_inf_events = 0
    last_train_loss = float("nan")

    for _ in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_start = time.perf_counter()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with make_autocast_context(device, cfg.autocast_dtype):
                pred = model(xb, loss_scaler=loss_scaler)
                loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                nan_inf_events += 1
                continue

            loss.backward()

            grads_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                nan_inf_events += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            running_loss += loss.detach().float().item() * xb.shape[0]
            running_count += xb.shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        if running_count > 0:
            last_train_loss = running_loss / running_count

    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with make_autocast_context(device, cfg.autocast_dtype):
                pred = model(xb, loss_scaler=loss_scaler)
                vloss = loss_fn(pred, yb)
            if torch.isfinite(vloss):
                val_loss_sum += vloss.detach().float().item() * xb.shape[0]
                val_count += xb.shape[0]
            else:
                nan_inf_events += 1

    final_val_loss = val_loss_sum / max(1, val_count)
    mean_epoch_s = float(np.mean(epoch_times))
    train_throughput = args.train_samples / mean_epoch_s if mean_epoch_s > 0 else float("nan")
    peak_mem_mib = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    return {
        "config": cfg.name,
        "final_train_loss": float(last_train_loss),
        "final_val_loss": float(final_val_loss),
        "nan_inf_events": int(nan_inf_events),
        "mean_epoch_s": float(mean_epoch_s),
        "train_samples_per_s": float(train_throughput),
        "peak_mem_mib": float(peak_mem_mib),
        "status": "ok",
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "config",
        "final_train_loss",
        "final_val_loss",
        "val_vs_fp32_pct",
        "nan_inf_events",
        "mean_epoch_s",
        "train_samples_per_s",
        "peak_mem_mib",
        "status",
    ]

    baseline = None
    for r in rows:
        if r["config"] == "fp32_unscaled" and r["status"] == "ok":
            baseline = r["final_val_loss"]
            break

    for r in rows:
        if baseline is None or not np.isfinite(baseline) or baseline == 0 or r["status"] != "ok":
            r["val_vs_fp32_pct"] = float("nan")
        else:
            r["val_vs_fp32_pct"] = 100.0 * (r["final_val_loss"] - baseline) / baseline

    def fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6f}"
        return str(v)

    widths = {h: max(len(h), *(len(fmt(row.get(h, ""))) for row in rows)) for h in headers}

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(fmt(row.get(h, "")).ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed-precision experiment matrix for torchfde fractional adjoint."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--adjoint-method", type=str, default="predictor-f")
    parser.add_argument("--teacher-method", type=str, default="predictor")
    parser.add_argument("--targets-batch-size", type=int, default=512)
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print("Generating synthetic fractional supervised dataset...")

    x_train = torch.randn(args.train_samples, args.dim, dtype=torch.float32)
    x_val = torch.randn(args.val_samples, args.dim, dtype=torch.float32)

    set_seed(args.seed + 1)
    teacher = TeacherModel(
        dim=args.dim,
        hidden=args.hidden,
        out_dim=args.out_dim,
        beta=args.beta,
        t_final=args.t_final,
        step_size=args.step_size,
        method=args.teacher_method,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    y_train = make_teacher_targets(teacher, x_train, args.targets_batch_size, device=device)
    y_val = make_teacher_targets(teacher, x_val, args.targets_batch_size, device=device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    configs = build_configs(device)
    print("Running configs:", ", ".join(cfg.name for cfg in configs))

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        try:
            row = train_one_config(cfg, args, device, train_loader, val_loader)
        except Exception as exc:  # Keep matrix running if one config fails.
            row = {
                "config": cfg.name,
                "final_train_loss": float("nan"),
                "final_val_loss": float("nan"),
                "nan_inf_events": -1,
                "mean_epoch_s": float("nan"),
                "train_samples_per_s": float("nan"),
                "peak_mem_mib": float("nan"),
                "status": f"error: {type(exc).__name__}",
            }
        results.append(row)

    print("\nMixed-Precision Fractional Matrix Results")
    print_table(results)

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV: {args.csv_out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
