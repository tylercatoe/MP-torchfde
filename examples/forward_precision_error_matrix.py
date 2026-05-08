#!/usr/bin/env python3
"""Forward-pass precision error benchmark for fractional ODE solving.

This script builds a manufactured fractional IVP with a known exact solution and
compares absolute errors across:
  - float64 reference solve
  - float32 baseline
  - mixed precision (autocast low dtype, params/state in fp32)
  - low-only precision (params/state directly cast to low dtype)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Ensure local imports work when run from repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchfde import fdeint


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gamma(x: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x_t = torch.tensor(x, dtype=dtype, device=device)
    return torch.exp(torch.lgamma(x_t))


class ManufacturedFractionalRHS(nn.Module):
    """RHS for D^beta y = A y + g(t), where g is chosen so y_true is exact."""

    def __init__(
        self,
        A: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        beta: float,
    ):
        super().__init__()
        self.register_buffer("A", A)    # (d, d)
        self.register_buffer("c0", c0)  # (d,)
        self.register_buffer("c1", c1)
        self.register_buffer("c2", c2)
        self.register_buffer("c3", c3)
        self.beta = float(beta)

    def y_true(self, t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        t = t.to(device=device, dtype=dtype)
        c0 = self.c0.to(device=device, dtype=dtype)
        c1 = self.c1.to(device=device, dtype=dtype)
        c2 = self.c2.to(device=device, dtype=dtype)
        c3 = self.c3.to(device=device, dtype=dtype)
        b = self.beta
        return c0 + c1 * torch.pow(t, b) + c2 * torch.pow(t, 2.0 * b) + c3 * torch.pow(t, 3.0 * b)

    def caputo_y_true(self, t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        t = t.to(device=device, dtype=dtype)
        c1 = self.c1.to(device=device, dtype=dtype)
        c2 = self.c2.to(device=device, dtype=dtype)
        c3 = self.c3.to(device=device, dtype=dtype)
        b = self.beta

        # D^b t^{k b} = Gamma(k b + 1) / Gamma((k-1)b + 1) * t^{(k-1)b}
        g1 = _gamma(b + 1.0, dtype, device)
        g2 = _gamma(2.0 * b + 1.0, dtype, device) / _gamma(b + 1.0, dtype, device)
        g3 = _gamma(3.0 * b + 1.0, dtype, device) / _gamma(2.0 * b + 1.0, dtype, device)

        term1 = c1 * g1
        term2 = c2 * g2 * torch.pow(t, b)
        term3 = c3 * g3 * torch.pow(t, 2.0 * b)
        return term1 + term2 + term3

    def forcing(self, t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        y_t = self.y_true(t, dtype=dtype, device=device)            # (d,)
        dy_t = self.caputo_y_true(t, dtype=dtype, device=device)    # (d,)
        A = self.A.to(device=device, dtype=dtype)
        return dy_t - A @ y_t

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dtype = y.dtype
        device = y.device
        A = self.A.to(device=device, dtype=dtype)
        g_t = self.forcing(t, dtype=dtype, device=device)

        # Support both (d,) and (batch, d) states.
        if y.dim() == 1:
            return A @ y + g_t
        return y @ A.t() + g_t.unsqueeze(0)


@dataclass
class RunResult:
    label: str
    solve_dtype: str
    mode: str
    max_abs_err: float
    mean_abs_err: float
    l2_abs_err: float


def _autocast_ctx(device: torch.device, low_dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=low_dtype)
    if device.type == "cpu" and low_dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _solve_forward(
    rhs: ManufacturedFractionalRHS,
    y0: torch.Tensor,
    beta: float,
    t_final: float,
    step_size: float,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    beta_t = torch.tensor(beta, dtype=y0.dtype, device=device)
    t_t = torch.tensor(t_final, dtype=y0.dtype, device=device)
    h_t = torch.tensor(step_size, dtype=y0.dtype, device=device)
    return fdeint(rhs, y0, beta=beta_t, t=t_t, step_size=h_t, method=method)


def _error_stats(y_pred: torch.Tensor, y_exact: torch.Tensor) -> Dict[str, float]:
    diff = (y_pred.to(torch.float64) - y_exact.to(torch.float64)).abs()
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "l2_abs_err": torch.linalg.norm(diff).item(),
    }


def run_case(
    label: str,
    rhs_template: ManufacturedFractionalRHS,
    y0_template: torch.Tensor,
    beta: float,
    t_final: float,
    step_size: float,
    method: str,
    device: torch.device,
    solve_dtype: torch.dtype,
    mode: str,
    low_dtype: Optional[torch.dtype] = None,
) -> RunResult:
    rhs = ManufacturedFractionalRHS(
        A=rhs_template.A.clone(),
        c0=rhs_template.c0.clone(),
        c1=rhs_template.c1.clone(),
        c2=rhs_template.c2.clone(),
        c3=rhs_template.c3.clone(),
        beta=rhs_template.beta,
    ).to(device=device, dtype=solve_dtype if mode == "low-only" else torch.float32 if mode == "mixed" else solve_dtype)

    y_exact = rhs_template.y_true(
        t=torch.tensor(t_final, dtype=torch.float64, device=device),
        dtype=torch.float64,
        device=device,
    )

    if mode == "mixed":
        assert low_dtype is not None
        y0 = y0_template.to(device=device, dtype=torch.float32)
        with _autocast_ctx(device, low_dtype):
            y_pred = _solve_forward(
                rhs=rhs,
                y0=y0,
                beta=beta,
                t_final=t_final,
                step_size=step_size,
                method=method,
                device=device,
            )
    else:
        y0 = y0_template.to(device=device, dtype=solve_dtype)
        y_pred = _solve_forward(
            rhs=rhs,
            y0=y0,
            beta=beta,
            t_final=t_final,
            step_size=step_size,
            method=method,
            device=device,
        )

    stats = _error_stats(y_pred, y_exact)
    return RunResult(
        label=label,
        solve_dtype=str(y_pred.dtype).replace("torch.", ""),
        mode=mode,
        max_abs_err=stats["max_abs_err"],
        mean_abs_err=stats["mean_abs_err"],
        l2_abs_err=stats["l2_abs_err"],
    )


def print_table(rows: List[RunResult]) -> None:
    headers = ["label", "mode", "solve_dtype", "max_abs_err", "mean_abs_err", "l2_abs_err"]

    def row_map(r: RunResult) -> Dict[str, Any]:
        return {
            "label": r.label,
            "mode": r.mode,
            "solve_dtype": r.solve_dtype,
            "max_abs_err": f"{r.max_abs_err:.6e}",
            "mean_abs_err": f"{r.mean_abs_err:.6e}",
            "l2_abs_err": f"{r.l2_abs_err:.6e}",
        }

    mapped = [row_map(r) for r in rows]
    widths = {h: max(len(h), *(len(m[h]) for m in mapped)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for m in mapped:
        print(" | ".join(m[h].ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward precision error matrix for known fractional IVP.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--method", type=str, default="predictor")
    p.add_argument("--beta", type=float, default=0.73)
    p.add_argument("--t-final", type=float, default=1.2)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--dim", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--csv-out", type=str, default="")
    p.add_argument("--json-out", type=str, default="")
    return p.parse_args()


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

    d = args.dim
    # Fixed dense matrix for a coupled "complicated" linear fractional system.
    A = torch.tensor(
        [
            [-0.80, 0.22, -0.05, 0.10, -0.08, 0.03],
            [0.15, -0.65, 0.21, -0.09, 0.04, -0.02],
            [-0.12, 0.18, -0.72, 0.16, -0.03, 0.05],
            [0.07, -0.11, 0.14, -0.58, 0.19, -0.06],
            [-0.09, 0.05, -0.02, 0.17, -0.69, 0.13],
            [0.04, -0.03, 0.08, -0.07, 0.16, -0.61],
        ],
        dtype=torch.float64,
    )
    if d != 6:
        # For custom dim, generate a stable dense matrix deterministically.
        g = torch.Generator().manual_seed(args.seed)
        A = (torch.randn(d, d, generator=g, dtype=torch.float64) * 0.12) - 0.55 * torch.eye(d, dtype=torch.float64)

    idx = torch.arange(d, dtype=torch.float64)
    c0 = 0.4 + 0.07 * (idx + 1.0)
    c1 = -0.2 + 0.05 * (idx + 0.5)
    c2 = 0.1 * torch.cos(0.7 * idx + 0.2)
    c3 = 0.06 * torch.sin(0.9 * idx + 0.3)

    rhs_template = ManufacturedFractionalRHS(A=A, c0=c0, c1=c1, c2=c2, c3=c3, beta=args.beta).to(
        device=device, dtype=torch.float64
    )
    y0_template = c0.clone().to(device=device, dtype=torch.float64)

    rows: List[RunResult] = []

    # High-precision and fp32 references.
    rows.append(
        run_case(
            label="float64_reference",
            rhs_template=rhs_template,
            y0_template=y0_template,
            beta=args.beta,
            t_final=args.t_final,
            step_size=args.step_size,
            method=args.method,
            device=device,
            solve_dtype=torch.float64,
            mode="direct",
        )
    )
    rows.append(
        run_case(
            label="float32_baseline",
            rhs_template=rhs_template,
            y0_template=y0_template,
            beta=args.beta,
            t_final=args.t_final,
            step_size=args.step_size,
            method=args.method,
            device=device,
            solve_dtype=torch.float32,
            mode="direct",
        )
    )

    # BF16 comparison.
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cpu":
        bf16_supported = True
    if bf16_supported:
        rows.append(
            run_case(
                label="bf16_mixed",
                rhs_template=rhs_template,
                y0_template=y0_template,
                beta=args.beta,
                t_final=args.t_final,
                step_size=args.step_size,
                method=args.method,
                device=device,
                solve_dtype=torch.float32,
                mode="mixed",
                low_dtype=torch.bfloat16,
            )
        )
        rows.append(
            run_case(
                label="bf16_low_only",
                rhs_template=rhs_template,
                y0_template=y0_template,
                beta=args.beta,
                t_final=args.t_final,
                step_size=args.step_size,
                method=args.method,
                device=device,
                solve_dtype=torch.bfloat16,
                mode="low-only",
            )
        )
    else:
        print("Skipping bf16 rows: bf16 not supported on this device.")

    # FP16 comparison (CUDA only in this benchmark).
    if device.type == "cuda":
        rows.append(
            run_case(
                label="fp16_mixed",
                rhs_template=rhs_template,
                y0_template=y0_template,
                beta=args.beta,
                t_final=args.t_final,
                step_size=args.step_size,
                method=args.method,
                device=device,
                solve_dtype=torch.float32,
                mode="mixed",
                low_dtype=torch.float16,
            )
        )
        rows.append(
            run_case(
                label="fp16_low_only",
                rhs_template=rhs_template,
                y0_template=y0_template,
                beta=args.beta,
                t_final=args.t_final,
                step_size=args.step_size,
                method=args.method,
                device=device,
                solve_dtype=torch.float16,
                mode="low-only",
            )
        )
    else:
        print("Skipping fp16 rows: fp16 low-only/mixed comparison is CUDA-only here.")

    print("\nForward Precision Error Matrix (absolute errors vs exact manufactured solution)")
    print_table(rows)

    # Convenience comparisons.
    by_label = {r.label: r for r in rows}
    if "bf16_mixed" in by_label and "bf16_low_only" in by_label:
        ratio = by_label["bf16_low_only"].mean_abs_err / max(by_label["bf16_mixed"].mean_abs_err, 1e-30)
        print(f"\nbf16 error ratio (low_only / mixed): {ratio:.3f}x")
    if "fp16_mixed" in by_label and "fp16_low_only" in by_label:
        ratio = by_label["fp16_low_only"].mean_abs_err / max(by_label["fp16_mixed"].mean_abs_err, 1e-30)
        print(f"fp16 error ratio (low_only / mixed): {ratio:.3f}x")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["label", "mode", "solve_dtype", "max_abs_err", "mean_abs_err", "l2_abs_err"],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        "label": r.label,
                        "mode": r.mode,
                        "solve_dtype": r.solve_dtype,
                        "max_abs_err": r.max_abs_err,
                        "mean_abs_err": r.mean_abs_err,
                        "l2_abs_err": r.l2_abs_err,
                    }
                )
        print(f"Saved CSV: {args.csv_out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                [
                    {
                        "label": r.label,
                        "mode": r.mode,
                        "solve_dtype": r.solve_dtype,
                        "max_abs_err": r.max_abs_err,
                        "mean_abs_err": r.mean_abs_err,
                        "l2_abs_err": r.l2_abs_err,
                    }
                    for r in rows
                ],
                f,
                indent=2,
            )
        print(f"Saved JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
