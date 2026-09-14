#!/usr/bin/env python
"""Compare FDE state and gradient accuracy across precision/scaling modes.

The reference is an independent float64/autograd implementation of the same
uniform product-rectangle recurrence used by ``predictor_fdeint``.  Therefore,
the reported errors measure precision and adjoint effects separately from the
continuous-time discretization error.

Run from the repository root with, for example:

    PYTHONPATH="$PWD/rampfde:$PYTHONPATH" \
        python tests/float16_stress_test/gradient_precision_table.py
"""

import argparse
import csv
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# Prefer the package in this checkout over any unrelated installed ``rampde``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "rampfde"))

from rampde import DynamicScaler, predictor_fdeint


class LinearFDE(nn.Module):
    """Scalar test equation ``D^beta y = -lambda*y`` with learnable lambda."""

    def __init__(self, lam: float, *, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.lam = nn.Parameter(
            torch.tensor(lam, dtype=dtype, device=device)
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Explicitly cast the function inputs and parameter so autocast tests
        # the low-precision function-evaluation path, not only the solver
        # bookkeeping.
        if torch.is_autocast_enabled():
            low_dtype = torch.get_autocast_dtype("cuda")
            y = y.to(low_dtype)
            lam = self.lam.to(low_dtype)
        else:
            lam = self.lam
        return -lam * y


def reference_predictor(
    model: nn.Module,
    y0: torch.Tensor,
    beta: float,
    T: float,
    step_size: float,
) -> torch.Tensor:
    """Run the uniform predictor recurrence with ordinary PyTorch autograd."""
    n_steps = int(round(T / step_size)) + 1
    tspan = torch.linspace(
        0.0, T, n_steps, dtype=y0.dtype, device=y0.device
    )
    h = tspan[1] - tspan[0]
    coefficient = h.pow(beta) / math.gamma(beta + 1.0)
    f_history: List[torch.Tensor] = []
    y_current = y0

    for k in range(n_steps - 1):
        f_history.append(model(tspan[k], y_current))
        j = torch.arange(k + 1, dtype=y0.dtype, device=y0.device)
        weights = coefficient * (
            (k + 1 - j).pow(beta) - (k - j).pow(beta)
        )
        convolution = sum(
            (weights[index] * value for index, value in enumerate(f_history)),
            torch.zeros_like(y0),
        )
        y_current = y0 + convolution

    return y_current


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    # The numerical run is on CUDA while the independent reference is on CPU.
    # Compare them in one device and in float64.
    value = value.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    denominator = torch.linalg.norm(reference).clamp_min(1e-30)
    return float((torch.linalg.norm(value - reference) / denominator).item())


def make_reference(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    model = LinearFDE(args.lam, dtype=torch.float64, device=torch.device("cpu"))
    y0 = torch.tensor(
        [args.y0], dtype=torch.float64, device="cpu", requires_grad=True
    )
    y_T = reference_predictor(model, y0, args.beta, args.T, args.step_size)
    loss = 0.5 * y_T.square().sum()
    loss.backward()
    return {
        "y_T": y_T.detach(),
        "grad_y0": y0.grad.detach().clone(),
        "grad_lam": model.lam.grad.detach().clone(),
    }


def run_case(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    scaling: str,
) -> Dict[str, object]:
    model = LinearFDE(args.lam, dtype=torch.float32, device=device)
    y0 = torch.tensor(
        [args.y0], dtype=torch.float32, device=device, requires_grad=True
    )

    if scaling == "dynamic":
        loss_scaler = DynamicScaler(dtype_low=dtype)
    else:
        loss_scaler = False

    autocast_context = (
        nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type="cuda", dtype=dtype)
    )

    try:
        with autocast_context:
            y_T = predictor_fdeint(
                model,
                y0,
                beta=args.beta,
                t=args.T,
                step_size=args.step_size,
                loss_scaler=loss_scaler,
            )
            loss = 0.5 * y_T.float().square().sum()
        loss.backward()

        grad_y0 = y0.grad.detach().clone()
        grad_lam = model.lam.grad.detach().clone()
        values = {
            "status": "ok",
            "rel_err_y_T": relative_error(y_T, args.reference["y_T"]),
            "rel_err_grad_y0": relative_error(
                grad_y0, args.reference["grad_y0"]
            ),
            "rel_err_grad_lam": relative_error(
                grad_lam, args.reference["grad_lam"]
            ),
            "y_T_finite": bool(torch.isfinite(y_T).all().item()),
            "grad_y0_finite": bool(torch.isfinite(grad_y0).all().item()),
            "grad_lam_finite": bool(torch.isfinite(grad_lam).all().item()),
        }
    except (RuntimeError, ValueError, OverflowError) as error:
        values = {
            "status": f"failed: {type(error).__name__}: {error}",
            "rel_err_y_T": None,
            "rel_err_grad_y0": None,
            "rel_err_grad_lam": None,
            "y_T_finite": False,
            "grad_y0_finite": False,
            "grad_lam_finite": False,
        }

    return {
        "dtype": str(dtype),
        "scaling": scaling,
        **values,
    }


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: object) -> str:
    if value is None:
        return "FAIL"
    if isinstance(value, float):
        return f"{value:.4e}"
    return str(value)


def write_markdown(
    rows: List[Dict[str, object]], path: str, args: argparse.Namespace
) -> None:
    columns = [
        ("Precision", "dtype"),
        ("Scaling", "scaling"),
        ("Status", "status"),
        ("RE terminal state", "rel_err_y_T"),
        ("RE dL/dy0", "rel_err_grad_y0"),
        ("RE dL/dlambda", "rel_err_grad_lam"),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# FDE Precision and Adjoint-Gradient Accuracy\n\n")
        handle.write(
            "Reference: float64 ordinary-autograd evaluation of the same "
            "uniform predictor recurrence. The loss is "
            "`L = 1/2 ||y(T)||^2`.\n\n"
        )
        handle.write(
            f"Parameters: beta={args.beta}, T={args.T}, "
            f"step size={args.step_size}, lambda={args.lam}, y0={args.y0}.\n\n"
        )
        handle.write("| " + " | ".join(label for label, _ in columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write(
                "| " + " | ".join(_format(row[key]) for _, key in columns) + " |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--y0", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="gradient_precision_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")

    device = torch.device(args.device)
    args.reference = make_reference(args)
    cases = [
        (torch.float32, "unscaled"),
        (torch.float32, "dynamic"),
        (torch.bfloat16, "unscaled"),
        (torch.bfloat16, "dynamic"),
        (torch.float16, "unscaled"),
        (torch.float16, "dynamic"),
    ]
    rows = [run_case(args, device, dtype, scaling) for dtype, scaling in cases]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "gradient_precision_table.csv")
    markdown_path = os.path.join(args.output_dir, "gradient_precision_table.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print("\nRelative errors:")
    for row in rows:
        print(
            f"  {row['dtype']:>14} {row['scaling']:>9} | "
            f"state={_format(row['rel_err_y_T']):>10} | "
            f"y0={_format(row['rel_err_grad_y0']):>10} | "
            f"lambda={_format(row['rel_err_grad_lam']):>10} | "
            f"{row['status']}"
        )


if __name__ == "__main__":
    main()
