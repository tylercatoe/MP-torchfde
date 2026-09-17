#!/usr/bin/env python
"""Compare FDE state and gradient accuracy across precision/scaling modes.

The reference is an independent float64/autograd implementation of the same
selected mesh and predictor or predictor-corrector recurrence used by
``predictor_fdeint``. Therefore, the reported errors measure precision and
adjoint effects separately from the continuous-time discretization error.

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
    graded_time: bool,
    predictor_corrector: bool,
) -> torch.Tensor:
    """Run the requested recurrence with ordinary float64 PyTorch autograd."""
    n_steps = int(round(T / step_size)) + 1
    if graded_time:
        index = torch.arange(n_steps, dtype=y0.dtype, device=y0.device)
        center = (n_steps - 1) / 2
        q = 1 - torch.abs(index - center) / center
        grading_power = (2.0 - beta) / beta
        tspan = T / 2 * (
            1
            + torch.sign(index - center)
            * (1 - torch.pow(q, grading_power))
        )
    else:
        tspan = torch.linspace(
            0.0, T, n_steps, dtype=y0.dtype, device=y0.device
        )

    f_history: List[torch.Tensor] = []
    y_current = y0

    for k in range(n_steps - 1):
        f_history.append(model(tspan[k], y_current))
        t_next = tspan[k + 1]
        left = tspan[: k + 1]
        right = tspan[1: k + 2]
        predictor_weights = (
            (t_next - left).pow(beta) - (t_next - right).pow(beta)
        ) / math.gamma(beta + 1.0)
        predictor = y0 + sum(
            (
                predictor_weights[index] * value
                for index, value in enumerate(f_history)
            ),
            torch.zeros_like(y0),
        )

        if not predictor_corrector:
            y_current = predictor
            continue

        f_predictor = model(t_next, predictor)
        corrected_history = torch.zeros_like(y0)
        if k > 0:
            interval_left = tspan[:k]
            interval_right = tspan[1: k + 1]
            interval_width = interval_right - interval_left
            x_left = t_next - interval_left
            x_right = t_next - interval_right
            i0 = (
                x_left.pow(beta) - x_right.pow(beta)
            ) / beta
            i1 = (
                x_left.pow(beta + 1.0) - x_right.pow(beta + 1.0)
            ) / (beta + 1.0)
            left_weights = (
                i1 - x_right * i0
            ) / (interval_width * math.gamma(beta))
            right_weights = (
                x_left * i0 - i1
            ) / (interval_width * math.gamma(beta))
            corrected_history = sum(
                (
                    left_weights[index] * f_history[index]
                    + right_weights[index] * f_history[index + 1]
                    for index in range(k)
                ),
                torch.zeros_like(y0),
            )

        local_weight = (
            (tspan[k + 1] - tspan[k]).pow(beta)
            / math.gamma(beta + 2.0)
        )
        y_current = (
            y0
            + corrected_history
            + local_weight * (beta * f_history[k] + f_predictor)
        )

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
    y_T = reference_predictor(
        model,
        y0,
        args.beta,
        args.T,
        args.step_size,
        args.graded_time,
        args.predictor_corrector,
    )
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
        adj_dtype = dtype if args.adjoint_storage == "match" else torch.float32
        with autocast_context:
            y_T = predictor_fdeint(
                model,
                y0,
                beta=args.beta,
                t=args.T,
                step_size=args.step_size,
                loss_scaler=loss_scaler,
                adj_dtype=adj_dtype,
                graded_time=args.graded_time,
                predictor_corrector=args.predictor_corrector,
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
            "final_scale": (
                loss_scaler.S
                if isinstance(loss_scaler, DynamicScaler)
                else "n/a"
            ),
            "scale_events": (
                len(loss_scaler.scale_history)
                if isinstance(loss_scaler, DynamicScaler)
                else 0
            ),
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
            "final_scale": (
                loss_scaler.S
                if isinstance(loss_scaler, DynamicScaler)
                else "n/a"
            ),
            "scale_events": (
                len(loss_scaler.scale_history)
                if isinstance(loss_scaler, DynamicScaler)
                else 0
            ),
        }

    return {
        "mesh": "graded" if args.graded_time else "uniform",
        "method": (
            "predictor-corrector"
            if args.predictor_corrector
            else "predictor"
        ),
        "dtype": str(dtype),
        "adjoint_dtype": str(
            dtype if args.adjoint_storage == "match" else torch.float32
        ),
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
        ("Adjoint storage", "adjoint_dtype"),
        ("Scaling", "scaling"),
        ("Status", "status"),
        ("RE terminal state", "rel_err_y_T"),
        ("RE dL/dy0", "rel_err_grad_y0"),
        ("RE dL/dlambda", "rel_err_grad_lam"),
        ("Final scale", "final_scale"),
        ("Scale events", "scale_events"),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# FDE Precision and Adjoint-Gradient Accuracy\n\n")
        handle.write(
            "Reference: float64 ordinary-autograd evaluation of the same "
            "selected mesh and integration recurrence. The loss is "
            "`L = 1/2 ||y(T)||^2`.\n\n"
        )
        handle.write(
            f"Parameters: beta={args.beta}, T={args.T}, "
            f"step size={args.step_size}, lambda={args.lam}, y0={args.y0}.\n\n"
        )
        handle.write(
            f"Mesh: {'graded' if args.graded_time else 'uniform'}; "
            "method: "
            f"{'predictor-corrector' if args.predictor_corrector else 'predictor'}; "
            f"adjoint storage: {args.adjoint_storage}.\n\n"
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
    parser.add_argument(
        "--adjoint-storage",
        choices=["match", "float32"],
        default="match",
        help="Store adjoint history in the tested dtype or float32",
    )
    parser.add_argument(
        "--graded-time",
        action="store_true",
        help="Use the symmetric graded mesh",
    )
    parser.add_argument(
        "--predictor-corrector",
        action="store_true",
        help="Use the predictor-corrector integration rule",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/float16_stress_test/gradient_precision_results",
    )
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
            f"adj={row['adjoint_dtype']:>14} | "
            f"state={_format(row['rel_err_y_T']):>10} | "
            f"y0={_format(row['rel_err_grad_y0']):>10} | "
            f"lambda={_format(row['rel_err_grad_lam']):>10} | "
            f"scale={_format(row['final_scale']):>10} | "
            f"{row['status']}"
        )


if __name__ == "__main__":
    main()
