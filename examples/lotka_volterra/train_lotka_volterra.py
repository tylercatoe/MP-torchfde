#!/usr/bin/env python
"""Train a fractional Lotka--Volterra model on uniform or graded meshes.

The experiment estimates the positive parameters [a, b, c, d] in

    D^beta x = x (a - c y),
    D^beta y = -y (b - d x),

from noisy terminal observations.  Every run recreates exactly the same
training/validation data and starts from exactly the same parameter values.

The uniform and graded cases both use the predictor--corrector recurrence.
The solver's public API historically uses ``graded_time=False`` for the
predictor-only method, so this example constructs the time grid explicitly
and calls the exported solver class with its predictor--corrector branch for
both meshes.  This keeps the comparison focused on mesh grading.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


# Make the in-repository rampde package importable when this file is run from
# the example directory or from the repository root.
_RAMPDE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "rampfde")
)
if _RAMPDE_ROOT not in sys.path:
    sys.path.insert(0, _RAMPDE_ROOT)

from rampde.predictor_fdeint import (  # noqa: E402
    PredictorFDESolverUnscaled,
    PredictorFDESolverUnscaledSafe,
)


TRUE_PARAMS = torch.tensor([1.0, 0.5, 1.0, 0.3], dtype=torch.float32)
BETA = 0.7
T_END = 5.0
STEP_SIZE = 0.1
DATA_STEP_SIZE = 0.02


def lotka_volterra_rhs(params: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Right-hand side of the fractional Lotka--Volterra system."""
    a, b, c, d = params
    x, prey = y[..., 0], y[..., 1]
    dx = x * (a - c * prey)
    dy = -prey * (b - d * x)
    return torch.stack((dx, dy), dim=-1)


class LotkaVolterraFunc(nn.Module):
    """Learnable positive Lotka--Volterra parameters."""

    def __init__(self) -> None:
        super().__init__()
        init = torch.tensor([0.99, 0.48, 1.05, 0.33], dtype=torch.float32)
        self.log_params = nn.Parameter(torch.log(init))

    @property
    def params(self) -> torch.Tensor:
        return self.log_params.exp()

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return lotka_volterra_rhs(self.params, t, y)


class FixedLotkaVolterraFunc(nn.Module):
    """Non-learnable RHS used to generate the common synthetic targets."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("params", TRUE_PARAMS.to(device=device, dtype=torch.float64))

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return lotka_volterra_rhs(self.params, t, y)


def make_tspan(
    t_end: float,
    step_size: float,
    beta: float,
    mesh: str,
    device: torch.device,
) -> torch.Tensor:
    """Construct the same uniform or double-graded mesh used by the solver."""
    n_steps = int(round(t_end / step_size)) + 1
    if mesh == "uniform":
        return torch.linspace(0.0, t_end, n_steps, dtype=torch.float32, device=device)

    indices = torch.arange(n_steps, dtype=torch.float32, device=device)
    midpoint = (n_steps - 1) / 2.0
    exponent = (2.0 - beta) / beta
    q = (1.0 - torch.abs(indices - midpoint) / midpoint).clamp_min(0.0)
    return t_end / 2.0 * (
        1.0 + torch.sign(indices - midpoint) * (1.0 - torch.pow(q, exponent))
    )


def solve_predictor_corrector(
    func: nn.Module,
    y0: torch.Tensor,
    beta: float,
    t_end: float,
    step_size: float,
    mesh: str,
    precision: str,
) -> torch.Tensor:
    """Solve using the same predictor--corrector method on either mesh."""
    tspan = make_tspan(t_end, step_size, beta, mesh, y0.device)
    params = tuple(func.parameters())

    if precision == "fp32":
        # ``graded_time=True`` selects the predictor--corrector branch.  The
        # explicitly supplied uniform tspan makes its mesh uniform here.
        return PredictorFDESolverUnscaled.apply(
            func, y0, tspan, beta, None, None, True, *params
        )

    if precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("The fp16 experiment requires a CUDA GPU.")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = PredictorFDESolverUnscaledSafe.apply(
                func, y0, tspan, beta, torch.float16, False, True, *params
            )
        return out.float()

    raise ValueError(f"Unknown precision: {precision}")


def generate_data(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Generate one deterministic train/validation data set for every run."""
    total = args.n_train + args.n_val
    initial_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    y0 = torch.rand(
        total, 2, generator=initial_generator, dtype=torch.float32
    ) * 4.5 + 0.5
    y0_device = y0.to(device)

    # Use a finer uniform predictor--corrector solve for the synthetic target,
    # so the training mesh is not being treated as the ground truth.
    reference_func = FixedLotkaVolterraFunc(device)
    reference_tspan = make_tspan(
        args.t_end, args.data_step_size, args.beta, "uniform", device
    )
    with torch.no_grad():
        clean_target = PredictorFDESolverUnscaled.apply(
            reference_func,
            y0_device.to(torch.float64),
            reference_tspan,
            args.beta,
            None,
            None,
            True,
        )

    noise = torch.randn(
        total, 2, generator=noise_generator, dtype=torch.float32
    ).to(device) * args.noise_std
    targets = clean_target.float() + noise
    split = args.n_train
    return y0_device[:split], targets[:split], y0_device[split:], targets[split:]


def save_results(path: str, results: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def train(args: argparse.Namespace) -> Dict:
    if not torch.cuda.is_available():
        raise RuntimeError("This training experiment is intended to run on a CUDA GPU.")

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.save, exist_ok=True)
    y0_train, target_train, y0_val, target_val = generate_data(args, device)

    torch.manual_seed(args.seed)
    model = LotkaVolterraFunc().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    true_params = TRUE_PARAMS.to(device)

    results = {
        "experiment": "lotka_volterra_mesh_comparison",
        "mesh": args.mesh,
        "precision": args.precision,
        "beta": args.beta,
        "t_end": args.t_end,
        "step_size": args.step_size,
        "data_step_size": args.data_step_size,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "noise_std": args.noise_std,
        "learning_rate": args.lr,
        "seed": args.seed,
        "true_params": TRUE_PARAMS.tolist(),
        "initial_params": model.params.detach().cpu().tolist(),
        "data_checksums": {
            "y0_sum": float((y0_train.sum() + y0_val.sum()).item()),
            "target_sum": float((target_train.sum() + target_val.sum()).item()),
        },
        "iterations": [],
    }

    print(
        f"Device: {device} | mesh={args.mesh} | precision={args.precision} "
        f"| beta={args.beta} | T={args.t_end}"
    )
    print(f"{'Iter':>6} {'Train loss':>13} {'Val loss':>13} "
          f"{'Param err':>12} {'Peak MB':>10} {'Time (s)':>10}")
    print("-" * 72)

    for iteration in range(1, args.niters + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        prediction = solve_predictor_corrector(
            model, y0_train, args.beta, args.t_end, args.step_size,
            args.mesh, args.precision
        )
        train_loss = criterion(prediction, target_train)
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

        if iteration % args.log_freq == 0 or iteration == 1 or iteration == args.niters:
            with torch.no_grad():
                val_prediction = solve_predictor_corrector(
                    model, y0_val, args.beta, args.t_end, args.step_size,
                    args.mesh, args.precision
                )
                val_loss = criterion(val_prediction, target_val)
                learned = model.params.detach()
                param_err = (learned - true_params).abs().mean()

            record = {
                "iter": iteration,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "params": learned.cpu().tolist(),
                "param_err": float(param_err.item()),
                "peak_mem_mb": float(peak_mb),
                "iter_time_s": float(elapsed),
            }
            results["iterations"].append(record)
            save_results(os.path.join(args.save, "results.json"), results)
            print(
                f"{iteration:6d} {record['train_loss']:13.6e} "
                f"{record['val_loss']:13.6e} {record['param_err']:12.6e} "
                f"{record['peak_mem_mb']:10.1f} {record['iter_time_s']:10.3f}"
            )

    final = results["iterations"][-1]
    print(f"Final parameters: {final['params']}")
    print(f"Results saved to {args.save}/results.json")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mesh", choices=["uniform", "graded"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--niters", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=25)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=25)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--t_end", type=float, default=T_END)
    parser.add_argument("--step_size", type=float, default=STEP_SIZE)
    parser.add_argument("--data_step_size", type=float, default=DATA_STEP_SIZE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
