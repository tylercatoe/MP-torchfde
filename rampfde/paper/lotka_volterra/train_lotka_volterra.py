#!/usr/bin/env python
"""
Fractional Lotka-Volterra parameter estimation: torchfde FP32 vs rampde FP16.

Replicates Section 5.1 of "Efficient Training of Neural FDE via Adjoint
Backpropagation" (Kang et al., AAAI 2025, arXiv:2503.16666).

True system:  D^β x = x(a - cy)
              D^β y = -y(b - dx)
True params:  [a, b, c, d] = [1.0, 0.5, 1.0, 0.3]
β = 0.7 (we use 0.7 instead of their unspecified value)

Task: Given noisy trajectory data, learn [a, b, c, d] by fitting the FDE.
Both solvers see identical data and optimizer. We compare:
  - Final parameter error vs ground truth
  - Peak GPU memory used during training
  - Training loss convergence

Usage:
    python train_lotka_volterra.py --solver torchfde_fp32
    python train_lotka_volterra.py --solver rampde_fp16
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

_RAMPDE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TORCHFDE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "MP-torchfde"))
for _p in [_RAMPDE_DIR, _TORCHFDE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from torchfde import fdeint as _torchfde_fdeint
    TORCHFDE_OK = True
except ImportError:
    TORCHFDE_OK = False

try:
    from rampde import fdeint as _rampde_fdeint
    RAMPDE_OK = True
except ImportError:
    RAMPDE_OK = False


# ---------------------------------------------------------------------------
# True system and data generation
# ---------------------------------------------------------------------------

TRUE_PARAMS = torch.tensor([1.0, 0.5, 1.0, 0.3])  # [a, b, c, d]
BETA_TRUE   = 0.7
T_END       = 5.0
STEP_SIZE   = 0.1


def lotka_volterra_rhs(params, t, y):
    """D^β [x, y] = [x(a - cy),  -y(b - dx)]"""
    a, b, c, d = params
    x, yy = y[..., 0], y[..., 1]
    dxdt =  x * (a - c * yy)
    dydt = -yy * (b - d * x)
    return torch.stack([dxdt, dydt], dim=-1)


def generate_data(n_traj: int = 50, noise_std: float = 0.05,
                  seed: int = 0) -> tuple:
    """
    Generate noisy fractional Lotka-Volterra trajectories using torchfde FP32.
    Returns (y0s, y_targets) each of shape (n_traj, 2).
    """
    torch.manual_seed(seed)
    # Sample initial conditions from [0.5, 5]^2
    y0s = torch.rand(n_traj, 2) * 4.5 + 0.5

    true_params = TRUE_PARAMS

    class TrueFunc(nn.Module):
        def forward(self, t, y):
            return lotka_volterra_rhs(true_params, t, y)

    func = TrueFunc()
    beta = torch.tensor(BETA_TRUE)

    with torch.no_grad():
        y_T = _torchfde_fdeint(func, y0s, beta, t=T_END,
                               step_size=STEP_SIZE, method="l1")
    noise = torch.randn_like(y_T) * noise_std
    return y0s, (y_T + noise).detach()


# ---------------------------------------------------------------------------
# Learnable FDE model
# ---------------------------------------------------------------------------

class FractionalLVFunc(nn.Module):
    """Learnable Lotka-Volterra RHS with 4 free parameters."""

    def __init__(self, init: torch.Tensor = None):
        super().__init__()
        if init is None:
            init = torch.tensor([0.99, 0.48, 1.05, 0.33])  # near true values
        self.log_params = nn.Parameter(torch.log(init.clamp(min=1e-3)))

    @property
    def params(self):
        return self.log_params.exp()

    def forward(self, t, y):
        return lotka_volterra_rhs(self.params, t, y)


def solve_torchfde(func, y0, beta):
    return _torchfde_fdeint(func, y0, beta, t=T_END, step_size=STEP_SIZE, method="l1")


def solve_rampde(func, y0, beta):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = _rampde_fdeint(func, y0, beta, T_END, STEP_SIZE,
                             loss_scaler=False, adj_dtype=torch.float16)
    return out.float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.save, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Solver: {args.solver}  |  β={BETA_TRUE}  T={T_END}")

    # Generate data (always uses torchfde FP32 for reproducibility)
    print("Generating ground-truth trajectories...")
    y0s, y_targets = generate_data(n_traj=args.n_traj, noise_std=args.noise_std,
                                   seed=args.seed)
    y0s = y0s.to(device)
    y_targets = y_targets.to(device)

    # Model
    torch.manual_seed(args.seed)
    func = FractionalLVFunc().to(device)
    beta = BETA_TRUE  # scalar — rampde fdeint accepts float beta

    solve_fn = solve_torchfde if args.solver == "torchfde_fp32" else solve_rampde

    optimizer = torch.optim.Adam(func.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    results = {
        "solver": args.solver, "beta": BETA_TRUE, "T": T_END,
        "true_params": TRUE_PARAMS.tolist(), "iterations": [],
    }

    print(f"{'Iter':>6}  {'Loss':>10}  {'a':>7}  {'b':>7}  {'c':>7}  {'d':>7}  "
          f"{'param_err':>10}  {'peak_MB':>8}")
    print("-" * 75)

    torch.cuda.reset_peak_memory_stats(device)

    for it in range(1, args.niters + 1):
        optimizer.zero_grad()

        # Batch over trajectories
        y_pred = solve_fn(func, y0s, beta)
        loss = criterion(y_pred, y_targets)
        loss.backward()
        optimizer.step()

        if it % args.log_freq == 0 or it == 1:
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                learned = func.params.cpu()
                param_err = (learned - TRUE_PARAMS).abs().mean().item()

            rec = {
                "iter": it, "loss": loss.item(),
                "params": learned.tolist(), "param_err": param_err,
                "peak_mem_mb": peak_mem,
            }
            results["iterations"].append(rec)

            print(f"{it:6d}  {loss.item():10.6f}  "
                  f"{learned[0].item():7.4f}  {learned[1].item():7.4f}  "
                  f"{learned[2].item():7.4f}  {learned[3].item():7.4f}  "
                  f"{param_err:10.6f}  {peak_mem:8.1f}")

            with open(os.path.join(args.save, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

    final_params = func.params.detach().cpu()
    final_err = (final_params - TRUE_PARAMS).abs().mean().item()
    print(f"\nFinal params: {final_params.tolist()}")
    print(f"True  params: {TRUE_PARAMS.tolist()}")
    print(f"Mean abs error: {final_err:.6f}")
    print(f"Results saved to {args.save}/results.json")
    return results


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--solver",    type=str,   default="rampde_fp16",
                   choices=["torchfde_fp32", "rampde_fp16"])
    p.add_argument("--n_traj",    type=int,   default=50,
                   help="Number of trajectories in training set")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--niters",    type=int,   default=500)
    p.add_argument("--log_freq",  type=int,   default=50)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--gpu",       type=int,   default=0)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--save",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save is None:
        args.save = os.path.join("results", args.solver)
    train(args)
