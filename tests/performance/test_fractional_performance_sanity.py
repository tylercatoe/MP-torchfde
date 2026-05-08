#!/usr/bin/env python3
"""Performance sanity checks for fractional adjoint mixed precision."""

import os
import sys
import time
import unittest
from dataclasses import dataclass
from statistics import mean, pstdev

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torchfde import DynamicScaler, fdeint_adjoint
from torchfde.fdeadjoint import (
    FDEAdjointMethodDynamic,
    FDEAdjointMethodUnscaled,
    FDEAdjointMethodUnscaledSafe,
    _select_adjoint_solver,
)

QUIET = os.environ.get("TORCHFDE_TEST_QUIET", "0") == "1"


class FractionalVectorField(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim, dtype=dtype)
        self.lin2 = nn.Linear(dim, dim, dtype=dtype)

    def forward(self, t, y):
        return torch.tanh(self.lin2(torch.tanh(self.lin1(y))))


@dataclass
class PerfConfig:
    name: str
    dtype: torch.dtype
    scaler_factory: callable
    expected_solver: type


class TestFractionalPerformanceSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping performance sanity tests")

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        cls.device = "cuda"
        cls.dim = 128
        cls.batch = 64
        cls.beta_value = 0.8
        cls.t_final = 1.0
        cls.step = 0.05
        cls.warmup = 2
        cls.runs = 4

    def _build_configs(self):
        configs = [
            PerfConfig(
                name="float32_unscaled",
                dtype=torch.float32,
                scaler_factory=lambda: False,
                expected_solver=FDEAdjointMethodUnscaled,
            ),
            PerfConfig(
                name="float16_safe",
                dtype=torch.float16,
                scaler_factory=lambda: False,
                expected_solver=FDEAdjointMethodUnscaledSafe,
            ),
            PerfConfig(
                name="float16_dynamic",
                dtype=torch.float16,
                scaler_factory=lambda: DynamicScaler(dtype_low=torch.float16),
                expected_solver=FDEAdjointMethodDynamic,
            ),
        ]

        if torch.cuda.is_bf16_supported():
            configs.insert(
                1,
                PerfConfig(
                    name="bfloat16_unscaled",
                    dtype=torch.bfloat16,
                    scaler_factory=lambda: False,
                    expected_solver=FDEAdjointMethodUnscaled,
                ),
            )

        return configs

    def _run_config(self, cfg: PerfConfig):
        model = FractionalVectorField(dim=self.dim, dtype=cfg.dtype).to(self.device)
        beta = torch.tensor(self.beta_value, dtype=cfg.dtype, device=self.device)
        t = torch.tensor(self.t_final, dtype=cfg.dtype, device=self.device)
        step_size = torch.tensor(self.step, dtype=cfg.dtype, device=self.device)

        scaler = cfg.scaler_factory()
        solver, _ = _select_adjoint_solver(scaler, cfg.dtype)
        self.assertEqual(
            solver,
            cfg.expected_solver,
            msg=f"wrong solver for {cfg.name}: got {solver.__name__}",
        )

        times = []
        for i in range(self.warmup + self.runs):
            model.zero_grad(set_to_none=True)
            y0 = torch.randn(self.batch, self.dim, dtype=cfg.dtype, device=self.device, requires_grad=True)

            torch.cuda.synchronize()
            start = time.perf_counter()
            sol = fdeint_adjoint(
                model,
                y0,
                beta=beta,
                t=t,
                step_size=step_size,
                method="predictor-f",
                loss_scaler=scaler,
            )
            loss = sol.square().mean()
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            self.assertEqual(sol.dtype, cfg.dtype)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertIsNotNone(y0.grad)
            self.assertEqual(y0.grad.dtype, cfg.dtype)
            self.assertTrue(torch.isfinite(y0.grad).all().item())

            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all().item())

            if i >= self.warmup:
                times.append(elapsed)

        if isinstance(scaler, DynamicScaler):
            self.assertGreater(len(scaler.scale_history), 0, "dynamic scaler never initialized")

        return {
            "name": cfg.name,
            "dtype": str(cfg.dtype).split(".")[-1],
            "solver": solver.__name__,
            "mean_s": mean(times),
            "std_s": pstdev(times) if len(times) > 1 else 0.0,
        }

    def test_fractional_adjoint_performance_sanity(self):
        results = []
        for cfg in self._build_configs():
            results.append(self._run_config(cfg))

        for row in results:
            self.assertGreater(row["mean_s"], 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(row["mean_s"])).item())
            self.assertTrue(torch.isfinite(torch.tensor(row["std_s"])).item())

        if not QUIET:
            print("\nFractional adjoint performance sanity summary")
            print("-" * 78)
            print(f"{'config':<20} {'dtype':<10} {'solver':<30} {'mean(s)':>8} {'std(s)':>8}")
            print("-" * 78)
            for row in results:
                print(
                    f"{row['name']:<20} {row['dtype']:<10} {row['solver']:<30} "
                    f"{row['mean_s']:>8.4f} {row['std_s']:>8.4f}"
                )
            print("-" * 78)


if __name__ == "__main__":
    unittest.main()
