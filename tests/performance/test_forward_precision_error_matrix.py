import os
import sys
import unittest
from statistics import geometric_mean
from typing import Dict, List, Tuple

import torch

# Add repo root and local performance directory for imports.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERF_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PERF_DIR)

import forward_precision_error_matrix as fpe


QUIET = os.environ.get("TORCHFDE_TEST_QUIET", "0") == "1"


def _is_finite_scalar(x: float) -> bool:
    return bool(torch.isfinite(torch.tensor(x)).item())


def _nonnull_ratios(values: List[float]) -> List[float]:
    return [v for v in values if v == v]


def _finite_positive(values: List[float]) -> List[float]:
    return [v for v in values if _is_finite_scalar(v) and v > 0.0]


class TestForwardPrecisionErrorMatrix(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = 0.73
        self.t_final = 1.2
        self.method = "predictor"
        self.dim = 6
        self.step_sizes = [0.005, 0.0025]
        self.seeds = [2026, 2027]

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def _build_problem(
        self,
        seed: int,
        beta: float,
        dim: int,
        device: torch.device,
    ) -> Tuple[fpe.ManufacturedFractionalRHS, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        A = (torch.randn(dim, dim, generator=g, dtype=torch.float64) * 0.12) - 0.55 * torch.eye(dim, dtype=torch.float64)

        c0 = 0.2 + 0.6 * torch.rand(dim, generator=g, dtype=torch.float64)
        c1 = -0.15 + 0.3 * torch.rand(dim, generator=g, dtype=torch.float64)
        c2 = 0.1 * torch.randn(dim, generator=g, dtype=torch.float64)
        c3 = 0.06 * torch.randn(dim, generator=g, dtype=torch.float64)

        rhs = fpe.ManufacturedFractionalRHS(A=A, c0=c0, c1=c1, c2=c2, c3=c3, beta=beta).to(
            device=device, dtype=torch.float64
        )
        y0 = c0.clone().to(device=device, dtype=torch.float64)
        return rhs, y0

    def _run_family(
        self,
        low_dtype: torch.dtype,
        low_label: str,
    ) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []

        for seed in self.seeds:
            for step_size in self.step_sizes:
                fpe.set_seed(seed)
                rhs_template, y0_template = self._build_problem(
                    seed=seed,
                    beta=self.beta,
                    dim=self.dim,
                    device=self.device,
                )

                fp32 = fpe.run_case(
                    label="float32_baseline",
                    rhs_template=rhs_template,
                    y0_template=y0_template,
                    beta=self.beta,
                    t_final=self.t_final,
                    step_size=step_size,
                    method=self.method,
                    device=self.device,
                    solve_dtype=torch.float32,
                    mode="direct",
                )
                mixed = fpe.run_case(
                    label=f"{low_label}_mixed",
                    rhs_template=rhs_template,
                    y0_template=y0_template,
                    beta=self.beta,
                    t_final=self.t_final,
                    step_size=step_size,
                    method=self.method,
                    device=self.device,
                    solve_dtype=torch.float32,
                    mode="mixed",
                    low_dtype=low_dtype,
                )
                low_only = fpe.run_case(
                    label=f"{low_label}_low_only",
                    rhs_template=rhs_template,
                    y0_template=y0_template,
                    beta=self.beta,
                    t_final=self.t_final,
                    step_size=step_size,
                    method=self.method,
                    device=self.device,
                    solve_dtype=low_dtype,
                    mode="low-only",
                )

                low_only_finite = all(
                    [
                        _is_finite_scalar(low_only.max_abs_err),
                        _is_finite_scalar(low_only.mean_abs_err),
                        _is_finite_scalar(low_only.l2_abs_err),
                    ]
                )

                # Basic sanity and stability checks.
                for name, row in [("fp32", fp32), ("mixed", mixed), ("low_only", low_only)]:
                    with self.subTest(seed=seed, step_size=step_size, family=low_label, row=name):
                        row_is_finite = all(
                            [
                                _is_finite_scalar(row.max_abs_err),
                                _is_finite_scalar(row.mean_abs_err),
                                _is_finite_scalar(row.l2_abs_err),
                            ]
                        )

                        # Keep strict guarantees for reference and mixed modes.
                        if name in ("fp32", "mixed"):
                            self.assertTrue(row_is_finite)
                            self.assertGreaterEqual(row.max_abs_err, 0.0)
                            self.assertGreaterEqual(row.mean_abs_err, 0.0)
                            self.assertGreaterEqual(row.l2_abs_err, 0.0)
                        else:
                            # For low-only, non-finite indicates precision instability and is tracked.
                            if row_is_finite:
                                self.assertGreaterEqual(row.max_abs_err, 0.0)
                                self.assertGreaterEqual(row.mean_abs_err, 0.0)
                                self.assertGreaterEqual(row.l2_abs_err, 0.0)

                ratio_low_over_mixed_mean = float("nan")
                ratio_low_over_mixed_max = float("nan")
                if mixed.mean_abs_err > 0.0:
                    if low_only_finite:
                        ratio_low_over_mixed_mean = float(low_only.mean_abs_err / mixed.mean_abs_err)
                    else:
                        ratio_low_over_mixed_mean = float("inf")
                if mixed.max_abs_err > 0.0:
                    if low_only_finite:
                        ratio_low_over_mixed_max = float(low_only.max_abs_err / mixed.max_abs_err)
                    else:
                        ratio_low_over_mixed_max = float("inf")

                rows.append(
                    {
                        "seed": float(seed),
                        "step_size": float(step_size),
                        "fp32_mean": float(fp32.mean_abs_err),
                        "mixed_mean": float(mixed.mean_abs_err),
                        "low_mean": float(low_only.mean_abs_err),
                        "fp32_max": float(fp32.max_abs_err),
                        "mixed_max": float(mixed.max_abs_err),
                        "low_max": float(low_only.max_abs_err),
                        "low_finite": 1.0 if low_only_finite else 0.0,
                        "ratio_low_over_mixed_mean": ratio_low_over_mixed_mean,
                        "ratio_low_over_mixed_max": ratio_low_over_mixed_max,
                    }
                )

        return rows

    def test_bf16_mixed_vs_low_only_across_steps_and_seeds(self):
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this GPU")

        rows = self._run_family(torch.bfloat16, "bf16")

        # We intentionally test multiple seeds/step sizes.
        self.assertEqual(len(rows), len(self.seeds) * len(self.step_sizes))

        ratios_mean_all = [r["ratio_low_over_mixed_mean"] for r in rows]
        ratios_max_all = [r["ratio_low_over_mixed_max"] for r in rows]
        ratios_mean_nonnull = _nonnull_ratios(ratios_mean_all)
        ratios_max_nonnull = _nonnull_ratios(ratios_max_all)
        ratios_mean_finite = _finite_positive(ratios_mean_all)
        ratios_max_finite = _finite_positive(ratios_max_all)
        low_only_nonfinite_count = sum(1 for r in rows if r["low_finite"] == 0.0)

        # Mixed should provide a meaningful accuracy benefit over low-only
        # for at least one case, and not be catastrophically worse overall.
        self.assertGreaterEqual(len(ratios_mean_nonnull), 1)
        self.assertGreaterEqual(len(ratios_max_nonnull), 1)
        self.assertGreater(max(ratios_mean_nonnull), 1.05)
        self.assertGreaterEqual(len(ratios_mean_finite), 1)
        self.assertGreaterEqual(len(ratios_max_finite), 1)
        self.assertGreater(geometric_mean(ratios_mean_finite), 0.50)
        self.assertGreater(geometric_mean(ratios_max_finite), 0.50)

        if not QUIET:
            print("\n[bf16] low_only / mixed mean-error ratios:", [f"{x:.3f}" for x in ratios_mean_nonnull])
            print("[bf16] low_only / mixed max-error ratios:", [f"{x:.3f}" for x in ratios_max_nonnull])
            print(f"[bf16] non-finite low-only cases: {low_only_nonfinite_count}/{len(rows)}")

    def test_fp16_mixed_vs_low_only_across_steps_and_seeds(self):
        if self.device.type != "cuda":
            self.skipTest("fp16 comparison is CUDA-only in this suite")

        rows = self._run_family(torch.float16, "fp16")
        self.assertEqual(len(rows), len(self.seeds) * len(self.step_sizes))

        ratios_mean = _finite_positive([r["ratio_low_over_mixed_mean"] for r in rows])
        ratios_max = _finite_positive([r["ratio_low_over_mixed_max"] for r in rows])
        self.assertGreaterEqual(len(ratios_mean), 1)
        self.assertGreaterEqual(len(ratios_max), 1)

        # For fp16, relative ranking can vary by case; enforce stability bounds
        # and require that mixed improves at least one measured scenario.
        self.assertGreater(max(ratios_mean + ratios_max), 1.01)
        self.assertGreater(min(ratios_mean + ratios_max), 0.10)
        self.assertLess(max(ratios_mean + ratios_max), 10.0)

        if not QUIET:
            print("\n[fp16] low_only / mixed mean-error ratios:", [f"{x:.3f}" for x in ratios_mean])
            print("[fp16] low_only / mixed max-error ratios:", [f"{x:.3f}" for x in ratios_max])


if __name__ == "__main__":
    unittest.main()
