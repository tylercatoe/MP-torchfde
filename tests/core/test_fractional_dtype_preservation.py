import os
import sys
import unittest
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torchfde import fdeint, fdeint_adjoint, DynamicScaler


class SimpleFractionalODE(nn.Module):
    def __init__(self, dim: int, target_dtype: torch.dtype, seed: int = 0):
        super().__init__()
        self.target_dtype = target_dtype
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(dim, dim, dtype=target_dtype) * 0.1)
        self.b = nn.Parameter(torch.zeros(dim, dtype=target_dtype))

    def forward(self, t, y):
        assert y.dtype == self.target_dtype, f"state dtype mismatch: {y.dtype} != {self.target_dtype}"
        return torch.tanh(y @ self.W.t() + self.b)


class TestFractionalDtypePreservation(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _skip_if_unsupported(self, dtype: torch.dtype, device: str):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this GPU")
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            self.skipTest(f"{dtype} on CPU is not covered in this suite")

    def _run_case(self, dtype: torch.dtype, device: str, use_adjoint: bool, loss_scaler=None):
        self._skip_if_unsupported(dtype, device)

        dim = 6
        func = SimpleFractionalODE(dim=dim, target_dtype=dtype, seed=self.seed).to(device)

        y0 = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)
        beta = torch.tensor(0.8, dtype=dtype, device=device)  # fractional order (< 1)
        t = torch.tensor(1.0, dtype=dtype, device=device)
        step_size = torch.tensor(0.05, dtype=dtype, device=device)

        if use_adjoint:
            sol = fdeint_adjoint(
                func, y0, beta=beta, t=t, step_size=step_size, method="predictor-f", loss_scaler=loss_scaler
            )
        else:
            sol = fdeint(
                func, y0, beta=beta, t=t, step_size=step_size, method="predictor"
            )

        self.assertEqual(sol.dtype, dtype, f"output dtype {sol.dtype} != {dtype}")

        loss = sol.sum()
        loss.backward()

        self.assertIsNotNone(y0.grad)
        self.assertEqual(y0.grad.dtype, dtype)
        self.assertIsNotNone(func.W.grad)
        self.assertEqual(func.W.grad.dtype, dtype)

    def test_float32_cpu_direct(self):
        self._run_case(torch.float32, "cpu", use_adjoint=False)

    def test_float64_cpu_direct(self):
        self._run_case(torch.float64, "cpu", use_adjoint=False)

    def test_float32_cpu_adjoint(self):
        self._run_case(torch.float32, "cpu", use_adjoint=True, loss_scaler=False)

    def test_float64_cpu_adjoint(self):
        self._run_case(torch.float64, "cpu", use_adjoint=True, loss_scaler=False)

    def test_float16_cuda_adjoint_safe(self):
        self._run_case(torch.float16, "cuda", use_adjoint=True, loss_scaler=False)

    def test_float16_cuda_adjoint_dynamic(self):
        self._run_case(
            torch.float16,
            "cuda",
            use_adjoint=True,
            loss_scaler=DynamicScaler(dtype_low=torch.float16),
        )

    def test_bfloat16_cuda_adjoint(self):
        self._run_case(torch.bfloat16, "cuda", use_adjoint=True, loss_scaler=False)


if __name__ == "__main__":
    unittest.main()
