import os
import sys
import unittest
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torchfde import fdeint, fdeint_adjoint, DynamicScaler


class NonlinearFractionalODE(nn.Module):
    def __init__(self, in_dim=3, hidden=12, seed=0, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, in_dim, dtype=dtype),
        )

    def forward(self, t, y):
        return self.net(y)


def _flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


class TestFractionalAdjointConsistency(unittest.TestCase):
    def setUp(self):
        self.seed = 7
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.dtype = torch.float32
        self.device = "cpu"
        self.beta = torch.tensor(0.8, dtype=self.dtype, device=self.device)  # fractional setting
        self.t = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        self.step_size = torch.tensor(0.05, dtype=self.dtype, device=self.device)

    def test_fractional_order_changes_solution(self):
        model = NonlinearFractionalODE(dtype=self.dtype, seed=self.seed).to(self.device)
        y0 = torch.randn(4, 3, dtype=self.dtype, device=self.device)

        y_beta_1 = fdeint(
            model, y0, beta=torch.tensor(1.0, dtype=self.dtype), t=self.t, step_size=self.step_size, method="predictor"
        )
        y_beta_frac = fdeint(
            model, y0, beta=self.beta, t=self.t, step_size=self.step_size, method="predictor"
        )

        delta = torch.norm(y_beta_1 - y_beta_frac).item()
        self.assertGreater(delta, 1e-5, "beta<1 should produce a different trajectory than beta=1 in this setup")

    def test_adjoint_matches_direct_gradients_fractional_predictor(self):
        base_model = NonlinearFractionalODE(dtype=self.dtype, seed=self.seed).to(self.device)
        y0 = torch.randn(5, 3, dtype=self.dtype, device=self.device)

        # Direct autograd through forward solver
        direct_model = deepcopy(base_model)
        y_direct = y0.clone().requires_grad_(True)
        out_direct = fdeint(
            direct_model,
            y_direct,
            beta=self.beta,
            t=self.t,
            step_size=self.step_size,
            method="predictor",
        )
        loss_direct = out_direct.pow(2).mean()
        loss_direct.backward()
        direct_grads = [y_direct.grad.detach().clone()] + [p.grad.detach().clone() for p in direct_model.parameters()]

        # Adjoint gradients
        adj_model = deepcopy(base_model)
        y_adj = y0.clone().requires_grad_(True)
        out_adj = fdeint_adjoint(
            adj_model,
            y_adj,
            beta=self.beta,
            t=self.t,
            step_size=self.step_size,
            method="predictor-f",
            loss_scaler=False,
        )
        loss_adj = out_adj.pow(2).mean()
        loss_adj.backward()
        adj_grads = [y_adj.grad.detach().clone()] + [p.grad.detach().clone() for p in adj_model.parameters()]

        # Forward closeness
        self.assertTrue(torch.allclose(out_direct, out_adj, rtol=1e-5, atol=1e-6))

        # Gradient closeness (adjoint approximation tolerance)
        for i, (g_ref, g_adj) in enumerate(zip(direct_grads, adj_grads)):
            rel_err = torch.norm(g_ref - g_adj) / (torch.norm(g_ref) + 1e-12)
            self.assertLess(
                rel_err.item(),
                0.10,
                msg=f"Gradient rel error too large at slot {i}: {rel_err.item():.4f}",
            )

    def test_dynamic_scaler_float32_matches_unscaled(self):
        base_model = NonlinearFractionalODE(dtype=self.dtype, seed=self.seed).to(self.device)
        y0 = torch.randn(5, 3, dtype=self.dtype, device=self.device)

        # Unscaled reference
        m_unscaled = deepcopy(base_model)
        y_unscaled = y0.clone().requires_grad_(True)
        out_unscaled = fdeint_adjoint(
            m_unscaled,
            y_unscaled,
            beta=self.beta,
            t=self.t,
            step_size=self.step_size,
            method="predictor-f",
            loss_scaler=False,
        )
        out_unscaled.pow(2).mean().backward()
        grads_unscaled = [y_unscaled.grad.detach().clone()] + [p.grad.detach().clone() for p in m_unscaled.parameters()]

        # Dynamic-scaler run
        m_dynamic = deepcopy(base_model)
        y_dynamic = y0.clone().requires_grad_(True)
        scaler = DynamicScaler(dtype_low=torch.float32)
        out_dynamic = fdeint_adjoint(
            m_dynamic,
            y_dynamic,
            beta=self.beta,
            t=self.t,
            step_size=self.step_size,
            method="predictor-f",
            loss_scaler=scaler,
        )
        out_dynamic.pow(2).mean().backward()
        grads_dynamic = [y_dynamic.grad.detach().clone()] + [p.grad.detach().clone() for p in m_dynamic.parameters()]

        self.assertTrue(torch.allclose(out_unscaled, out_dynamic, rtol=1e-6, atol=1e-7))
        for g_ref, g_dyn in zip(grads_unscaled, grads_dynamic):
            self.assertTrue(torch.allclose(g_ref, g_dyn, rtol=1e-6, atol=1e-7))
        self.assertGreater(len(scaler.scale_history), 0)


if __name__ == "__main__":
    unittest.main()
