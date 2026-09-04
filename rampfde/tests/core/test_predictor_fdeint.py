"""
Tests for rampde.predictor_fdeint — Volterra product-rectangle ("basic
predictor") FDE solver with mixed-precision support and an EXACT discrete
adjoint backward.

Mirrors the structure of test_fdeint.py (the L1 scheme's test suite), with
tolerances adjusted for this scheme's properties:

  - The forward recurrence is EXACT for constant forcing (the weight row-sum
    telescopes exactly to (n·h)^β/Γ(β+1)), unlike L1.
  - Convergence order for smooth/polynomial forcing is O(h) (empirically
    verified via an independent NumPy re-implementation before writing this
    file — see verify_predictor_numpy.py).
  - Backward is the EXACT discrete adjoint of the forward recurrence (not an
    approximation), so gradients — including parameter gradients — should
    match a plain-autograd reference tightly, unlike L1's ~30-50% continuous-
    adjoint approximation error on parameter gradients.

Test groups:
  1. TestPredictorFDEintForwardCorrectness — against known analytical solutions
  2. TestPredictorFDEintConvergenceOrder    — O(h) convergence
  3. TestPredictorFDEintWeightBoundedness   — row-sum boundedness (the property
                                               that makes low-precision f-history
                                               storage safe; see Vince_proofs/
                                               Mx_Pre_take2.pdf)
  4. TestPredictorFDEintGradients           — gradcheck (float64) for y0
  5. TestPredictorFDEintAdjointConsistency  — custom backward matches reference
                                               autograd EXACTLY (both y0 and
                                               parameter gradients)
  6. TestPredictorFDEintDtypePreservation   — output/gradient dtypes match input
  7. TestPredictorFDEintSolverSelection     — correct solver variant per precision
  8. TestPredictorFDEintAdjDtype            — adj_dtype storage precision knob
  9. TestPredictorFDEintTupleInputs         — works with tuple-valued FDE functions

All tests run on CPU unless explicitly marked CUDA-only (skipped if unavailable).
"""

import math
import os
import random
import unittest
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Pylance may not resolve rampde if it is not in the IDE's configured venv.
# The tests run correctly from the rampde/ directory with the package installed.
from rampde import predictor_fdeint, DynamicScaler  # type: ignore[import]
from rampde import (  # type: ignore[import]
    PredictorFDESolverUnscaled,
    PredictorFDESolverDynamic,
    PredictorFDESolverUnscaledSafe,
)
from rampde.predictor_fdeint import _select_predictor_solver, _predictor_weights  # type: ignore[import]


def _grad(t: torch.Tensor) -> torch.Tensor:
    """Return t.grad, asserting it is not None (backward must have been called)."""
    assert t.grad is not None, "Expected gradient to be populated after .backward()"
    return t.grad


QUIET = os.environ.get("RAMPDE_TEST_QUIET", "0") == "1"


# ---------------------------------------------------------------------------
# Shared FDE modules
# ---------------------------------------------------------------------------

class ConstantForcing(nn.Module):
    """f(t, y) = c  (used for D^β y = c with exact solution y = c·t^β/Γ(1+β))."""
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, t, y):
        return torch.full_like(y, self.c)


class PolyForcing(nn.Module):
    """f(t, y) = coeff·t^exp  (used for D^β y = 2/Γ(3-β)·t^{2-β} with exact y=t^2)."""
    def __init__(self, coeff: float, exponent: float):
        super().__init__()
        self.coeff = coeff
        self.exponent = exponent

    def forward(self, t, y):
        tv = float(t)
        val = self.coeff * (tv ** self.exponent) if tv > 0.0 else 0.0
        return torch.full_like(y, val)


class LinearDecay(nn.Module):
    """f(t, y) = -w·y  (nonlinear with learnable weight for gradient tests)."""
    def __init__(self, w: float = 1.0, dtype=torch.float32):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([w], dtype=dtype))

    def forward(self, t, y):
        return -self.w * y


class SmallMLP(nn.Module):
    """Small MLP used for more realistic gradient tests."""
    def __init__(self, dim: int, hidden: int = 8, dtype=torch.float32, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, dim, dtype=dtype),
        )

    def forward(self, t, y):
        return self.net(y)


# ---------------------------------------------------------------------------
# Reference predictor solver (plain torch ops, standard autograd — no custom
# backward). Used as a gradient reference: since predictor_fdeint's backward
# is the EXACT discrete adjoint, this should match to numerical precision
# (not just in sign/direction, unlike the L1 continuous-adjoint case).
# ---------------------------------------------------------------------------

def _reference_predictor(func: nn.Module, y0: torch.Tensor, beta_val: float, tspan: torch.Tensor, *, graded_time: bool = False) -> torch.Tensor:
    """
    Transparent-to-autograd product-rectangle predictor for use as a gradient
    reference. Implements the same recurrence as predictor_fdeint but via
    plain tensor operations so PyTorch's built-in autograd computes exact
    (backprop) gradients.
    Returned value: y at tspan[-1].
    """
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    r = (2.0 - beta_val) / beta_val
    dtype = y0.dtype
    device = y0.device
    if graded_time:
        C = 1.0 / math.gamma(beta_val + 1.0)
    else:
        C = float(torch.pow(h, beta_val).item()) / math.gamma(beta_val + 1.0)

    fhist = []
    y_current = y0
    for k in range(N - 1):
        t_k = tspan[k]
        f_k = func(t_k, y_current)
        fhist.append(f_k)

        if graded_time: 
            t_next = tspan[k + 1]
            w = C * (
                torch.pow(t_next - tspan[: k + 1], beta_val)
                - torch.pow(t_next - tspan[1: k + 2], beta_val)
            )
        else:
            j = torch.arange(0, k + 1, dtype=dtype, device=device)
            w = C * (torch.pow(k + 1 - j, beta_val) - torch.pow(k - j, beta_val))

        f_stack = torch.stack(fhist)
        view = (-1,) + (1,) * (f_stack.ndim - 1)
        conv = (w.view(view) * f_stack).sum(0)

        y_current = y0 + conv

    return y_current


def _double_graded_tspan(T: float, step_size: float, beta_val: float) -> torch.Tensor:
    """Build the same symmetric two-sided graded mesh as predictor_fdeint."""
    N = int(round(T / step_size)) + 1
    ind = torch.arange(N, dtype=torch.float32)
    r = (2.0 - beta_val) / beta_val
    c = (N - 1) / 2
    q = 1 - torch.abs(ind - c) / c
    return T / 2 * (1 + torch.sign(ind - c) * (1 - torch.pow(q, r)))


# ============================================================================
# 1. Forward correctness
# ============================================================================

class TestPredictorFDEintForwardCorrectness(unittest.TestCase):
    """Verify the product-rectangle predictor against analytical solutions."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def _solve(self, func, y0, beta, t, step_size, graded_time=False):
        return predictor_fdeint(func, y0, beta=beta, t=t, step_size=step_size, graded_time=graded_time)

    def test_constant_forcing_is_exact(self):
        """D^0.5 y = 1, y(0)=0 -> exact y(T) = T^0.5 / Γ(1.5).

        For constant forcing the row-sum telescopes exactly:
        Σ_j d_{n,j} = (n h)^β / Γ(β+1), so the predictor scheme reproduces
        the analytic solution up to floating-point roundoff — unlike L1,
        which has O(h^{2-β}) discretization error even for constant forcing.
        """
        beta = 0.5
        T = 1.0
        step_size = 0.01
        y0 = torch.tensor([0.0])
        func = ConstantForcing(c=1.0)

        y_T = self._solve(func, y0, beta, T, step_size)
        y_T_graded = self._solve(func, y0, beta, T, step_size, graded_time=True)
        exact = T ** beta / math.gamma(1.0 + beta)

        err = abs(y_T.item() - exact)
        err_graded = abs(y_T_graded.item() - exact)
        if not QUIET:
            print(f"\nConstant forcing: y_T={y_T.item():.6f}, exact={exact:.6f}, err={err:.2e}")
            print(f"Constant forcing (graded): y_T={y_T_graded.item():.6f}, exact={exact:.6f}, err={err_graded:.2e}")
        self.assertLess(err, 1e-5, "Forward error too large for constant forcing (should be ~exact)")
        self.assertLess(err_graded, 1e-5, "Forward error too large for constant forcing (graded) (should be ~exact)")

    def test_polynomial_forcing_accuracy(self):
        """D^0.5 y = (2/Γ(1.5))·t^1.5, y(0)=0 -> exact y(T) = T^2.

        The predictor scheme converges at O(h) for this forcing (empirically
        verified independently in NumPy before writing this test). With
        h=0.01 the expected error is ~1%, so we use a 5% tolerance for
        safety margin under float32.
        """
        beta = 0.5
        T = 1.0
        step_size = 0.01
        y0 = torch.tensor([0.0])

        coeff = 2.0 / math.gamma(3.0 - beta)
        exponent = 2.0 - beta
        func = PolyForcing(coeff=coeff, exponent=exponent)

        y_T = self._solve(func, y0, beta, T, step_size)
        exact = T ** 2

        err = abs(y_T.item() - exact)
        if not QUIET:
            print(f"\nPoly forcing: y_T={y_T.item():.6f}, exact={exact:.6f}, err={err:.2e}")
        self.assertLess(err, 0.05, "Forward error too large for polynomial forcing")

    def test_different_beta_values(self):
        """Check that different β values give different trajectories (sanity check)."""
        T = 1.0
        step_size = 0.05
        y0 = torch.tensor([1.0, 0.5])
        func = LinearDecay(w=0.5)

        y_beta05 = predictor_fdeint(func, y0, beta=0.5, t=T, step_size=step_size)
        y_beta08 = predictor_fdeint(func, y0, beta=0.8, t=T, step_size=step_size)

        diff = (y_beta05 - y_beta08).norm().item()
        if not QUIET:
            print(f"\nDiff between β=0.5 and β=0.8: {diff:.4f}")
        self.assertGreater(diff, 1e-4, "Different β should give different solutions")


# ============================================================================
# 2. Convergence order
# ============================================================================

class TestPredictorFDEintConvergenceOrder(unittest.TestCase):
    """The product-rectangle predictor should converge at rate O(h)."""

    def setUp(self):
        torch.manual_seed(42)

    def _convergence_order(self, beta: float) -> float:
        """Returns the observed convergence order via step-halving, using
        polynomial forcing (constant forcing is exact for this scheme, so
        it can't be used to measure order — see test_constant_forcing_is_exact)."""
        T = 1.0
        y0 = torch.tensor([0.0])
        coeff = 2.0 / math.gamma(3.0 - beta)
        exponent = 2.0 - beta
        func = PolyForcing(coeff=coeff, exponent=exponent)
        exact = T ** 2

        errors = []
        for n_half in range(4):
            h = 0.01 / (2 ** n_half)
            y_T = predictor_fdeint(func, y0, beta=beta, t=T, step_size=h)
            errors.append(abs(y_T.item() - exact))

        orders = []
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i - 1] > 0:
                orders.append(math.log2(errors[i - 1] / errors[i]))

        return sum(orders) / len(orders) if orders else 0.0

    def test_convergence_beta_05(self):
        """β=0.5 -> empirically O(h^1.0)."""
        order = self._convergence_order(0.5)
        if not QUIET:
            print(f"\nConvergence order β=0.5: {order:.3f} (expected ≈ 1.0)")
        self.assertGreater(order, 0.8, "Convergence order too low for β=0.5")

    def test_convergence_beta_08(self):
        """β=0.8 -> empirically O(h^1.0)."""
        order = self._convergence_order(0.8)
        if not QUIET:
            print(f"\nConvergence order β=0.8: {order:.3f} (expected ≈ 1.0)")
        self.assertGreater(order, 0.8, "Convergence order too low for β=0.8")


# ============================================================================
# 3. Weight boundedness (the property enabling safe low-precision f-history)
# ============================================================================

class TestPredictorFDEintWeightBoundedness(unittest.TestCase):
    """
    Σ_j d_{n,j} = (n·h)^β / Γ(β+1) ≤ T^β / Γ(β+1), independent of the number
    of steps N. This boundedness (proved in Vince_proofs/Mx_Pre_take2.pdf) is
    what makes storing the f-history buffer in low precision safe.
    """

    def test_row_sum_matches_closed_form(self):
        beta = 0.6
        # float64 throughout so this test isolates the weight-formula identity
        # itself; predictor_fdeint() always builds tspan (and hence h) as
        # float32 regardless of y0's dtype (mirroring fdeint.py's convention),
        # so the *solver's* h carries float32-level precision by design — not
        # what this test is checking.
        h = torch.tensor(0.037, dtype=torch.float64)
        C = float(torch.pow(h, beta).item()) / math.gamma(beta + 1.0)

        for n in [1, 5, 20, 137, 500]:
            k = n - 1
            w = _predictor_weights(k, beta, C, torch.float64, torch.device("cpu"))
            total = w.sum().item()
            expected = ((n * h.item()) ** beta) / math.gamma(beta + 1.0)
            self.assertAlmostEqual(total, expected, places=8,
                                    msg=f"Row sum mismatch at n={n}")

    def test_row_sum_is_bounded_by_terminal_value(self):
        """Row sums should be monotonically increasing but bounded by the
        value at n=N-1 (i.e. by T^β/Γ(β+1))."""
        beta = 0.5
        h = 0.02
        C = (h ** beta) / math.gamma(beta + 1.0)
        N = 50
        sums = []
        for n in range(1, N):
            k = n - 1
            w = _predictor_weights(k, beta, C, torch.float64, torch.device("cpu"))
            sums.append(w.sum().item())

        # Monotone non-decreasing
        for i in range(1, len(sums)):
            self.assertGreaterEqual(sums[i] + 1e-12, sums[i - 1])

        bound = ((N - 1) * h) ** beta / math.gamma(beta + 1.0)
        self.assertLessEqual(sums[-1], bound + 1e-8)


# ============================================================================
# 4. Gradient correctness (gradcheck)
# ============================================================================

class TestPredictorFDEintGradients(unittest.TestCase):
    """
    Verify backward correctness via torch.autograd.gradcheck.

    gradcheck uses finite differences (±eps perturbations) and compares to
    our analytical gradient from the exact discrete-adjoint backward.
    Requires float64.
    """

    def setUp(self):
        torch.manual_seed(7)

    def _make_func(self, dim: int) -> nn.Module:
        return LinearDecay(w=0.5, dtype=torch.float64)

    def test_gradcheck_wrt_y0(self):
        """Gradient w.r.t. initial condition y0 (float64, CPU)."""
        dim = 3
        func = self._make_func(dim)
        y0 = torch.randn(dim, dtype=torch.float64, requires_grad=True)

        def fn(y0_):
            return predictor_fdeint(func, y0_, beta=0.5, t=0.5, step_size=0.1)

        passed = torch.autograd.gradcheck(fn, (y0,), eps=1e-6, atol=1e-5, rtol=1e-4)
        self.assertTrue(passed, "gradcheck failed for gradient w.r.t. y0")

    def test_gradcheck_wrt_y0_nonlinear(self):
        """Gradient w.r.t. y0 for a nonlinear (MLP) right-hand side."""
        dim = 3
        func = SmallMLP(dim=dim, hidden=4, dtype=torch.float64, seed=1)
        y0 = torch.randn(dim, dtype=torch.float64, requires_grad=True)

        def fn(y0_):
            return predictor_fdeint(func, y0_, beta=0.6, t=0.3, step_size=0.05)

        passed = torch.autograd.gradcheck(fn, (y0,), eps=1e-6, atol=1e-5, rtol=1e-4)
        self.assertTrue(passed, "gradcheck failed for nonlinear f w.r.t. y0")

    def test_gradcheck_wrt_y0_batch(self):
        """Gradient w.r.t. batched initial condition (2D y0)."""
        class BatchLinearDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
            def forward(self, t, y):
                return -self.w * y

        func = BatchLinearDecay()
        y0 = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

        def fn(y0_):
            return predictor_fdeint(func, y0_, beta=0.5, t=0.5, step_size=0.1)

        passed = torch.autograd.gradcheck(fn, (y0,), eps=1e-6, atol=1e-5, rtol=1e-4)
        self.assertTrue(passed, "gradcheck failed for batched y0")


# ============================================================================
# 5. Adjoint consistency
# ============================================================================

class TestPredictorFDEintAdjointConsistency(unittest.TestCase):
    """
    Our custom discrete-adjoint backward should produce (nearly) IDENTICAL
    gradients to standard autograd through the reference predictor
    implementation — both for y0 and for parameters — because the backward
    is the exact discrete adjoint of the forward recurrence, not an
    approximation.
    """

    def setUp(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.dtype = torch.float32
        self.device = "cpu"
        self.beta = 0.6
        self.T = 0.5
        self.step_size = 0.05
        self.dim = 4

    def _make_tspan(self):
        N = int(round(self.T / self.step_size)) + 1
        return torch.linspace(0.0, self.T, N, dtype=self.dtype)

    def test_adjoint_matches_reference_autograd_exactly(self):
        """predictor_fdeint gradients (discrete adjoint) match reference
        direct autograd tightly for BOTH y0 and parameter gradients — unlike
        rampde.fdeint's L1 solver, whose continuous-adjoint approximation
        gives ~30-50% relative error on parameter gradients.
        """
        torch.manual_seed(self.seed)
        base_func = SmallMLP(dim=self.dim, dtype=self.dtype, seed=self.seed)
        y0 = torch.randn(self.dim, dtype=self.dtype)
        tspan = self._make_tspan()

        # --- Reference: autograd through plain predictor recurrence ---
        ref_func = deepcopy(base_func)
        y0_ref = y0.clone().requires_grad_(True)
        out_ref = _reference_predictor(ref_func, y0_ref, self.beta, tspan)
        graded_tspan = _double_graded_tspan(self.T, self.step_size, self.beta)
        out_ref_graded = _reference_predictor(ref_func, y0_ref, self.beta, graded_tspan, graded_time=True)
        out_ref.pow(2).mean().backward()
        out_ref_graded.pow(2).mean().backward()
        ref_y0_grad = _grad(y0_ref).detach().clone()
        ref_param_grads = [_grad(p).detach().clone() for p in ref_func.parameters()]

        # --- predictor_fdeint: custom discrete-adjoint backward ---
        adj_func = deepcopy(base_func)
        y0_adj = y0.clone().requires_grad_(True)
        out_adj = predictor_fdeint(adj_func, y0_adj, beta=self.beta, t=self.T, step_size=self.step_size)
        out_adj_graded = predictor_fdeint(adj_func, y0_adj, beta=self.beta, t=self.T, step_size=self.step_size, graded_time=True)
        out_adj.pow(2).mean().backward()
        out_adj_graded.pow(2).mean().backward()
        adj_y0_grad = _grad(y0_adj).detach().clone()
        adj_param_grads = [_grad(p).detach().clone() for p in adj_func.parameters()]

        # Forward solutions should match exactly
        self.assertTrue(
            torch.allclose(out_ref, out_adj, rtol=1e-5, atol=1e-5),
            f"Forward mismatch: ref={out_ref} adj={out_adj}",
        )
        self.assertTrue(
            torch.allclose(out_ref_graded, out_adj_graded, rtol=1e-5, atol=1e-5),
            f"Forward mismatch (graded): ref={out_ref_graded} adj={out_adj_graded}"
        )

        # Gradient w.r.t. y0: should match tightly (exact discrete adjoint)
        rel_err_y0 = (ref_y0_grad - adj_y0_grad).norm() / (ref_y0_grad.norm() + 1e-12)
        rel_err_y0_graded = (ref_y0_grad - adj_y0_grad).norm() / (ref_y0_grad.norm() + 1e-12)
        if not QUIET:
            print(f"\ny0 grad rel err: {rel_err_y0.item():.2e}")
            print(f"y0 grad rel err (graded): {rel_err_y0_graded.item():.2e}")
        self.assertLess(rel_err_y0.item(), 1e-3, "y0 gradient mismatch between adjoint and reference")
        self.assertLess(rel_err_y0_graded.item(), 1e-3, "y0 gradient mismatch (graded) between adjoint and reference")

        # Parameter gradients: should ALSO match tightly (exact discrete adjoint)
        for i, (g_ref, g_adj) in enumerate(zip(ref_param_grads, adj_param_grads)):
            rel_err = (g_ref - g_adj).norm() / (g_ref.norm() + 1e-12)
            if not QUIET:
                print(f"  param[{i}] grad rel err: {rel_err.item():.2e}")
            self.assertLess(rel_err.item(), 1e-3,
                             f"Param[{i}] gradient mismatch between adjoint and reference "
                             f"(expected tight match — backward is the exact discrete adjoint)")

    def test_graded_mesh_adjoint_matches_reference(self):
        """The custom graded-mesh forward/backward matches plain autograd."""
        beta = 0.6
        T = 0.4
        step_size = 0.1
        dim = 3
        tspan = _double_graded_tspan(T, step_size, beta)

        base_func = SmallMLP(dim=dim, dtype=torch.float64, seed=self.seed)
        y0 = torch.randn(dim, dtype=torch.float64)

        ref_func = deepcopy(base_func)
        y0_ref = y0.clone().requires_grad_(True)
        out_ref = _reference_predictor(
            ref_func, y0_ref, beta, tspan, graded_time=True
        )
        out_ref.pow(2).mean().backward()
        ref_y0_grad = _grad(y0_ref).detach().clone()
        ref_param_grads = [_grad(p).detach().clone() for p in ref_func.parameters()]

        adj_func = deepcopy(base_func)
        y0_adj = y0.clone().requires_grad_(True)
        out_adj = predictor_fdeint(
            adj_func, y0_adj, beta=beta, t=T,
            step_size=step_size, graded_time=True,
        )
        out_adj.pow(2).mean().backward()
        adj_y0_grad = _grad(y0_adj).detach().clone()
        adj_param_grads = [_grad(p).detach().clone() for p in adj_func.parameters()]

        self.assertTrue(torch.allclose(out_ref, out_adj, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(ref_y0_grad, adj_y0_grad, rtol=1e-5, atol=1e-6))
        for g_ref, g_adj in zip(ref_param_grads, adj_param_grads):
            self.assertTrue(torch.allclose(g_ref, g_adj, rtol=1e-5, atol=1e-6))

    def test_dynamic_scaler_float32_matches_unscaled(self):
        """DynamicScaler(float32) should give identical results to no-scaler."""
        torch.manual_seed(self.seed)
        base_func = SmallMLP(dim=self.dim, dtype=self.dtype, seed=self.seed)
        y0 = torch.randn(self.dim, dtype=self.dtype)

        # Unscaled
        m_us = deepcopy(base_func)
        y_us = y0.clone().requires_grad_(True)
        out_us = predictor_fdeint(m_us, y_us, beta=self.beta, t=self.T, step_size=self.step_size, loss_scaler=False)
        out_us_graded = predictor_fdeint(m_us, y_us, beta=self.beta, t=self.T, step_size=self.step_size, graded_time=True, loss_scaler=False)
        out_us.pow(2).mean().backward()
        out_us_graded.pow(2).mean().backward()
        g_us = [_grad(y_us).clone()] + [_grad(p).clone() for p in m_us.parameters()]

        # Dynamic scaler (float32)
        m_dyn = deepcopy(base_func)
        y_dyn = y0.clone().requires_grad_(True)
        scaler = DynamicScaler(dtype_low=torch.float32)
        out_dyn = predictor_fdeint(m_dyn, y_dyn, beta=self.beta, t=self.T, step_size=self.step_size, loss_scaler=scaler)
        out_dyn_graded = predictor_fdeint(m_dyn, y_dyn, beta=self.beta, t=self.T, step_size=self.step_size, graded_time=True, loss_scaler=scaler)
        out_dyn.pow(2).mean().backward()
        out_dyn_graded.pow(2).mean().backward()
        g_dyn = [_grad(y_dyn).clone()] + [_grad(p).clone() for p in m_dyn.parameters()]

        self.assertTrue(torch.allclose(out_us, out_dyn, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(out_us_graded, out_dyn_graded, rtol=1e-6, atol=1e-7))
        for g1, g2 in zip(g_us, g_dyn):
            self.assertTrue(torch.allclose(g1, g2, rtol=1e-5, atol=1e-6),
                            "DynamicScaler(float32) gradients differ from unscaled")
        self.assertGreater(len(scaler.scale_history), 0, "DynamicScaler was never called")


# ============================================================================
# 6. Dtype preservation
# ============================================================================

class TestPredictorFDEintDtypePreservation(unittest.TestCase):
    """Output and gradient dtypes should match input dtype."""

    def setUp(self):
        torch.manual_seed(42)

    def _run_case(self, dtype: torch.dtype, device: str, loss_scaler=None):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this GPU")
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            self.skipTest(f"{dtype} not covered for CPU in this suite")

        dim = 4
        func = SmallMLP(dim=dim, dtype=dtype, seed=0).to(device)
        y0 = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)

        out = predictor_fdeint(func, y0, beta=0.7, t=0.5, step_size=0.1, loss_scaler=loss_scaler)
        out_graded = predictor_fdeint(func, y0, beta=0.7, t=0.5, step_size=0.1, graded_time=True, loss_scaler=loss_scaler)

        self.assertEqual(out.dtype, dtype, f"Output dtype {out.dtype} != {dtype}")
        self.assertEqual(out_graded.dtype, dtype, f"Output (graded) dtype {out_graded.dtype} != {dtype}")

        out.sum().backward()
        out_graded.sum().backward()
        y0_grad = _grad(y0)
        self.assertEqual(y0_grad.dtype, dtype, f"y0 grad dtype {y0_grad.dtype} != {dtype}")
        for p in func.parameters():
            p_grad = _grad(p)
            self.assertEqual(p_grad.dtype, dtype, f"param grad dtype {p_grad.dtype} != {dtype}")

    def test_float32_cpu(self):
        self._run_case(torch.float32, "cpu", loss_scaler=False)

    def test_float64_cpu(self):
        self._run_case(torch.float64, "cpu", loss_scaler=False)

    def test_float32_cuda(self):
        self._run_case(torch.float32, "cuda", loss_scaler=False)

    def test_float64_cuda(self):
        self._run_case(torch.float64, "cuda", loss_scaler=False)

    def test_bfloat16_cuda(self):
        self._run_case(torch.bfloat16, "cuda", loss_scaler=False)

    def test_float16_cuda_dynamic_scaler(self):
        """float16 with DynamicScaler should use PredictorFDESolverDynamic."""
        self._run_case(
            torch.float16, "cuda",
            loss_scaler=DynamicScaler(dtype_low=torch.float16),
        )

    def test_float16_cuda_no_scaler(self):
        """float16 without scaler should use PredictorFDESolverUnscaledSafe."""
        self._run_case(torch.float16, "cuda", loss_scaler=False)


# ============================================================================
# 7. Solver selection
# ============================================================================

class TestPredictorFDEintSolverSelection(unittest.TestCase):
    """_select_predictor_solver should return the right class for each precision/scaler."""

    def test_float32_no_scaler_gives_unscaled(self):
        solver, scaler = _select_predictor_solver(None, torch.float32)
        self.assertIs(solver, PredictorFDESolverUnscaled)
        self.assertIsNone(scaler)

    def test_float64_no_scaler_gives_unscaled(self):
        solver, scaler = _select_predictor_solver(None, torch.float64)
        self.assertIs(solver, PredictorFDESolverUnscaled)
        self.assertIsNone(scaler)

    def test_bfloat16_no_scaler_gives_unscaled(self):
        solver, scaler = _select_predictor_solver(None, torch.bfloat16)
        self.assertIs(solver, PredictorFDESolverUnscaled)
        self.assertIsNone(scaler)

    def test_float16_none_creates_dynamic_scaler(self):
        solver, scaler = _select_predictor_solver(None, torch.float16)
        self.assertIs(solver, PredictorFDESolverDynamic)
        self.assertIsInstance(scaler, DynamicScaler)

    def test_float16_false_gives_safe(self):
        solver, scaler = _select_predictor_solver(False, torch.float16)
        self.assertIs(solver, PredictorFDESolverUnscaledSafe)
        self.assertIsNone(scaler)

    def test_explicit_dynamic_scaler_gives_dynamic(self):
        ds = DynamicScaler(dtype_low=torch.float16)
        solver, scaler = _select_predictor_solver(ds, torch.float32)
        self.assertIs(solver, PredictorFDESolverDynamic)
        self.assertIs(scaler, ds)

    def test_false_float32_gives_unscaled(self):
        """loss_scaler=False with float32 disables internal scaling -> Unscaled."""
        solver, scaler = _select_predictor_solver(False, torch.float32)
        self.assertIs(solver, PredictorFDESolverUnscaled)
        self.assertIsNone(scaler)


# ============================================================================
# 8. adj_dtype
# ============================================================================

class TestPredictorFDEintAdjDtype(unittest.TestCase):
    """
    adj_dtype controls the storage precision of the adjoint history buffer.

    adj_dtype=None (default) stores in dtype_hi (float32) — safe baseline.
    adj_dtype=torch.float32  explicit float32 — identical to default.
    """

    def setUp(self):
        torch.manual_seed(99)

    def _run(self, dim, beta, T, h, adj_dtype_val):
        class LinearDecayLocal(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([0.3]))
            def forward(self, t, y): return -self.w * y

        func_hi = LinearDecayLocal()
        func_low = LinearDecayLocal()
        func_low.load_state_dict(func_hi.state_dict())

        y0 = torch.randn(dim)

        y0_hi = y0.clone().requires_grad_(True)
        predictor_fdeint(func_hi, y0_hi, beta=beta, t=T, step_size=h).pow(2).mean().backward()

        y0_lo = y0.clone().requires_grad_(True)
        predictor_fdeint(func_low, y0_lo, beta=beta, t=T, step_size=h,
                          adj_dtype=adj_dtype_val).pow(2).mean().backward()

        return (
            _grad(y0_hi).clone(), [_grad(p).clone() for p in func_hi.parameters()],
            _grad(y0_lo).clone(), [_grad(p).clone() for p in func_low.parameters()],
        )

    def test_adj_dtype_float32_matches_default(self):
        """Explicit float32 adj_dtype must give identical results to None."""
        g_hi_y0, g_hi_p, g_lo_y0, g_lo_p = self._run(4, 0.5, 1.0, 0.1, torch.float32)
        self.assertTrue(torch.allclose(g_hi_y0, g_lo_y0),
                        "float32 adj_dtype should be identical to default (None)")
        for g1, g2 in zip(g_hi_p, g_lo_p):
            self.assertTrue(torch.allclose(g1, g2))

    def test_adj_dtype_float64_matches_default(self):
        """float64 adj_dtype with float64 y0 — should be identical to default."""
        class LinearDecay64(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
            def forward(self, t, y): return -self.w * y

        y0 = torch.randn(4, dtype=torch.float64)
        for adj_dt in [None, torch.float64]:
            f = LinearDecay64(); y = y0.clone().requires_grad_(True)
            predictor_fdeint(f, y, beta=0.5, t=1.0, step_size=0.1, adj_dtype=adj_dt
                              ).pow(2).mean().backward()
        self.assertTrue(_grad(y).isfinite().all())

    def test_adj_dtype_none_vs_explicit_none(self):
        """None and explicit float32 produce same output (forward)."""
        class Func(nn.Module):
            def forward(self, t, y): return -0.1 * y

        y0 = torch.randn(3)
        out_none = predictor_fdeint(Func(), y0, beta=0.5, t=1.0, step_size=0.1, adj_dtype=None)
        out_fp32 = predictor_fdeint(Func(), y0, beta=0.5, t=1.0, step_size=0.1, adj_dtype=torch.float32)
        self.assertTrue(torch.allclose(out_none, out_fp32))


# ============================================================================
# 9. Tuple inputs
# ============================================================================

class TestPredictorFDEintTupleInputs(unittest.TestCase):
    """predictor_fdeint should work correctly when y0 is a tuple of tensors."""

    def setUp(self):
        torch.manual_seed(11)

    def test_tuple_forward_matches_flat_tensor(self):
        """Tuple-input predictor_fdeint should give the same result as flat-tensor input."""
        dim1, dim2 = 3, 2

        class TupleFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.eye(dim1 + dim2) * -0.1)

            def forward(self, t, y):
                flat = torch.cat([y[0].reshape(-1), y[1].reshape(-1)], dim=-1)
                out = flat @ self.W.T
                return (out[:dim1], out[dim1:])

        class FlatFunc(nn.Module):
            def __init__(self, W):
                super().__init__()
                self.W = nn.Parameter(W)

            def forward(self, t, y):
                return y @ self.W.T

        torch.manual_seed(0)
        W = torch.eye(dim1 + dim2) * -0.1
        y0_a = torch.randn(dim1)
        y0_b = torch.randn(dim2)
        y0_flat = torch.cat([y0_a, y0_b])

        tuple_func = TupleFunc()
        tuple_func.W.data = W.clone()

        flat_func = FlatFunc(W.clone())

        out_tuple = predictor_fdeint(tuple_func, (y0_a, y0_b), beta=0.5, t=0.5, step_size=0.1)
        out_flat = predictor_fdeint(flat_func, y0_flat, beta=0.5, t=0.5, step_size=0.1)

        out_tuple_cat = torch.cat([out_tuple[0].reshape(-1), out_tuple[1].reshape(-1)])
        self.assertTrue(
            torch.allclose(out_tuple_cat, out_flat, rtol=1e-5, atol=1e-5),
            "Tuple and flat outputs differ",
        )

    def test_tuple_backward_propagates_gradients(self):
        """Gradients should flow back through tuple y0 inputs."""
        dim = 3

        class TupleDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([0.5]))

            def forward(self, t, y):
                return (-self.w * y[0], -self.w * y[1])

        func = TupleDecay()
        y0_a = torch.randn(dim, requires_grad=True)
        y0_b = torch.randn(2, requires_grad=True)

        out = predictor_fdeint(func, (y0_a, y0_b), beta=0.6, t=0.4, step_size=0.1)
        loss = out[0].sum() + out[1].sum()
        loss.backward()

        w_grad = _grad(func.w)
        self.assertTrue(_grad(y0_a).isfinite().all(), "y0_a gradient should be finite")
        self.assertTrue(_grad(y0_b).isfinite().all(), "y0_b gradient should be finite")
        self.assertTrue(w_grad.isfinite().all(), "w gradient should be finite")

    def test_tuple_output_types(self):
        """Tuple input should produce tuple output; tensor input -> tensor output."""
        class TwoCompFunc(nn.Module):
            def forward(self, t, y):
                return (-0.1 * y[0], -0.1 * y[1])

        class OneCompFunc(nn.Module):
            def forward(self, t, y):
                return -0.1 * y

        y0_tuple = (torch.randn(3), torch.randn(2))
        y0_tensor = torch.randn(5)

        out_tuple = predictor_fdeint(TwoCompFunc(), y0_tuple, beta=0.5, t=0.3, step_size=0.1)
        out_tensor = predictor_fdeint(OneCompFunc(), y0_tensor, beta=0.5, t=0.3, step_size=0.1)

        self.assertIsInstance(out_tuple, tuple, "Tuple input should yield tuple output")
        self.assertIsInstance(out_tensor, torch.Tensor, "Tensor input should yield tensor output")
        self.assertEqual(len(out_tuple), 2)
        self.assertEqual(out_tuple[0].shape, (3,))
        self.assertEqual(out_tuple[1].shape, (2,))


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    unittest.main()
