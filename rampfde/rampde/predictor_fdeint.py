"""
Mixed-precision Volterra product-rectangle ("basic predictor") FDE solver for rampde.

Implements the one-step Adams-Bashforth / product-rectangle predictor for Caputo
fractional differential equations (Diethelm et al. 2004; referred to as the
"basic predictor" in Kang et al. 2024):

    D^β y(t) = f(t, y),  y(0) = y0,  β ∈ (0, 1)

Update formula:

    U^n = U^0 + Σ_{j=0}^{n-1} d_{n,j} · f(t_j, U^j)

    d_{n,j} = h^β / Γ(β+1) · [ (n-j)^β − (n-j-1)^β ]

Architecture mirrors rampde's fdeint.py (L1 scheme) pattern:
  - Forward:  product-rectangle scheme with autocast for f-eval, high-precision
              accumulation of the weighted sum
  - Backward: the EXACT discrete adjoint of the forward recurrence (not the
              continuous-adjoint approximation used by the L1 solver), run in
              reversed time
  - Three solver classes: Unscaled, Dynamic, UnscaledSafe (same adj_dtype API)

Why mixed precision is safe here
---------------------------------
Unlike the L1 scheme (which stores the y-trajectory and whose convolution
weights c_j^(k) are unbounded as N grows), this scheme stores the *function
values* f(t_j, U^j) in a low-precision buffer of shape (N, *state) and
accumulates Σ d_{n,j} f_j in float32. The key property making this safe is

    Σ_{j=0}^{n-1} d_{n,j} = (n·h)^β / Γ(β+1) ≤ T^β / Γ(β+1)

i.e. the row sums of the weight matrix are bounded independent of N (they
telescope: Σ_j [(n-j)^β - (n-j-1)^β] = n^β - 0^β = n^β). This boundedness is
what keeps the low-precision quantisation error from being amplified as the
number of steps grows — proved rigorously (forward-pass roundoff analysis,
mixed vs. high vs. low precision) in
Vince_proofs/Mx_Pre_take2.pdf.

Discrete adjoint (backward)
----------------------------
Because U^n depends on U^0 additively and on each U^j only through the single
node f_j = f(t_j, U^j) (used with weight d_{n,j} in every later state U^n,
n > j), the exact reverse-mode adjoint of this scheme is:

    a_{N-1} = ∂L/∂U^{N-1}                      (given, = upstream cotangent)
    v_j     = Σ_{n=j+1}^{N-1} d_{n,j} · a_n     for j = N-2, ..., 0
    a_j     = VJP_y( f(t_j, U^j), v_j )         (also yields ∂L/∂θ contributions)
    grad_y0 = Σ_{n=0}^{N-1} a_n                  (U^0 appears directly in every U^n)

Each v_j requires a fresh weighted sum over all "future" adjoint states, so
(like the forward pass) this is an O(N^2) computation. Remarkably, because
d_{n,j} depends only on (n - j), the same per-step weight vector used in the
forward convolution (Σ_j d_{k+1,j} f_j) is reused verbatim for the backward
convolution (Σ_i d_{N-1-i,·} a_·) — see `_predictor_weights`.
"""

import math
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.amp import autocast

try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd

from .loss_scalers import DynamicScaler
from .utils import _is_any_infinite
from .fdeint import (
    _is_tuple,
    _StateHistoryBuffer,
    _l1_convolution as _weighted_history_sum,
    _TupleFuncFDE,
    _tuple_to_tensor,
    _tensor_to_tuple,
)


# ============================================================================
# Product-rectangle weight computation
# ============================================================================

def _predictor_weights(
    k: int,
    beta_val: float,
    C: float,
    dtype: torch.dtype,
    device: torch.device,
    *, 
    graded_time: bool = False,
    tspan: Optional[torch.Tensor] = None,
    backward_time: bool = False,
) -> torch.Tensor:
    """
    Compute d_{k+1,j} for j = 0..k, i.e. the weight vector applied to the
    length-(k+1) history slice [h_0, ..., h_k] (f-history in the forward
    pass, adjoint-history in the backward pass — the two are identical
    because d_{n,j} depends only on n - j).

    If `graded_time` is True, the weights are adjusted to account for a graded time grid.

    If uniform time, returns a 1-D tensor of shape (k+1,):
        w[j] = C · [ (k+1-j)^β − (k-j)^β ],  j = 0..k

    If graded time, returns a 1-D tensor of shape (k+1,) with weights computed based on the graded time mesh.
        w[j] = C · [ ( (k+1)^r - (j)^r )^β − ( (k+1)^r - (j+1)^r )^β ],  j = 0..k

    No special-casing is needed at j=0 (unlike the L1 scheme): (k-j)^β = 0^β
    = 0 when j=k is well defined for β > 0.
    """
    if graded_time:
        if tspan is None:
            raise ValueError("tspan is required for graded-time predictor weights")
        t_next = tspan[k + 1]
        t_left = tspan[: k + 1]
        if backward_time:
            t_prev = tspan[k]
            return C * (
                torch.pow(t_next - t_left, beta_val)
                - torch.pow(t_prev - t_left, beta_val)
            )
        t_right = tspan[1: k + 2]
        return C * (
            torch.pow(t_next - t_left, beta_val)
            - torch.pow(t_next - t_right, beta_val)
        )

    j = torch.arange(0, k + 1, dtype=dtype, device=device)
    return C * (torch.pow(k + 1 - j, beta_val) - torch.pow(k - j, beta_val))


# ============================================================================
# Core forward helper
# ============================================================================

def _predictor_forward_impl(
    func: nn.Module,
    y0: torch.Tensor,
    tspan: torch.Tensor,
    beta_val: float,
    dtype_hi: torch.dtype,
    dtype_low: torch.dtype,
    *, 
    graded_time: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the product-rectangle predictor forward with mixed precision.

    Args:
        func      : FDE RHS f(t, y)
        y0        : Initial condition, shape (*state)
        tspan     : Equally-spaced time points, shape (N,)
        beta_val  : Fractional order as Python float
        dtype_hi  : High-precision dtype for weights and accumulation
        dtype_low : Low-precision dtype for function evaluation and f-history
        graded_time : If True, use a graded time grid.

    Returns:
        y_T  : Final solution U^{N-1}, shape (*state), dtype dtype_hi
        yt   : Full y-trajectory buffer, shape (N, *state), dtype dtype_low
               (saved for backward; NOT the same as the transient f-history)
    """
    N = len(tspan)
    if graded_time:
        C = 1 / math.gamma(beta_val + 1.0)
    else:
        h = (tspan[-1] - tspan[0]) / (N - 1)
        C = float(torch.pow(h, beta_val).item()) / math.gamma(beta_val + 1.0)
    device = y0.device

    y0_hi = y0.to(dtype_hi)

    # y-trajectory: persisted (low precision) for backward reconstruction of
    # the computation graph of each f(t_j, U^j).
    yt = torch.empty(N, *y0.shape, dtype=dtype_low, device=device)
    yt[0] = y0.to(dtype_low)

    # f-history: transient low-precision buffer — this is the buffer whose
    # boundedness argument (Σ_j d_{n,j} bounded) makes low precision safe.
    # Freed at the end of the forward pass; not needed for backward (which
    # recomputes f_j from the saved y-trajectory).
    fhist = torch.empty(N - 1, *y0.shape, dtype=dtype_low, device=device)

    y_current = y0_hi

    for k in range(N - 1):
        t_k = tspan[k]

        with autocast(device_type="cuda", dtype=dtype_low):
            f_k = func(t_k, y_current)
        fhist[k] = f_k.to(dtype_low)

        with autocast(device_type="cuda", enabled=False):
            if graded_time:
                # For graded time, the coefficients are no longer only dependent on n-j, but we can still compute the weights for the current step.
                weights = _predictor_weights(k, beta_val, C, dtype_hi, device, graded_time=graded_time, tspan=tspan)
            else:
                weights = _predictor_weights(k, beta_val, C, dtype_hi, device)
            conv_sum = _weighted_history_sum(weights, fhist[: k + 1], out_dtype=dtype_hi)
            y_current = y0_hi + conv_sum

        yt[k + 1] = y_current.to(dtype_low)

    del fhist
    return y_current, yt


# ============================================================================
# Core discrete-adjoint backward helper
# ============================================================================

def _predictor_backward_impl(
    func: nn.Module,
    at: torch.Tensor,
    yt: torch.Tensor,
    tspan: torch.Tensor,
    beta_val: float,
    params: Tuple[torch.Tensor, ...],
    dtype_hi: torch.dtype,
    dtype_low: torch.dtype,
    scale: Optional[float] = None,
    check_finite: bool = False,
    adj_storage_dtype: Optional[torch.dtype] = None,
    *, 
    graded_time: bool = False,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Exact discrete adjoint of the product-rectangle predictor, run forward
    in reversed time.

    For j = N-2, ..., 0 (processed in that order):
        v_j = Σ_{n=j+1}^{N-1} d_{n,j} · a_n
        a_j = VJP_y( f(t_j, U^j), v_j )
        grad_θ += VJP_θ( f(t_j, U^j), v_j )        (no extra h scaling — the
                                                      h-dependence is already
                                                      fully inside d_{n,j})
    with a_{N-1} = at (the incoming cotangent). Finally:
        grad_y0 = Σ_{n=0}^{N-1} a_n

    because U^0 appears as a direct (identity-Jacobian) additive term in
    every U^n, n = 1..N-1, in addition to being the argument of f_0.

    Args:
        func              : FDE RHS f(t, y)
        at                : Gradient w.r.t. y_T = U^{N-1}, shape (*state)
        yt                : Stored forward y-trajectory, shape (N, *state), dtype_low
        tspan             : Forward time points, shape (N,)
        beta_val          : Fractional order as Python float
        params            : Tuple of FDE function parameters
        dtype_hi          : High-precision dtype (used for all computations)
        dtype_low         : Low-precision dtype (used for f-eval autocast)
        scale             : Optional scaling factor for VJP cotangents (DynamicScaler.S)
        check_finite      : If True, raise OverflowError on non-finite values
        adj_storage_dtype : Dtype for storing the adjoint history buffer.
                            None (default) → dtype_hi. Set to dtype_low to
                            halve adjoint memory (same trade-off as fdeint's
                            adj_dtype).
        graded_time       : If True, the time points in tspan are treated as graded,

    Returns:
        grad_y0    : Gradient w.r.t. the initial condition y_0
        grad_params: Tuple of gradients for each parameter
    """
    N = len(tspan)
    backward_tspan = tspan[-1] - tspan.flip(0)
    if graded_time:
        C = 1 / math.gamma(beta_val + 1.0)
    else:
        h = (tspan[-1] - tspan[0]) / (N - 1)
        C = float(torch.pow(h, beta_val).item()) / math.gamma(beta_val + 1.0)
    device = yt.device

    _adj_dtype = adj_storage_dtype if adj_storage_dtype is not None else dtype_hi

    any_param_req_grad = any(p.requires_grad for p in params) if params else False

    # adj_buf[i] holds a_{N-1-i} for i = 0..N-1:
    #   adj_buf[0]   = a_{N-1} = at             (given)
    #   adj_buf[r+1] = a_{N-2-r}                (computed at reversed step r)
    adj_buf = _StateHistoryBuffer(at, N, _adj_dtype)
    adj_buf.set(0, at.to(dtype_hi))

    grad_params = [torch.zeros_like(p) for p in params]

    for r in range(N - 1):
        target_j = N - 2 - r
        t_j = tspan[target_j]  # For graded time, tspan is already adjusted

        y_j = yt[target_j].to(dtype_hi).detach().requires_grad_(True)

        with torch.enable_grad():
            with autocast(device_type="cuda", dtype=dtype_low):
                f_j = func(t_j, y_j)

        # v_j = Σ_{i=0}^{r} d_{N-1-i, target_j} · adj_buf[i]
        # Uses the SAME weight formula as the forward pass (row k=r), since
        # d_{n,j} depends only on n-j — see _predictor_weights docstring.
        if graded_time:
            # For graded time, the coefficients are no longer only dependent on n-j, but we can still compute the weights for the current reversed step.
            weights = _predictor_weights(
                r,
                beta_val,
                C,
                dtype_hi,
                device,
                graded_time=graded_time,
                tspan=backward_tspan,
                backward_time=True,
            )
        else:
            weights = _predictor_weights(r, beta_val, C, dtype_hi, device)
        v_j = _weighted_history_sum(weights, adj_buf[: r + 1], out_dtype=dtype_hi)

        scaled_v = (scale * v_j) if scale is not None else v_j

        if check_finite and _is_any_infinite(scaled_v):
            raise OverflowError(f"Non-finite scaled cotangent at reversed step {r}")

        if any_param_req_grad:
            vjp_all = torch.autograd.grad(
                f_j,
                (y_j, *params),
                scaled_v.to(f_j.dtype),
                allow_unused=True,
                create_graph=False,
            )
            vjp_y = vjp_all[0]
            vjp_params = list(vjp_all[1:])
        else:
            vjp_y = torch.autograd.grad(
                f_j, y_j, scaled_v.to(f_j.dtype), create_graph=False
            )[0]
            vjp_params = [None] * len(params)

        if vjp_y is None:
            vjp_y = torch.zeros_like(y_j)

        if scale is not None:
            inv_scale = 1.0 / scale
            vjp_y = inv_scale * vjp_y
            vjp_params = [
                None if vp is None else inv_scale * vp for vp in vjp_params
            ]

        if check_finite and _is_any_infinite(vjp_y):
            raise OverflowError(f"Non-finite VJP at reversed step {r}")

        # grad_θ += VJP_θ(f_j, v_j) — no extra h scaling (see docstring).
        for g, vp in zip(grad_params, vjp_params):
            if vp is not None:
                g.add_(vp.to(g.dtype))

        a_j = vjp_y.to(dtype_hi)

        if check_finite and _is_any_infinite(a_j):
            raise OverflowError(f"Non-finite adjoint state at reversed step {r}")

        adj_buf.set(r + 1, a_j)

    # grad_y0 = Σ_{n=0}^{N-1} a_n  (direct identity term across all n=1..N-1,
    # plus the VJP-based a_0 contribution via f_0)
    grad_y0 = torch.zeros_like(at, dtype=dtype_hi)
    for i in range(N):
        grad_y0 = grad_y0 + adj_buf[i].to(dtype_hi)

    if check_finite and _is_any_infinite(grad_y0):
        raise OverflowError("Non-finite grad_y0 in discrete adjoint")

    return grad_y0, tuple(grad_params)


# ============================================================================
# Base solver class — shared forward pass
# ============================================================================

class PredictorFDESolverBase(torch.autograd.Function):
    """
    Base class for fixed-grid product-rectangle predictor FDE solvers.

    Forward:  runs the predictor scheme with mixed-precision and stores the
              y-trajectory.
    Backward: must be implemented by subclasses (exact discrete adjoint).

    Forward signature:
        forward(ctx, func, y0, tspan, beta_val, adj_storage_dtype, loss_scaler, *params) -> y_T
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        func: nn.Module,
        y0: torch.Tensor,
        tspan: torch.Tensor,
        beta_val: float,
        adj_storage_dtype: Optional[torch.dtype],
        loss_scaler: Any,
        graded_time: bool = False,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            dtype_hi = y0.dtype
            dtype_low = (
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled()
                else dtype_hi
            )
            y_T, yt = _predictor_forward_impl(
                func, y0, tspan, beta_val, dtype_hi, dtype_low, graded_time=graded_time
            )

        ctx.save_for_backward(yt, *params)
        ctx.func = func
        ctx.tspan = tspan
        ctx.beta_val = beta_val
        ctx.dtype_hi = dtype_hi
        ctx.adj_storage_dtype = adj_storage_dtype
        ctx.loss_scaler = loss_scaler
        ctx.graded_time = graded_time

        return y_T

    @staticmethod
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        raise NotImplementedError("Subclasses must implement backward.")


# ============================================================================
# Unscaled backward — optimal for float32 / bfloat16
# ============================================================================

class PredictorFDESolverUnscaled(PredictorFDESolverBase):
    """Predictor FDE solver without scaling. Use for float32 / bfloat16."""

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, at: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        yt, *params = ctx.saved_tensors
        params = tuple(params)
        dtype_hi = ctx.dtype_hi
        dtype_low = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else dtype_hi
        )

        with torch.no_grad():
            grad_y0, grad_params = _predictor_backward_impl(
                ctx.func, at, yt, ctx.tspan, ctx.beta_val,
                params, dtype_hi, dtype_low,
                scale=None, check_finite=False,
                adj_storage_dtype=ctx.adj_storage_dtype,
                graded_time=ctx.graded_time,
            )

        # Signature: (func, y0, tspan, beta_val, adj_storage_dtype, loss_scaler, *params)
        return (None, grad_y0, None, None, None, None, None, *grad_params)


# ============================================================================
# Dynamic scaling backward — for float16 with DynamicScaler
# ============================================================================

class PredictorFDESolverDynamic(PredictorFDESolverBase):
    """
    Predictor FDE solver with DynamicScaler retry loop.

    As with the L1 FDE adjoint, the discrete adjoint here has memory across
    reversed steps, so a scale change requires rerunning the full backward
    from scratch — the retry wraps the entire adjoint solve.
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, at: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        yt, *params = ctx.saved_tensors
        params = tuple(params)
        dtype_hi = ctx.dtype_hi
        dtype_low = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else dtype_hi
        )
        scaler: DynamicScaler = ctx.loss_scaler

        if scaler.S is None:
            scaler.init_scaling(at.to(dtype_hi))

        old_params = {name: p.data for name, p in ctx.func.named_parameters()}
        for _, p in ctx.func.named_parameters():
            p.data = p.data.to(dtype_low)

        try:
            attempts = 0
            while attempts < scaler.max_attempts:
                try:
                    with torch.no_grad():
                        grad_y0, grad_params = _predictor_backward_impl(
                            ctx.func, at, yt, ctx.tspan, ctx.beta_val,
                            params, dtype_hi, dtype_low,
                            scale=scaler.S, check_finite=True,
                            adj_storage_dtype=ctx.adj_storage_dtype,
                            graded_time=ctx.graded_time
                        )
                    if _is_any_infinite((grad_y0, *grad_params)):
                        raise OverflowError("Non-finite gradients after adjoint solve.")
                    break
                except OverflowError:
                    scaler.update_on_overflow()
                    attempts += 1
            else:
                raise RuntimeError(
                    f"Predictor FDE dynamic backward exceeded {scaler.max_attempts} attempts."
                )

            if scaler.check_for_increase(grad_y0):
                scaler.update_on_small_grad()

        finally:
            for name, p in ctx.func.named_parameters():
                p.data = old_params[name]

        return (None, grad_y0, None, None, None, None, None, *grad_params)


# ============================================================================
# Unscaled-safe backward — for float16 with PyTorch GradScaler
# ============================================================================

class PredictorFDESolverUnscaledSafe(PredictorFDESolverBase):
    """
    Predictor FDE solver with exception handling and inf-gradient fallback.

    Compatible with PyTorch's GradScaler. On overflow, returns inf gradients
    so that GradScaler can detect and reduce the outer loss scale.
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, at: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        yt, *params = ctx.saved_tensors
        params = tuple(params)
        dtype_hi = ctx.dtype_hi
        dtype_low = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else dtype_hi
        )

        try:
            with torch.no_grad():
                grad_y0, grad_params = _predictor_backward_impl(
                    ctx.func, at, yt, ctx.tspan, ctx.beta_val,
                    params, dtype_hi, dtype_low,
                    scale=None, check_finite=True,
                    adj_storage_dtype=ctx.adj_storage_dtype,
                    graded_time=ctx.graded_time
                )
            if _is_any_infinite((grad_y0, *grad_params)):
                raise OverflowError("Non-finite gradients after adjoint solve.")
        except OverflowError:
            grad_y0 = torch.full_like(at, float("inf"))
            grad_params = tuple(torch.full_like(p, float("inf")) for p in params)

        return (None, grad_y0, None, None, None, None, None, *grad_params)


# ============================================================================
# Solver selection (mirrors rampde's _select_fde_solver / _select_ode_solver)
# ============================================================================

ScalerType = Union[DynamicScaler, None, Literal[False]]


def _select_predictor_solver(
    loss_scaler: ScalerType,
    precision: torch.dtype,
) -> Tuple[Type[PredictorFDESolverBase], Optional[DynamicScaler]]:
    """
    Select the optimal predictor solver variant based on scaler type and precision.

    Selection logic mirrors rampde._select_fde_solver:
      - DynamicScaler instance       → PredictorFDESolverDynamic
      - None + float16 under autocast → auto-create DynamicScaler → Dynamic
      - None + float32/bfloat16      → PredictorFDESolverUnscaled
      - None + float16 (no autocast) → PredictorFDESolverUnscaledSafe
      - False                        → disable internal scaling → Unscaled/Safe
    """
    if loss_scaler is False:
        loss_scaler = None
    elif loss_scaler is None:
        dtype_low = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else precision
        )
        if dtype_low == torch.float16:
            loss_scaler = DynamicScaler(dtype_low=dtype_low)

    if isinstance(loss_scaler, DynamicScaler):
        return PredictorFDESolverDynamic, loss_scaler

    if loss_scaler is None:
        if precision in (torch.float32, torch.bfloat16, torch.float64):
            return PredictorFDESolverUnscaled, loss_scaler
        return PredictorFDESolverUnscaledSafe, loss_scaler

    return PredictorFDESolverUnscaledSafe, loss_scaler


# ============================================================================
# Main public API
# ============================================================================

def predictor_fdeint(
    func: nn.Module,
    y0: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    beta: Union[float, torch.Tensor],
    t: Union[float, torch.Tensor],
    step_size: Union[float, torch.Tensor],
    *,
    loss_scaler: ScalerType = None,
    adj_dtype: Optional[torch.dtype] = None,
    graded_time: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Solve a Caputo fractional ODE with the mixed-precision Volterra
    product-rectangle ("basic predictor") scheme.

    Solves:
        D^β y(t) = func(t, y),  y(0) = y0,  β ∈ (0, 1]

    using:
        U^n = U^0 + Σ_{j=0}^{n-1} d_{n,j} f(t_j, U^j),
        d_{n,j} = h^β / Γ(β+1) · [(n-j)^β - (n-j-1)^β] on a uniform time grid, or a graded time grid we use
        d_{n,j} = C · [ ( (n)^r - (j)^r )^β − ( (n)^r - (j+1)^r )^β ] with r = (2-β)/β and C = T^β / ( Γ(β+1) • N^(rβ) ).

    with automatic solver selection based on precision, and the EXACT
    discrete adjoint for backward (not the continuous-adjoint approximation
    used by rampde.fdeint's L1 scheme).

    Args:
        func       : FDE RHS as an nn.Module implementing forward(t, y).
        y0         : Initial condition — Tensor or tuple of Tensors.
        beta       : Fractional order in (0, 1).
        t          : End time (float or scalar Tensor, must be > 0).
        step_size  : Uniform time step (float or scalar Tensor, must be < t).
        loss_scaler: Mixed-precision scaling strategy:
                     - None  : auto-select (DynamicScaler for float16)
                     - False : disable internal scaling
                     - DynamicScaler instance : use provided scaler
        adj_dtype  : Dtype for storing the adjoint history during backward.
                     - None (default) : use dtype_hi (float32) — safest
                     - torch.float16 / torch.bfloat16 : halve adjoint memory

    Returns:
        Solution y(t) — same structure as y0 (Tensor or tuple of Tensors).

    Solver selection (same logic as rampde.fdeint):
        - float32 / bfloat16 : PredictorFDESolverUnscaled  (fastest)
        - float16 + autocast  : PredictorFDESolverDynamic   (DynamicScaler)
        - float16 otherwise   : PredictorFDESolverUnscaledSafe

    Example::

        class FDEFunc(nn.Module):
            def forward(self, t, y):
                return -y

        y0 = torch.ones(10, device='cuda')
        y_T = predictor_fdeint(FDEFunc(), y0, beta=0.5, t=10.0, step_size=0.1)
    """
    if not isinstance(func, nn.Module):
        raise TypeError("func must be an instance of nn.Module.")

    device = y0[0].device if _is_tuple(y0) else y0.device

    beta_val: float
    if isinstance(beta, torch.Tensor):
        beta_val = float(beta.item())
    else:
        beta_val = float(beta)
    if not (0.0 < beta_val <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta_val}")

    t_val: float
    if isinstance(t, torch.Tensor):
        t_val = float(t.item())
    else:
        t_val = float(t)
    if t_val <= 0.0:
        raise ValueError(f"t must be > 0, got {t_val}")

    h_val: float
    if isinstance(step_size, torch.Tensor):
        h_val = float(step_size.item())
    else:
        h_val = float(step_size)
    if h_val <= 0.0 or h_val >= t_val:
        raise ValueError(f"step_size must be in (0, t), got {h_val}")

    num_steps = int(round(t_val / h_val)) + 1

    if graded_time:
        # Use double-graded time mesh with r = (2 - beta) / beta
        ind = torch.arange(num_steps, dtype=torch.float32, device=device)
        r = (2.0 - beta_val) / beta_val
        c = (num_steps - 1) / 2
        q = 1 - torch.abs(ind - c) / c
        tspan = t_val/2 * (1 + torch.sign(ind - c) * (1 - torch.pow(q,r)))
    else:
        tspan = torch.linspace(0.0, t_val, num_steps, dtype=torch.float32, device=device)

    y0_is_tuple = _is_tuple(y0)
    if y0_is_tuple:
        shapes = [yi.shape for yi in y0]
        numels = [int(yi.numel()) for yi in y0]
        func = _TupleFuncFDE(func, shapes, numels)
        y0 = _tuple_to_tensor(y0)

    precision = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled()
        else y0.dtype
    )
    solver_class, loss_scaler = _select_predictor_solver(loss_scaler, precision)

    params = tuple(func.parameters())

    solution = solver_class.apply(func, y0, tspan, beta_val, adj_dtype, loss_scaler, graded_time, *params)

    if y0_is_tuple:
        return _tensor_to_tuple(solution, numels, shapes)
    return solution
