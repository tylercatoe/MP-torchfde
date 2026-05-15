import torch
import math
import torch.nn as nn
from typing import Any, Optional, Tuple, Union, Literal, Type
from .utils_fde import _flatten, _flatten_convert_none_to_zeros,_check_inputs, _flat_to_shape
from .utils_fde import _addmul_inplace, _mul_inplace, _minusmul_inplace
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus, ReversedListView
from . import config
#from torch.amp import autocast

ScalerType = Union["DynamicScaler", None, Literal[False]]


def _is_any_infinite(x: Union[torch.Tensor, tuple, list, None]) -> bool:
    """Recursively check if any tensor contains inf or NaN."""
    if x is None:
        return False
    if isinstance(x, torch.Tensor):
        return not x.isfinite().all().item()
    if isinstance(x, (list, tuple)):
        return any(_is_any_infinite(elem) for elem in x)
    return False


def _state_storage_dtype(dtype_hi: torch.dtype) -> torch.dtype:
    """Choose dtype used to store forward trajectory snapshots."""
    return torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi


def _cast_state_dtype(state, dtype: torch.dtype):
    """Cast tensor or tuple-of-tensors to a target dtype (no-op if already matching)."""
    if _is_tuple(state):
        return tuple(s if s.dtype == dtype else s.to(dtype) for s in state)
    return state if state.dtype == dtype else state.to(dtype)


def _cast_state_like(state, like_state):
    """Cast `state` to match dtypes of `like_state` (tensor or tuple-of-tensors)."""
    if _is_tuple(like_state):
        return tuple(
            s if s.dtype == l.dtype else s.to(l.dtype)
            for s, l in zip(state, like_state)
        )
    return state if state.dtype == like_state.dtype else state.to(like_state.dtype)


class _StateHistoryBuffer:
    """Preallocated trajectory storage with list-like indexing semantics."""

    def __init__(self, template_state, length: int, storage_dtype: torch.dtype):
        self.length = int(length)
        self.storage_dtype = storage_dtype
        self.is_tuple = _is_tuple(template_state)

        if self.is_tuple:
            self._buffers = tuple(
                torch.empty((self.length, *s.shape), dtype=storage_dtype, device=s.device)
                for s in template_state
            )
        else:
            self._buffers = torch.empty(
                (self.length, *template_state.shape),
                dtype=storage_dtype,
                device=template_state.device,
            )

    def set(self, idx: int, state) -> None:
        if self.is_tuple:
            for buf, s in zip(self._buffers, state):
                src = s if s.dtype == self.storage_dtype else s.to(self.storage_dtype)
                buf[idx].copy_(src)
        else:
            src = state if state.dtype == self.storage_dtype else state.to(self.storage_dtype)
            self._buffers[idx].copy_(src)

    def __getitem__(self, idx: int):
        if self.is_tuple:
            return tuple(buf[idx] for buf in self._buffers)
        return self._buffers[idx]

    def __len__(self):
        return self.length


class DynamicScaler:
    """Dynamic loss scaler for mixed-precision adjoint backpropagation."""

    def __init__(
        self,
        dtype_low: torch.dtype,
        target_factor: Optional[float] = None,
        increase_factor: float = 2.0,
        decrease_factor: float = 0.5,
        max_attempts: int = 50,
        delta: float = 0,
        verbose: bool = False,
    ):
        self.dtype_low = dtype_low
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_attempts = max_attempts
        self.delta = delta
        self.is_initialized = False
        self.S: Optional[float] = None
        self.__name__ = "DynamicScaler"
        self.verbose = verbose
        self.scale_history = []

    def init_scaling(self, a: torch.Tensor) -> None:
        if not a.isfinite().all() or a.isnan().any():
            n_inf = torch.isinf(a).sum().item()
            n_nan = torch.isnan(a).sum().item()
            raise ValueError(
                f"Input tensor contains non-finite values: {n_inf} inf, {n_nan} nan (shape: {a.shape})"
            )

        target = self.target / math.sqrt(max(1.0, a.numel() / max(1, a.shape[0])))
        a_max = a.abs().max()
        self.S = target / (a_max + self.delta).to(torch.float32)
        self.S = 2 ** (torch.round(torch.log2(self.S))).item()

        initial_S = self.S
        for _ in range(20):
            if (self.S * a).isfinite().all():
                break
            self.S *= 0.5
        else:
            raise RuntimeError(
                f"Scaler failed to find finite scale after 20 steps for {a.shape} with ||a||_inf = {a.abs().max()}."
            )

        if self.verbose and self.S != initial_S:
            print(f"[DynamicScaler] Adjusted initial scale from {initial_S:.3e} to {self.S:.3e}")

        self.is_initialized = True
        self.scale_history.append(("init", self.S))

    def update_on_overflow(self) -> None:
        old_S = self.S
        self.S *= self.decrease_factor
        if self.verbose:
            print(f"[DynamicScaler] Overflow detected: scale reduced from {old_S:.6e} to {self.S:.6e}")
        self.scale_history.append(("overflow", self.S))

    def check_for_increase(self, a: torch.Tensor) -> bool:
        a_max = a.abs().max()
        return (a_max / self.target).item() < 0.5

    def update_on_small_grad(self) -> None:
        old_S = self.S
        self.S *= self.increase_factor
        if self.verbose:
            print(f"[DynamicScaler] Small gradient: scale increased from {old_S:.6e} to {self.S:.6e}")
        self.scale_history.append(("increase", self.S))


def _select_adjoint_solver(
    loss_scaler: ScalerType,
    precision: torch.dtype,
) -> Tuple[Type[torch.autograd.Function], Optional[DynamicScaler]]:
    """
    Select an adjoint backend following the same high-level policy as rampde:
      - DynamicScaler => dynamic scaling backend
      - None + stable precision => unscaled backend
      - None + float16 => auto-create DynamicScaler and use dynamic backend
      - False + float16 => unscaled safe backend
    """
    if loss_scaler is False:
        loss_scaler = None
        if precision == torch.float16:
            return FDEAdjointMethodUnscaledSafe, loss_scaler

    elif loss_scaler is None:
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else precision
        if dtype_low == torch.float16:
            loss_scaler = DynamicScaler(dtype_low=dtype_low)

    if isinstance(loss_scaler, DynamicScaler):
        return FDEAdjointMethodDynamic, loss_scaler
    if loss_scaler is None:
        if precision in [torch.float32, torch.bfloat16, torch.float64]:
            return FDEAdjointMethodUnscaled, loss_scaler
        return FDEAdjointMethodUnscaledSafe, loss_scaler
    return FDEAdjointMethodUnscaledSafe, loss_scaler


def fdeint_adjoint(func, y0, beta, t, step_size, method, options=None, loss_scaler: ScalerType = None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    # Wrap single tensor inputs in a tuple for unified processing
    if torch.is_tensor(y0):
        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0, ) # Convert tensor to tuple
        func = TupleFunc(func) # Wrap function to handle tensor input/output

    # Validate inputs and prepare for solving
    shapes, _, func, y0, tspan, method, beta = _check_inputs(func, y0, t, step_size, method, beta, SOLVERS_Forward)

    if options is None:
        options = {}

    # Get parameters
    params = find_parameters(func)
    n_state = len(y0)
    n_params = len(params)

    # Determine precision and select adjoint backend
    precision = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else y0[0].dtype
    adjoint_solver, loss_scaler = _select_adjoint_solver(loss_scaler, precision)

    # Call selected adjoint backend
    solution = adjoint_solver.apply(
        func, n_state, n_params, loss_scaler, *y0, beta, tspan, method, *params, options
    )

    # Post-process solution based on tensor mode
    if config.TENSOR_MODE == 'concat':
        # CONCAT MODE: Always reshape the flattened solution back to original structure
        # Note: In adjoint method, inputs are always flattened/concatenated regardless of original type
        assert shapes is not None, 'for tuple, we need to provide shapes'
        solution = solution[0] # Extract from solver output
        solution = _flat_to_shape(solution, (), shapes) # Reshape to original structure
        if tensor_input:
            solution = solution[0] # If original input was a tensor, extract it from the tuple
    else:
        # NON-CONCAT MODE: Only unwrap if original input was a tensor
        if tensor_input:
            solution = solution[0]

    # Validate output type matches original input type
    if tensor_input:
        assert torch.is_tensor(solution)
    else:
        assert isinstance(solution, tuple)

    return solution

def _parse_adjoint_args(n_state: int, n_params: int, args: tuple):
    y0_tuple = tuple(args[:n_state])
    beta = args[n_state]
    tspan = args[n_state + 1]
    method = args[n_state + 2]
    func_params = tuple(args[n_state + 3 : n_state + 3 + n_params])
    options = args[n_state + 3 + n_params]
    return y0_tuple, beta, tspan, method, func_params, options


def _backward_none_tuple(ctx):
    n_state = ctx.n_state
    n_params = ctx.n_params
    grads = []
    grads.append(None)  # func
    grads.append(None)  # n_state
    grads.append(None)  # n_params
    grads.append(None)  # loss_scaler
    grads.extend([None] * n_state)  # y0_1,...,y0_n
    grads.append(None)  # beta
    grads.append(None)  # tspan
    grads.append(None)  # method
    grads.extend([None] * n_params)  # params
    grads.append(None)  # options
    return tuple(grads)


def _cleanup_ctx(ctx):
    for name in (
        "ans",
        "yhistory",
        "func",
        "func_params",
        "beta",
        "method",
        "loss_scaler",
        "options",
        "_grad_output",
    ):
        if hasattr(ctx, name):
            delattr(ctx, name)


def _forward_impl(ctx, func, n_state, n_params, loss_scaler, *args):
    n_state = int(n_state)
    n_params = int(n_params)
    y0_tuple, beta, tspan, method, func_params, options = _parse_adjoint_args(n_state, n_params, args)

    with torch.no_grad():
        ans, yhistory = SOLVERS_Forward[method](func=func, y0=y0_tuple, beta=beta, tspan=tspan, **options)

    y0_needs_grad = any(t.requires_grad for t in y0_tuple)
    params_need_grad = any(p.requires_grad for p in func_params) if func_params else False

    ctx.n_state = n_state
    ctx.n_params = n_params
    if y0_needs_grad or params_need_grad:
        ctx.save_for_backward(tspan)
        ctx.ans = ans
        ctx.yhistory = yhistory
        ctx.func = func
        ctx.beta = beta
        ctx.method = method
        ctx.func_params = func_params
        ctx.loss_scaler = loss_scaler
        ctx.options = options
    else:
        del yhistory
    return ans


def _build_augmented_dynamics(func, n_tensors, func_params, scale: Optional[float] = None, check_finite: bool = False):
    class AugDynamics:
        def __init__(self, func_, n_tensors_, func_params_, scale_, check_finite_):
            self.func = func_
            self.n_tensors = n_tensors_
            self.f_params = func_params_
            self.scale = scale_
            self.check_finite = check_finite_

        def __call__(self, t, y_aug):
            y, adj_y, adj_params = y_aug

            with torch.set_grad_enabled(True):
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = self.func(t, y)

                if self.scale is None:
                    grad_outputs = tuple(adj_y)
                else:
                    grad_outputs = tuple(self.scale * adj_y_ for adj_y_ in adj_y)

                if self.check_finite and _is_any_infinite((func_eval, grad_outputs)):
                    raise OverflowError("Non-finite values detected before VJP computation.")

                vjp_y_and_params = torch.autograd.grad(
                    func_eval,
                    y + self.f_params,
                    grad_outputs,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )

            vjp_y = vjp_y_and_params[:self.n_tensors]
            vjp_params = vjp_y_and_params[self.n_tensors:]

            if self.scale is not None:
                inv_scale = 1.0 / self.scale
                vjp_y = tuple(None if v is None else inv_scale * v for v in vjp_y)
                vjp_params = tuple(None if v is None else inv_scale * v for v in vjp_params)

            vjp_y = tuple(
                torch.zeros_like(y_) if vjp_y_ is None else vjp_y_
                for vjp_y_, y_ in zip(vjp_y, y)
            )
            vjp_params = tuple(
                torch.zeros_like(p) if vp is None else vp
                for vp, p in zip(vjp_params, self.f_params)
            )

            if self.check_finite and _is_any_infinite((vjp_y, vjp_params)):
                raise OverflowError("Non-finite values detected in VJP outputs.")

            return (func_eval, vjp_y, vjp_params)

    return AugDynamics(func, n_tensors, func_params, scale, check_finite)


def _run_adjoint_solver(ctx, scale: Optional[float], check_finite: bool):
    tspan = ctx.saved_tensors[0]
    ans = tuple(ctx.ans)
    yhistory = ctx.yhistory
    func = ctx.func
    beta = ctx.beta
    method = ctx.method
    func_params = ctx.func_params
    n_tensors = ctx.n_state
    options = getattr(ctx, "options", {})

    tspan_flip = tspan.flip(0)
    yhistory_flip = ReversedListView(yhistory) if yhistory is not None else None
    augmented_dynamics = _build_augmented_dynamics(
        func, n_tensors, func_params, scale=scale, check_finite=check_finite
    )

    with torch.no_grad():
        grad_output = tuple(
            torch.zeros_like(ans_i) if go is None else go
            for ans_i, go in zip(ans, ctx._grad_output)
        )
        if func_params:
            adj_params = tuple(torch.zeros_like(p) for p in func_params)
        else:
            adj_params = ()
        aug_y0 = (ans, grad_output, adj_params)
        adj_y, adj_params = SOLVERS_Backward[method](
            augmented_dynamics,
            aug_y0,
            beta,
            tspan_flip,
            yhistory_flip,
            **options,
        )

    return adj_y, adj_params


def _backward_impl(ctx, grad_output, mode: str):
    if not hasattr(ctx, "yhistory"):
        return _backward_none_tuple(ctx)

    ctx._grad_output = grad_output

    if mode == "unscaled":
        adj_y, adj_params = _run_adjoint_solver(ctx, scale=None, check_finite=False)

    elif mode == "dynamic":
        scaler = ctx.loss_scaler
        if scaler is None:
            dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else ctx.ans[0].dtype
            scaler = DynamicScaler(dtype_low=dtype_low)
            ctx.loss_scaler = scaler

        flat_grad = _flatten(tuple(
            torch.zeros_like(ctx.ans[i]) if g is None else g for i, g in enumerate(grad_output)
        ))
        if scaler.S is None:
            scaler.init_scaling(flat_grad)

        # Optional parameter dtype conversion to mirror rampde dynamic backend.
        # Keep references to original parameter tensors so we can restore them
        # without cloning, which reduces transient peak-memory spikes.
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else flat_grad.dtype
        old_params = {name: p.data for name, p in ctx.func.named_parameters()}
        for _, p in ctx.func.named_parameters():
            p.data = p.data.to(dtype_low)

        try:
            attempts = 0
            while attempts < scaler.max_attempts:
                try:
                    adj_y, adj_params = _run_adjoint_solver(ctx, scale=scaler.S, check_finite=True)
                    if _is_any_infinite((adj_y, adj_params)):
                        raise OverflowError("Non-finite values detected after adjoint solve.")
                    break
                except OverflowError:
                    scaler.update_on_overflow()
                    attempts += 1
            else:
                raise RuntimeError(
                    f"Reached maximum number of {scaler.max_attempts} attempts in dynamic adjoint backward."
                )

            if scaler.check_for_increase(_flatten(adj_y)):
                scaler.update_on_small_grad()
        finally:
            for name, p in ctx.func.named_parameters():
                p.data = old_params[name]

    elif mode == "safe":
        try:
            adj_y, adj_params = _run_adjoint_solver(ctx, scale=None, check_finite=True)
            if _is_any_infinite((adj_y, adj_params)):
                raise OverflowError("Non-finite values detected after adjoint solve.")
        except OverflowError:
            adj_y = tuple(torch.full_like(a_i, float("inf")) for a_i in ctx.ans)
            adj_params = tuple(torch.full_like(p, float("inf")) for p in ctx.func_params)
    else:
        raise ValueError(f"Unknown adjoint backward mode: {mode}")

    _cleanup_ctx(ctx)
    return None, None, None, None, *adj_y, None, None, None, *adj_params, None


class FDEAdjointMethodUnscaled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, n_state, n_params, loss_scaler, *args):
        return _forward_impl(ctx, func, n_state, n_params, loss_scaler, *args)

    @staticmethod
    def backward(ctx, *grad_output):
        return _backward_impl(ctx, grad_output, mode="unscaled")


class FDEAdjointMethodDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, n_state, n_params, loss_scaler, *args):
        return _forward_impl(ctx, func, n_state, n_params, loss_scaler, *args)

    @staticmethod
    def backward(ctx, *grad_output):
        return _backward_impl(ctx, grad_output, mode="dynamic")


class FDEAdjointMethodUnscaledSafe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, n_state, n_params, loss_scaler, *args):
        return _forward_impl(ctx, func, n_state, n_params, loss_scaler,  *args)

    @staticmethod
    def backward(ctx, *grad_output):
        return _backward_impl(ctx, grad_output, mode="safe")


# Backward-compat alias.
FDEAdjointMethod = FDEAdjointMethodUnscaled


def forward_predictor(func, y0, beta, tspan, **options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the intial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        gamma_beta = 1 / math.gamma(beta)
        h_beta_over_beta = torch.pow(h, beta) / beta

        fhistory = []
        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
            dtype_hi = y0[0].dtype
        else:
            device = y0.device
            dtype_hi = y0.dtype
        dtype_store = _state_storage_dtype(dtype_hi)
        yn = _clone(y0)
        yhistory = _StateHistoryBuffer(yn, N, dtype_store)
        # yn = y0

        for k in range(N - 1):
            tn = tspan[k]

            with torch.autocast(device_type='cuda', dtype=dtype_store):
                f_k = func(tn, yn)

            fhistory.append(_cast_state_dtype(f_k, dtype_store))
            yhistory.set(k, yn)

            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device).unsqueeze(1)

            b_j_k_1 = h_beta_over_beta * (torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

            convolution_sum = None

            with torch.autocast(device_type='cuda', enabled=False):
                hist = [_cast_state_dtype(hist_item, dtype_hi) for hist_item in fhistory[start_idx : k + 1]]
                b_vals = b_j_k_1.reshape(-1) # Flatten b_j_k_1 to match the shape of hist for broadcasting
                
                if _is_tuple(hist[0]):
                    # If the history elements are tuples, we need to handle each component separately
                    convolution_sum = tuple(
                        torch.tensordot(b_vals, torch.stack([h[i] for h in hist], dim=0), dims=([0],[0])) for i in range(len(hist[0]))
                    )
                else:
                    hist = torch.stack(hist, dim=0)
                    convolution_sum = torch.tensordot(b_vals, hist, dims=([0],[0]))  # Vectorized computation of convolution sum for non-tuple case

                # for j in range(start_idx, k + 1):
                #     local_idx = j - start_idx  # CHANGED: Use local index for b_j_k_1
                #     if convolution_sum is None:
                #         convolution_sum = _multiply(b_j_k_1[local_idx], fhistory[j])
                #     else:
                #         # convolution_sum = _add(convolution_sum, _multiply(b_j_k_1[local_idx], fhistory[j]))
                #         #_addmul_inplace(target, source, alpha):
                #         # In-place fused multiply-add operation: target += alpha * source
                #         convolution_sum = _addmul_inplace(convolution_sum, fhistory[j], b_j_k_1[local_idx])

                # Final update step
                # weight_term = _multiply(gamma_beta, convolution_sum)
                weight_term = _mul_inplace(convolution_sum, gamma_beta)
                yn = _add(y0, weight_term)

        yhistory.set(N - 1, yn)
        # release memory
        del fhistory
        return yn, yhistory

def backward_predictor(func, y_aug, beta, tspan, yhistory, **options):
    # mixed order predictor with beta and 1.
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        gamma_beta = 1 / math.gamma(beta)
        # CHANGED: Pre-compute h^beta/beta for efficiency
        h_beta_over_beta = torch.pow(h, beta) / beta

        fadj_history = []
        if yhistory is None:  # CHANGED: Fixed condition (was "if True:")
            fy_history = []

        y0, adj_y0, adj_params0 = y_aug  ### we will use yhistory rather than compute y again
        if _is_tuple(adj_y0):
            device = adj_y0[0].device
            dtype_hi = adj_y0[0].dtype
        else:
            device = adj_y0.device
            dtype_hi = adj_y0.dtype
        dtype_store = _state_storage_dtype(dtype_hi)

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)
        y = _clone(y0)

        for k in range(N - 1):
            tn = tspan[k]

            # CHANGED: Fixed memory handling to match corrected forward_predictor
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            # CHANGED: Corrected start_idx calculation
            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: j_vals now starts from start_idx instead of 0
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device).unsqueeze(1)

            # CHANGED: Use torch.pow and pre-computed h_beta_over_beta
            b_j_k_1 = h_beta_over_beta * (
                    torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

            with torch.autocast(device_type='cuda', dtype=dtype_store):
                func_eval, vjp_y, vjp_params = func(tn, (y, adj_y, adj_params))

            fadj_history.append(_cast_state_dtype(vjp_y, dtype_store))
            if yhistory is None:
                fy_history.append(_cast_state_dtype(func_eval, dtype_store))

            hist = [_cast_state_dtype(hist_item, dtype_hi) for hist_item in fadj_history[start_idx : k + 1]]
            b_vals = b_j_k_1.reshape(-1)

            with torch.autocast(device_type='cuda', enabled=False):
                if _is_tuple(hist[0]):
                    convolution_sum = tuple(
                        torch.tensordot(
                            b_vals,
                            torch.stack([hist_item[i] for hist_item in hist], dim=0),
                            dims=([0], [0]),
                        )
                        for i in range(len(hist[0]))
                    )
                else:
                    hist = torch.stack(hist, dim=0)
                    convolution_sum = torch.tensordot(b_vals, hist, dims=([0], [0]))

                # Final update step
                # CHANGED: Use in-place multiplication
                weight_term = _mul_inplace(convolution_sum, gamma_beta)
                adj_y = _add(adj_y0, weight_term)

                # Handle y update
                if yhistory is not None and k < N - 1:
                    y = _cast_state_like(yhistory[k + 1], y)
                elif yhistory is None:
                    hist = [_cast_state_dtype(hist_item, dtype_hi) for hist_item in fy_history[start_idx : k + 1]]
                    if _is_tuple(hist[0]):
                        y_convolution_sum = tuple(
                            torch.tensordot(
                                b_vals,
                                torch.stack([hist_item[i] for hist_item in hist], dim=0),
                                dims=([0], [0]),
                            )
                            for i in range(len(hist[0]))
                        )
                    else:
                        hist = torch.stack(hist, dim=0)
                        y_convolution_sum = torch.tensordot(b_vals, hist, dims=([0], [0]))

                    # CHANGED: Use in-place multiplication
                    y_weight_term = _mul_inplace(y_convolution_sum, gamma_beta)
                    y = _add(y0, y_weight_term)

                # Update parameter gradients - already using in-place operation, good!
                if adj_params and vjp_params:
                    for ap, vp in zip(adj_params, vjp_params):
                        ap.add_(vp, alpha=h)

        # release memory
        del fadj_history
        if yhistory is None:  # CHANGED: Only delete fy_history if it was created
            del fy_history
        return adj_y, adj_params


def forward_gl(func, y0, beta, tspan, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
            dtype_hi = y0[0].dtype
        else:
            device = y0.device
            dtype_hi = y0.dtype
        dtype_store = _state_storage_dtype(dtype_hi)

        c = torch.zeros(N + 1, dtype=dtype_hi, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + beta) / j) * c[j - 1]

        # CHANGED: Compute h^beta once outside the loop for efficiency
        h_power = torch.pow(h, beta)

        # CHANGED: Use y_current for clarity and consistency
        y_current = _clone(y0)
        y_history = _StateHistoryBuffer(y_current, N, dtype_store)
        y_history.set(0, y_current)

        # CHANGED: Loop range from range(1, N) to range(N - 1) to match correct algorithm
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time (not tspan[k] when k starts from 1)
            t_k = tspan[k]

            # CHANGED: Evaluate function at current time with current y (not future y)
            f_k = func(t_k, y_current)

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Fix summation indices and coefficients
            # The sum should be Σ c_{k+1-j} * y_j for j from start_idx to k
            for j in range(start_idx, k + 1):
                # CHANGED: Use correct coefficient index (k+1-j instead of j)
                coefficient_idx = k + 1 - j

                if convolution_sum is None:
                    convolution_sum = _multiply(c[coefficient_idx], y_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, y_history[j], c[coefficient_idx])

            # # CHANGED: Move h_power multiplication outside loop and use it here
            # f_h_term = _multiply(h_power, f_k)
            # # Compute y_{k+1} = h^α * f(t_k, y_k) - convolution_sum
            # y_current = _minus(f_h_term, convolution_sum)

            #In-place fused multiply-add operation: target = -target + alpha * source
            y_current = _minusmul_inplace(convolution_sum, f_k, h_power)

            # Store y_{k+1} in history
            y_history.set(k + 1, y_current)

        return y_current, y_history  # CHANGED: Fixed - return y_current instead of yn

def backward_gl(func, y_aug, beta, tspan, yhistory, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        _, adj_y0, adj_params0 = y_aug  ### we will use yhistory rather than compute y again
        if _is_tuple(adj_y0):
            device = adj_y0[0].device
            dtype_hi = adj_y0[0].dtype
        else:
            device = adj_y0.device
            dtype_hi = adj_y0.dtype

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)

        c = torch.zeros(N + 1, dtype=dtype_hi, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + beta) / j) * c[j - 1]

        h_power = torch.pow(h, beta)

        # CHANGED: Use adj_y_current for clarity
        adj_y_current = _clone(adj_y0)
        adjy_history = [adj_y_current]

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Get the corresponding y from history at current time
            y_current = _cast_state_like(yhistory[k], adj_y_current)

            # CHANGED: Evaluate function at current time with current states
            func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Fix summation indices and coefficients
            # The sum should be Σ c_{k+1-j} * adjy_j for j from start_idx to k
            for j in range(start_idx, k + 1):
                # CHANGED: Use correct coefficient index (k+1-j instead of j)
                coefficient_idx = k + 1 - j

                if convolution_sum is None:
                    convolution_sum = _multiply(c[coefficient_idx], adjy_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, adjy_history[j], c[coefficient_idx])

            # Compute adj_y_{k+1} = h^α * vjp_y - convolution_sum
            # f_h_term = _multiply(h_power, vjp_y)
            # adj_y_current = _minus(f_h_term, convolution_sum)
            adj_y_current = _minusmul_inplace(convolution_sum, vjp_y, h_power)

            # Store adj_y_{k+1} in history
            adjy_history.append(adj_y_current)

            # Update parameter gradients - already using in-place operation, good!
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # CHANGED: Return adj_y_current instead of adj_y
        del adjy_history, yhistory
        return adj_y_current, adj_params


def forward_trap(func, y0, beta, tspan, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)

        # CHANGED: Pre-compute h^beta * Gamma(2-beta) for efficiency
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
            dtype_hi = y0[0].dtype
        else:
            device = y0.device
            dtype_hi = y0.dtype
        dtype_store = _state_storage_dtype(dtype_hi)

        # CHANGED: Removed unused c array computation
        # CHANGED: Use y_current for clarity
        y_current = _clone(y0)
        y_history = _StateHistoryBuffer(y_current, N, dtype_store)
        y_history.set(0, y_current)

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Evaluate function at current time with current y
            f_k = func(t_k, y_current)

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # CHANGED: Compute A_{j,k+1} weights correctly instead of RLcoeffs
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device)

            # Compute A_{j,k+1} weights
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)

            # General formula for j >= 1
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # CHANGED: Special handling for j=0 if it's in the range
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype_hi, device=device), one_minus_beta)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype_hi, device=device), -beta)
                A_j_kp1[0] = k_power - (k + beta) * kp1_neg_alpha

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Accumulate with correct indexing
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into A_j_kp1 array

                if convolution_sum is None:
                    convolution_sum = _multiply(A_j_kp1[local_idx], y_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, y_history[j], A_j_kp1[local_idx])

            # # CHANGED: Compute y_{k+1} correctly
            # f_term = _multiply(h_alpha_gamma, f_k)
            # # CHANGED: Use _minus or multiply by -1 properly
            # y_current = _minus(f_term, convolution_sum)

            # In-place fused multiply-add operation: target = -target + alpha * source
            y_current = _minusmul_inplace(convolution_sum, f_k, h_alpha_gamma)


            # Store y_{k+1} in history
            y_history.set(k + 1, y_current)

        return y_current, y_history  # CHANGED: Return y_current instead of yn


def backward_trap(func, y_aug, beta, tspan, yhistory_ori, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)

        # CHANGED: Pre-compute h^beta * Gamma(2-beta) for efficiency
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        _, adj_y0, adj_params0 = y_aug  ### we will use yhistory_ori rather than compute y again
        if _is_tuple(adj_y0):
            device = adj_y0[0].device
            dtype_hi = adj_y0[0].dtype
        else:
            device = adj_y0.device
            dtype_hi = adj_y0.dtype

        adj_params = _clone(adj_params0)

        # CHANGED: Removed unused c array computation
        # CHANGED: Use adj_y_current for clarity
        adj_y_current = _clone(adj_y0)
        adjy_history = [adj_y_current]

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Get the corresponding y from history at current time
            y_current = _cast_state_like(yhistory_ori[k], adj_y_current)

            # CHANGED: Evaluate function at current time with current states
            func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: Compute A_{j,k+1} weights correctly instead of RLcoeffs
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device)

            # Compute A_{j,k+1} weights (same as forward_trap)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)

            # General formula for j >= 1
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # CHANGED: Special handling for j=0 if it's in the range
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype_hi, device=device), one_minus_beta)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype_hi, device=device), -beta)
                A_j_kp1[0] = k_power - (k + beta) * kp1_neg_alpha

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Accumulate with correct indexing
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into A_j_kp1 array

                if convolution_sum is None:
                    convolution_sum = _multiply(A_j_kp1[local_idx], adjy_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, adjy_history[j], A_j_kp1[local_idx])

            # Compute adj_y_{k+1} = Γ(2-α) * h^α * vjp_y - convolution_sum
            # f_h_term = _multiply(h_alpha_gamma, vjp_y)
            # adj_y_current = _minus(f_h_term, convolution_sum)

            adj_y_current = _minusmul_inplace(convolution_sum, vjp_y, h_alpha_gamma)


            # Store adj_y_{k+1} in history
            adjy_history.append(adj_y_current)

            # Update parameter gradients - already using in-place operation, good!
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # CHANGED: Add memory cleanup
        del adjy_history, yhistory_ori
        # CHANGED: Return adj_y_current instead of adj_y
        return adj_y_current, adj_params

def forward_l1(func, y0, beta, tspan, **options):
    """Use L1 method to integrate Caputo equation (forward pass)
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          func: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
          yhistory: List of all computed y values
        """
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
            dtype_hi = y0[0].dtype
        else:
            device = y0.device
            dtype_hi = y0.dtype
        dtype_store = _state_storage_dtype(dtype_hi)

        y_current = _clone(y0)
        yhistory = _StateHistoryBuffer(y_current, N, dtype_store)
        yhistory.set(0, y_current)

        for k in range(N - 1):
            # Current time point t_k
            t_k = tspan[k]
            # Evaluate f(t_k, y_k)
            f_k = func(t_k, y_current)

            # Determine memory range
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # Vectorized computation of c_j^(k) weights for indices from start_idx to k
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device)

            # Compute c_j^(k) for all j values
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)

            c_j_k = kjp2 - 2 * kjp1 + kj

            # Special handling for j=0 if it's in the range
            if start_idx == 0:
                c_j_k[0] = -(torch.pow(torch.tensor(k + 1, dtype=dtype_hi, device=device), one_minus_beta) -
                             torch.pow(torch.tensor(k, dtype=dtype_hi, device=device), one_minus_beta))

            # Initialize accumulator for the sum
            convolution_sum = None

            # Accumulate: sum from j=start_idx to k
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into c_j_k array

                if convolution_sum is None:
                    convolution_sum = _multiply(c_j_k[local_idx], yhistory[j])
                else:
                    # Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, yhistory[j], c_j_k[local_idx])

            # Compute y_{k+1} = h^α * Γ(2-α) * f(t_k, y_k) - sum
            # f_term = _multiply(h_alpha_gamma, f_k)
            # y_current = _minus(f_term, convolution_sum)

            y_current = _minusmul_inplace(convolution_sum, f_k, h_alpha_gamma)


            # Store y_{k+1} in history
            yhistory.set(k + 1, y_current)

        return y_current, yhistory


def backward_l1(func, y_aug, beta, tspan, yhistory_ori, **options):
    """Use L1 method for backward pass (adjoint computation)
        Args:
          func: callable returning (func_eval, vjp_y, vjp_params)
          y_aug: tuple of (y0, adj_y0, adj_params0)
          beta: fractional exponent in the range (0,1)
          tspan: time points array
          yhistory_ori: forward pass y history
          options: additional options including memory
        Returns:
          adj_y: adjoint of y
          adj_params: adjoint of parameters
        """
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        _, adj_y0, adj_params0 = y_aug  # we will use yhistory_ori rather than compute y again

        if _is_tuple(adj_y0):
            device = adj_y0[0].device
            dtype_hi = adj_y0[0].dtype
        else:
            device = adj_y0.device
            dtype_hi = adj_y0.dtype

        adj_params = _clone(adj_params0)
        adj_y_current = _clone(adj_y0)
        adjy_history = [adj_y_current]

        for k in range(N - 1):
            # Current time point t_k
            t_k = tspan[k]

            # Get the corresponding y from history at current time
            y_current = _cast_state_like(yhistory_ori[k], adj_y_current)

            # Evaluate function at current time with current states
            func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))

            # Determine memory range
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # Vectorized computation of c_j^(k) weights (same as forward)
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype_hi, device=device)

            # Compute c_j^(k) for all j values
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)

            c_j_k = kjp2 - 2 * kjp1 + kj

            # Special handling for j=0 if it's in the range
            if start_idx == 0:
                c_j_k[0] = -(torch.pow(torch.tensor(k + 1, dtype=dtype_hi, device=device), one_minus_beta) -
                             torch.pow(torch.tensor(k, dtype=dtype_hi, device=device), one_minus_beta))

            # Initialize accumulator for the sum
            convolution_sum = None

            # Accumulate: sum from j=start_idx to k
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into c_j_k array

                if convolution_sum is None:
                    convolution_sum = _multiply(c_j_k[local_idx], adjy_history[j])
                else:
                    # Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, adjy_history[j], c_j_k[local_idx])

            # Compute adj_y_{k+1} = h^α * Γ(2-α) * vjp_y - sum
            # f_h_term = _multiply(h_alpha_gamma, vjp_y)
            # adj_y_current = _minus(f_h_term, convolution_sum)

            adj_y_current = _minusmul_inplace(convolution_sum, vjp_y, h_alpha_gamma)

            # Store adj_y_{k+1} in history
            adjy_history.append(adj_y_current)

            # Update parameter gradients using in-place operation
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # Release memory
        del adjy_history, yhistory_ori, c_j_k
        return adj_y_current, adj_params


def backward_euler_w_history(func, y_aug, beta, tspan, yhistory, **options):
    with torch.no_grad():
        N = len(tspan)
        # print('N = len(tspan)', N, tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        y0, adj_y0, adj_params0 = y_aug  ### we will use yhistory_ori rather than compute y again

        if _is_tuple(adj_y0):
            device = adj_y0[0].device
        else:
            device = adj_y0.device

        gamma_beta = 1 / math.gamma(beta)

        if True:#yhistory_ori is None:
            fy_history = []

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)
        y = _clone(y0)

        # return tuple(y_i.clone() for y_i in adj_y0), tuple(y_i.clone() for y_i in adj_params0)

        for k in range(N-1):
            tn = tspan[k]


            func_eval, vjp_y, vjp_params = func(tn, (y, adj_y, adj_params))
            y = _cast_state_like(yhistory[k + 1], adj_y)

            ## We assume having the full yhistory
            ## We do not consider the following case any more.
            # if yhistory is not None and k<N:
            #     y = yhistory[k+1]
            # else:
            #     fy_history.append(func_eval)
            #     j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
            #     b_j_k_1 = (torch.pow(h, beta) / beta) * (
            #             torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))
            #
            #     # Initialize accumulator with correct structure (tensor or tuple)
            #     if _is_tuple(fy_history[0]):
            #         b_all_k = tuple(torch.zeros_like(f_i) for f_i in fy_history[0])
            #     else:
            #         b_all_k = torch.zeros_like(fy_history[0])
            #
            #     # Loop through the range and accumulate results
            #     for i in range(0, k + 1):
            #         b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fy_history[i]))
            #
            #     # Final update step
            #     weight_term = _multiply(gamma_beta, b_all_k)
            #     y = _add(y0, weight_term)

            # adj_y = _add(adj_y, _multiply(h, vjp_y))
            adj_y = _addmul_inplace(adj_y, vjp_y, h)


            # Update parameter gradients using tuple comprehension
            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del yhistory, fy_history
    return adj_y, adj_params

def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())



# forward_gl_compiled = torch.compile(forward_gl)
# backward_gl_compiled = torch.compile(backward_gl)
# forward_predictor_compiled = torch.compile(forward_predictor)
# backward_predictor_compiled = torch.compile(backward_predictor)
# forward_trap_compiled = torch.compile(forward_trap)
# backward_trap_compiled = torch.compile(backward_trap)


forward_gl_compiled = forward_gl#torch.compile(forward_gl)
backward_gl_compiled = backward_gl#torch.compile(backward_gl)
forward_trap_compiled = forward_trap#torch.compile(forward_trap)
backward_trap_compiled = backward_trap#torch.compile(backward_trap)

forward_predictor_compiled = forward_predictor#torch.compile(forward_predictor)
backward_predictor_compiled = backward_predictor#torch.compile(backward_predictor)
forward_l1_compiled = forward_l1#torch.compile(forward_predictor)
backward_l1_compiled = backward_l1#torch.compile(backward_predictor)

backward_euler_w_history_compiled = backward_euler_w_history#torch.compile(backward_euler_w_history)



SOLVERS_Forward = {
            "predictor-f":forward_predictor_compiled,
           "predictor-o":forward_predictor_compiled,
           "gl-f":forward_gl_compiled,
           "gl-o":forward_gl_compiled,
           "trap-f":forward_trap_compiled,
           "trap-o":forward_trap_compiled,
            "l1-f":forward_l1_compiled,
            "l1-o":forward_l1_compiled,
            # "euler":forward_euler_w_history_compiled,
}

SOLVERS_Backward = {"predictor-f":backward_predictor_compiled,
           "predictor-o":backward_euler_w_history_compiled,
           "gl-f":backward_gl_compiled,
           "gl-o":backward_euler_w_history_compiled,
           "trap-f":backward_trap_compiled,
           "trap-o":backward_euler_w_history_compiled,
            "l1-f":backward_l1_compiled,
            "l1-o":backward_euler_w_history_compiled,
            # "euler": backward_euler_w_history_compiled,
}
