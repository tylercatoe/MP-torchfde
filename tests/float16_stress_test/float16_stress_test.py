import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Prefer the package in this checkout over any unrelated installed ``rampde``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "rampfde"))

from rampde import predictor_fdeint
from rampde.predictor_fdeint import _predictor_forward_impl


def evaluate_mittag_leffler(z, beta):
    """Evaluate E_beta(z), with a useful error if the optional package is absent."""
    try:
        from pymittagleffler import mittag_leffler
    except ImportError as error:
        raise RuntimeError(
            "The trajectory plot requires the 'pymittagleffler' package."
        ) from error
    return np.real_if_close(mittag_leffler(z, beta, 1)).real


# ---------------------------------------------------------------------------
# Forcing function
# ---------------------------------------------------------------------------

class Forcing(nn.Module):
    """The scalar test equation ``f(t, y) = -lambda*y(t)``."""

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, t, y):
        # Elementwise multiplication is not guaranteed to be autocast to the
        # requested CUDA dtype, so cast explicitly for this precision test.
        if torch.is_autocast_enabled():
            low_dtype = torch.get_autocast_dtype("cuda")
            y = y.to(low_dtype)
            lam = torch.as_tensor(self.lam, dtype=low_dtype, device=y.device)
        else:
            lam = torch.as_tensor(self.lam, dtype=y.dtype, device=y.device)
        return -lam * y


def make_tspan(T, step_size, beta, device, graded_time):
    """Build the same uniform or symmetric graded mesh as predictor_fdeint."""
    if not 0.0 < step_size < T:
        raise ValueError(f"step_size must be in (0, T), got {step_size}")

    n_steps = int(round(T / step_size)) + 1
    if not graded_time:
        return torch.linspace(0.0, T, n_steps, dtype=torch.float32, device=device)

    index = torch.arange(n_steps, dtype=torch.float32, device=device)
    center = (n_steps - 1) / 2
    q = 1 - torch.abs(index - center) / center
    grading_power = (2.0 - beta) / beta
    return T / 2 * (
        1 + torch.sign(index - center) * (1 - torch.pow(q, grading_power))
    )


def march_trajectory(
    y0,
    beta,
    T,
    step_size,
    lam,
    dtype: torch.dtype,
    device: torch.device,
    graded_time: bool,
    predictor_corrector: bool,
):

    y0 = y0.to(device)
    beta_val = float(beta.item())
    T_val = float(T.item())

    tspan = make_tspan(T_val, step_size, beta_val, device, graded_time)

    func = Forcing(lam=lam).to(device)
    autocast_context = (
        nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type="cuda", dtype=dtype)
    )

    with torch.no_grad():
        y_T_internal, yt, _ = _predictor_forward_impl(
            func,
            y0,
            tspan,
            beta_val,
            dtype_hi=torch.float32,
            dtype_low=dtype,
            graded_time=graded_time,
            predictor_corrector=predictor_corrector,
        )

        # Exercise the supported public API as well. The private helper above
        # is used only because the public API intentionally returns y(T), not
        # the complete trajectory needed by this plot.
        with autocast_context:
            y_T_public = predictor_fdeint(
                func,
                y0,
                beta=beta_val,
                t=T_val,
                step_size=step_size,
                graded_time=graded_time,
                predictor_corrector=predictor_corrector,
            )

    if not torch.allclose(y_T_internal, y_T_public, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            "Internal trajectory and public predictor_fdeint terminal states disagree: "
            f"internal={y_T_internal}, public={y_T_public}"
        )

    return tspan, yt


def run_upper_range(
    dtype,
    T,
    step_size,
    device,
    graded_time,
    predictor_corrector,
):
    """
    Test the upper range of the FDE solver for float16.
    """

    # Keep both y and |D^beta y| near the upper float16 range without making
    # the explicit predictor unstable at the default step size.
    y0 = torch.tensor([50000.0], dtype=torch.float32)
    beta = torch.tensor([0.7], dtype=torch.float32)
    T = torch.tensor([T], dtype=torch.float32)
    lam = 1.0

    return march_trajectory(
        y0, beta, T, step_size, lam, dtype, device,
        graded_time, predictor_corrector,
    )


def run_lower_range(
    dtype,
    T,
    step_size,
    device,
    graded_time,
    predictor_corrector,
):
    """
    Test the lower range of the FDE solver for float16.
    """

    y0 = torch.tensor([.10], dtype=torch.float32)
    beta = torch.tensor([0.9], dtype=torch.float32)
    T = torch.tensor([T], dtype=torch.float32)
    lam = 20.0

    return march_trajectory(
        y0, beta, T, step_size, lam, dtype, device,
        graded_time, predictor_corrector,
    )


def upper_analytical(tspan):
    """
    Analytical solution for the upper range test.
    """
    y0 = 50000.0
    beta = 0.7
    lam = 1.0

    analytical_solution = y0 * evaluate_mittag_leffler(-lam * tspan**beta, beta)

    analytical_deriv = -lam * analytical_solution

    return analytical_solution, analytical_deriv


def lower_analytical(tspan):
    """
    Analytical solution for the lower range test.
    """
    y0 = 0.10
    beta = 0.9
    lam = 20.0

    analytical_solution = y0 * evaluate_mittag_leffler(-lam * tspan**beta, beta)

    analytical_deriv = -lam * analytical_solution

    return analytical_solution, analytical_deriv



def to_numpy(x):
    return x.detach().float().cpu().numpy()

def make_plot(
    upper_tspan,
    upper_ys,
    upper_deriv,
    lower_tspan,
    lower_ys,
    lower_deriv,
    analytical_upper_soln,
    analytical_lower_soln,
    output,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        upper_tspan,
        np.abs(to_numpy(upper_ys)).squeeze(),
        label="|Numerical Soln|",
        color="blue",
    )
    plt.plot(
        upper_tspan,
        np.abs(analytical_upper_soln),
        label="|Analytical Soln|",
        color="black",
        linestyle="dashed",
    )
    plt.plot(
        upper_tspan,
        np.abs(upper_deriv),
        label="|Caputo Derivative|",
        color="orange",
    )
    plt.axhline(65504.0, label="Float16 Max", color="purple", linestyle="dotted")
    plt.axhline(
        6.1035e-5,
        label="Float16 Min Normal",
        color="brown",
        linestyle="dotted",
    )
    plt.title('FDE Solution and Derivative Upper Ranges')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(
        lower_tspan,
        np.abs(to_numpy(lower_ys)).squeeze(),
        label="|Numerical Soln|",
        color="green",
    )
    plt.plot(
        lower_tspan,
        np.abs(analytical_lower_soln),
        label="|Analytical Soln|",
        color="black",
        linestyle="dashed",
    )
    plt.plot(
        lower_tspan,
        np.abs(lower_deriv),
        label="|Caputo Derivative|",
        color="red",
    )
    plt.axhline(65504.0, label="Float16 Max", color="purple", linestyle="dotted")
    plt.axhline(
        6.1035e-5,
        label="Float16 Min Normal",
        color="brown",
        linestyle="dotted",
    )
    plt.title('FDE Solution and Derivative Lower Ranges')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Float16 Stress Test for FDE Solver")
    parser.add_argument(
        "--dtype",
        default="torch.float16",
        choices=["torch.float16", "torch.bfloat16", "torch.float32"],
        help="Data type for low precision computation",
    )
    parser.add_argument(
        "--T", type=float, default=8.0, help="Final time for the simulation"
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.01,
        help="Step size for the simulation",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--graded-time", action="store_true", help="Use the symmetric graded mesh"
    )
    parser.add_argument(
        "--predictor-corrector",
        action="store_true",
        help="Use the predictor-corrector rule",
    )
    parser.add_argument(
        "--output",
        default="fde_solution_derivative_ranges.png",
        help="Output plot path",
    )
    return parser.parse_args()

def get_dtype(dtype_str):
    if dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.bfloat16":
        return torch.bfloat16
    elif dtype_str == "torch.float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    

if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")

    T = args.T
    step_size = args.step_size
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    upper_tspan, upper_ys = run_upper_range(
        dtype, T, step_size, device, args.graded_time, args.predictor_corrector
    )
    lower_tspan, lower_ys = run_lower_range(
        dtype, T, step_size, device, args.graded_time, args.predictor_corrector
    )

    upper_tspan = upper_tspan.detach().cpu().numpy()
    lower_tspan = lower_tspan.detach().cpu().numpy()

    analytical_upper_soln, analytical_upper_deriv = upper_analytical(
        upper_tspan
    )
    analytical_lower_soln, analytical_lower_deriv = lower_analytical(
        lower_tspan
    )

    make_plot(
        upper_tspan,
        upper_ys,
        analytical_upper_deriv,
        lower_tspan,
        lower_ys,
        analytical_lower_deriv,
        analytical_upper_soln,
        analytical_lower_soln,
        args.output,
    )
    print(f"Wrote {args.output}")
