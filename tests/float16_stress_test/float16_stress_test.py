
import numpy as np
import argparse
import torch
import torch.nn as nn
from pymittagleffler import mittag_leffler

# Pylance may not resolve rampde if it is not in the IDE's configured venv.
# The tests run correctly from the rampde/ directory with the package installed.
from rampde import predictor_fdeint, DynamicScaler  # type: ignore[import]
from rampde.predictor_fdeint import _predictor_forward_impl


# ---------------------------------------------------------------------------
# Forcing function
# ---------------------------------------------------------------------------

class Forcing(nn.Module):
    """

    f(t, y) = -λy(t) 
    
    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, t, y):
        return -self.lam * y

def march_trajectory(y0, beta, T, step_size, lam, dtype: torch.dtype):
    device = torch.device("cuda")

    y0 = y0.to(device)
    beta_val = float(beta.item())
    T_val = float(T.item())

    n_steps = int(round(T_val / step_size)) + 1
    tspan = torch.linspace(
        0.0,
        T_val,
        n_steps,
        dtype=torch.float32,
        device=device,
    )

    func = Forcing(lam=lam).to(device)

    with torch.no_grad():
        y_T, yt, _ = _predictor_forward_impl(
            func,
            y0,
            tspan,
            beta_val,
            dtype_hi=torch.float32,
            dtype_low=dtype,
            graded_time=False,
        )

    return yt

def test_upper_range(dtype: torch.dtype = torch.float16, T: float = 8.0, step_size: float = 0.01):
    """
    Test the upper range of the FDE solver for float16.
    """

    y0 = torch.tensor([65504.0 / 200.0], dtype=torch.float32)
    beta = torch.tensor([0.7], dtype=torch.float32)
    T = torch.tensor([T], dtype=torch.float32)
    lam = 199.0

    ys = march_trajectory(y0, beta, T, step_size, lam, dtype=dtype)

    return ys

def test_lower_range(dtype: torch.dtype = torch.float16, T: float = 8.0, step_size: float = 0.01):
    """
    Test the lower range of the FDE solver for float16.
    """

    y0 = torch.tensor([1.0], dtype=torch.float32)
    beta = torch.tensor([0.9], dtype=torch.float32)
    T = torch.tensor([T], dtype=torch.float32)
    lam = 199.0

    ys = march_trajectory(y0, beta, T, step_size, lam, dtype=dtype)

    return ys

def upper_analytical(T, step_size):
    """
    Analytical solution for the upper range test.
    """
    y0 = 65504.0 / 200.0
    beta = 0.7
    lam = 199.0

    tspan = np.linspace(0.0, T, int(round(T / step_size)) + 1)
    analytical_solution = y0 * mittag_leffler(-lam * tspan**beta, beta, 1)

    analytical_deiv = -lam * analytical_solution

    return analytical_solution, analytical_deiv

def lower_analytical(T, step_size):
    """
    Analytical solution for the lower range test.
    """
    y0 = 1.0
    beta = 0.9
    lam = 199.0

    tspan = np.linspace(0.0, T, int(round(T / step_size)) + 1)
    analytical_solution = y0 * mittag_leffler(-lam * tspan**beta, beta, 1)

    analytical_deiv = -lam * analytical_solution

    return analytical_solution, analytical_deiv



def to_numpy(x):
    return x.detach().float().cpu().numpy()

def make_plot(upper_ys, upper_derv, lower_ys, lower_derv, analytical_upper_soln, analytical_lower_soln, T, step_size):
    import matplotlib.pyplot as plt

    tspan = np.linspace(0.0, T, int(round(T / step_size)) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(tspan, to_numpy(upper_ys), label='UR Numerical Solution', color='blue')
    plt.plot(tspan, analytical_upper_soln, label='UR Analytical', color='black', linestyle='dashed')
    plt.plot(tspan, abs(upper_derv), label='UR Derivative', color='orange')
    plt.plot(tspan, to_numpy(lower_ys), label='LR Numerical Solution', color='green')
    plt.plot(tspan, analytical_lower_soln, label='LR Analytical', color='black', linestyle='dashed')
    plt.plot(tspan, abs(lower_derv), label='LR Derivative', color='red')
    plt.plot(tspan, np.full_like(tspan, 65504.0), label='Float16 Max', color='purple', linestyle='dotted')
    plt.plot(tspan, np.full_like(tspan, 6.1e-5), label='Float16 Min', color='brown', linestyle='dotted')
    plt.title('FDE Solution and Derivative Ranges')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('fde_solution_derivative_ranges.png')


def parse_args():
    parser = argparse.ArgumentParser(description="Float16 Stress Test for FDE Solver")
    parser.add_argument("--dtype", type=str, default="torch.float16", choices=["torch.float16", "torch.bfloat16", "torch.float32"], help="Data type for low precision computation")
    parser.add_argument("--T", type=float, default=8.0, help="Final time for the simulation")
    parser.add_argument("--step_size", type=float, default=0.01, help="Step size for the simulation")
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
    T = args.T
    step_size = args.step_size
    dtype = get_dtype(args.dtype)

    upper_ys = test_upper_range(dtype, T=T, step_size=step_size)
    lower_ys = test_lower_range(dtype, T=T, step_size=step_size)

    analytical_upper_soln, analytical_upper_derv = upper_analytical(T=T, step_size=step_size)
    analytical_lower_soln, analytical_lower_derv = lower_analytical(T=T, step_size=step_size)

    make_plot(upper_ys, analytical_upper_derv, lower_ys, analytical_lower_derv, analytical_upper_soln, analytical_lower_soln, T, step_size)
    
