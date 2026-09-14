
import math
import os
import random
import unittest
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pymittagleffler import mittag_leffler

# Pylance may not resolve rampde if it is not in the IDE's configured venv.
# The tests run correctly from the rampde/ directory with the package installed.
from rampde import predictor_fdeint, DynamicScaler  # type: ignore[import]
from rampde import (  # type: ignore[import]
    PredictorFDESolverUnscaled,
    PredictorFDESolverDynamic,
    PredictorFDESolverUnscaledSafe,
)
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

def march_trajectory(y0, beta, T, step_size, lam):
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
            dtype_low=torch.float16,
            graded_time=False,
        )

    return yt

def test_upper_range():
    """
    Test the upper range of the FDE solver for float16.
    """

    y0 = torch.tensor([65504.0 / 180.0], dtype=torch.float32)
    beta = torch.tensor([0.7], dtype=torch.float32)
    T = torch.tensor([20.0], dtype=torch.float32)
    step_size = 0.01
    lam = 199.0

    ys = march_trajectory(y0, beta, T, step_size, lam)

    return ys

def test_lower_range():
    """
    Test the lower range of the FDE solver for float16.
    """

    y0 = torch.tensor([1.0], dtype=torch.float32)
    beta = torch.tensor([0.9], dtype=torch.float32)
    T = torch.tensor([20.0], dtype=torch.float32)
    step_size = 0.01
    lam = 199.0

    ys = march_trajectory(y0, beta, T, step_size, lam)

    return ys

def upper_analytical():
    """
    Analytical solution for the upper range test.
    """
    y0 = 65504.0 / 180.0
    beta = 0.7
    T = 20.0
    lam = 199.0

    tspan = np.linspace(0.0, T, int(round(T / 0.1)) + 1)
    analytical_solution = y0 * mittag_leffler(-lam * tspan**beta, beta, 1)

    analytical_deiv = -lam * analytical_solution

    return analytical_solution, analytical_deiv

def lower_analytical():
    """
    Analytical solution for the lower range test.
    """
    y0 = 1.0
    beta = 0.9
    T = 20.0
    lam = 199.0

    tspan = np.linspace(0.0, T, int(round(T / 0.1)) + 1)
    analytical_solution = y0 * mittag_leffler(-lam * tspan**beta, beta, 1)

    analytical_deiv = -lam * analytical_solution

    return analytical_solution, analytical_deiv



def to_numpy(x):
    return x.detach().float().cpu().numpy()

def make_plot(upper_ys, upper_derv, lower_ys, lower_derv, analytical_upper_soln, analytical_lower_soln):
    import matplotlib.pyplot as plt

    tspan = np.linspace(0.0, 20.0, int(round(20.0 / 0.1)) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(tspan, to_numpy(upper_ys), label='Upper Range Solution', color='blue')
    plt.plot(tspan, analytical_upper_soln, label='Upper Range Analytical', color='black', linestyle='dashed')
    plt.plot(tspan, abs(upper_derv), label='Upper Range Derivative', color='orange')
    plt.plot(tspan, to_numpy(lower_ys), label='Lower Range Solution', color='green')
    plt.plot(tspan, analytical_lower_soln, label='Lower Range Analytical', color='black', linestyle='dashed')
    plt.plot(tspan, abs(lower_derv), label='Lower Range Derivative', color='red')
    plt.title('FDE Solution and Derivative Ranges')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('fde_solution_derivative_ranges.png')
    

if __name__ == "__main__":
    upper_ys = test_upper_range()
    lower_ys = test_lower_range()

    analytical_upper_soln, analytical_upper_derv = upper_analytical()
    analytical_lower_soln, analytical_lower_derv = lower_analytical()

    make_plot(upper_ys, analytical_upper_derv, lower_ys, analytical_lower_derv, analytical_upper_soln, analytical_lower_soln)
    
