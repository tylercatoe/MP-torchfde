#!/usr/bin/env python
"""
Memory and speed benchmark: rampde.fdeint (FP16 MP) vs torchfde.fdeint (FP32).

Mirrors test_speed.py but for the FDE case.

Key differences from the ODE benchmark:
- Baseline is torchfde.fdeint (standard autograd, FP32) instead of torchdiffeq
- The FDE L1 scheme is O(N²): fewer steps and smaller dims than ODE test
- Memory savings come from two sources:
    1. yt stored in FP16 instead of FP32 (rampde custom adjoint)
    2. Custom adjoint avoids storing the full autograd graph — torchfde standard
       autograd retains ALL N steps' intermediate MLP activations simultaneously
       during backward; rampde custom adjoint recomputes one step at a time.
    3. adj_buf stored in float16 (adj_dtype=float16) instead of float32.

The model must be an MLP (not a linear layer) so that per-step activation memory
is non-trivial. For a linear layer torchfde's graph is tiny (no hidden activations),
so rampde's adj_buf overhead dominates and savings are negative.

Expected savings: ≥ 40% peak memory, speedup varies by hardware.

Note: Requires CUDA and torchfde (from MP-torchfde directory).
"""

import os
import sys
import time
import unittest

import torch
import torch.nn as nn

# MP-torchfde lives one level above the rampde package root
_TORCHFDE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "MP-torchfde")
)
if _TORCHFDE_DIR not in sys.path:
    sys.path.insert(0, _TORCHFDE_DIR)

from rampde import fdeint as rampde_fdeint  # type: ignore[import]

try:
    from torchfde import fdeint as torchfde_fdeint  # type: ignore[import]
    TORCHFDE_AVAILABLE = True
except ImportError:
    TORCHFDE_AVAILABLE = False

QUIET = os.environ.get("RAMPDE_TEST_QUIET", "0") == "1"


# ---------------------------------------------------------------------------
# Shared model
# ---------------------------------------------------------------------------

class MLPFDE(nn.Module):
    """MLP-valued fractional derivative: D^β y = MLP(y).

    Uses Tanh activations so that per-step activation tensors are non-trivial.
    This is what makes the comparison meaningful: torchfde's standard autograd
    must keep ALL N steps' activations alive in the graph during backward, while
    rampde's custom adjoint recomputes one step at a time from adj_buf.
    """

    def __init__(self, dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )
        self.to(device)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _measure(func_call, device, warmup: int = 2):
    """Run warmup then one timed, memory-measured forward+backward pass."""
    for _ in range(warmup):
        y_T, loss = func_call()
        loss.backward()
        torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    y_T, loss = func_call()
    torch.cuda.synchronize(device)
    t_fwd_end = time.perf_counter()

    # Record forward-only time before backward
    t_fwd_start = getattr(_measure, "_t0", t_fwd_end)  # set below

    torch.cuda.synchronize(device)
    t_bwd_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    t_bwd_end = time.perf_counter()

    peak_mem = torch.cuda.max_memory_allocated(device)
    return t_bwd_end - t_fwd_start, peak_mem  # total time, peak memory


def _bench(solver_fn, y0, beta, t_end, step_size, device, warmup=2, use_fp16=False,
           hidden_dim=512):
    """
    Run a timed + memory-measured forward+backward for a given solver.

    solver_fn: callable(func, y0, beta, t_end, step_size) → y_T tensor
    use_fp16:  wrap in autocast(float16) and pass loss_scaler=False
    hidden_dim: MLP hidden layer width (controls per-step activation memory)
    """
    dim = y0.shape[-1]
    func = MLPFDE(dim, hidden_dim, device)

    def _call():
        y = y0.detach().requires_grad_(True)
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_T = solver_fn(func, y, beta, t_end, step_size)
        else:
            y_T = solver_fn(func, y, beta, t_end, step_size)
        return y_T, y_T.float().sum()

    # Warmup
    for _ in range(warmup):
        y_T, loss = _call()
        loss.backward()
        torch.cuda.synchronize(device)

    # Timed run
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t_start = time.perf_counter()
    y_T, loss = _call()
    torch.cuda.synchronize(device)
    t_fwd = time.perf_counter()

    loss.backward()
    torch.cuda.synchronize(device)
    t_end_bwd = time.perf_counter()

    peak_mem = torch.cuda.max_memory_allocated(device)
    fwd_time = t_fwd - t_start
    bwd_time = t_end_bwd - t_fwd

    return fwd_time, bwd_time, peak_mem


def _torchfde_solver(func, y0, beta, t_end, step_size):
    """Thin wrapper so torchfde matches the common solver signature."""
    return torchfde_fdeint(func, y0, beta, t_end, step_size, method="l1")


def _rampde_solver(func, y0, beta, t_end, step_size):
    """rampde solver: FP16 forward + adj_buf stored in FP16 (adj_dtype=float16).

    adj_dtype=float16 reduces adj_buf from 4B → 2B per element, removing the
    memory overhead that would otherwise negate the FP16 trajectory savings.
    """
    return rampde_fdeint(
        func, y0, beta, t_end, step_size,
        loss_scaler=False,
        adj_dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class FDESpeedTest(unittest.TestCase):
    """
    Benchmark rampde.fdeint (FP16 autocast + adj_dtype=float16) vs torchfde.fdeint (FP32).

    Memory savings come from three compounding effects:
      1. FP16 forward trajectory (yt): 2B vs 4B per element.
      2. FP16 adjoint buffer (adj_buf via adj_dtype=float16): 2B vs 4B per element.
      3. Custom adjoint avoids retaining the full autograd graph — torchfde's
         standard autograd holds ALL N steps' MLP activations alive simultaneously
         during backward; rampde recomputes one step at a time.

    NOTE: The model must be an MLP (not a simple linear layer). For a linear
    layer, torchfde's graph stores only the y history (no hidden activations),
    so its graph is tiny and rampde's adj_buf overhead would dominate.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available — skipping FDE speed tests")
        if not TORCHFDE_AVAILABLE:
            raise unittest.SkipTest(
                f"torchfde not found at {_TORCHFDE_DIR} — skipping FDE speed tests"
            )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(42)

        cls.device = "cuda"
        cls.beta = 0.7
        cls.t_end = 5.0
        cls.step_size = 0.1          # N = 50 steps
        cls.batch_size = 512
        cls.dimensions = [256, 128]  # MLP input/output dim (hidden_dim is 4×dim)
        cls.hidden_dim = 512         # hidden width: dominates per-step activation cost

        # Minimum memory saving we require.
        # With MLP + adj_dtype=float16, empirically ≥ 40-50%.
        cls.min_mem_saving_pct = 40.0

    def setUp(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def tearDown(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _make_y0(self, dim):
        return torch.randn(
            self.batch_size, dim, device=self.device, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Memory saving test
    # ------------------------------------------------------------------

    def test_memory_saving(self):
        """rampde FP16 + adj_dtype=float16 must use ≥ min_mem_saving_pct% less peak memory."""
        rows = []

        for dim in self.dimensions:
            torch.cuda.empty_cache()
            y0 = self._make_y0(dim)

            # torchfde FP32 baseline
            torch.cuda.empty_cache()
            _, _, mem_fp32 = _bench(
                _torchfde_solver, y0, self.beta, self.t_end, self.step_size,
                self.device, use_fp16=False, hidden_dim=self.hidden_dim,
            )

            # rampde FP16 (autocast + adj_dtype=float16)
            torch.cuda.empty_cache()
            _, _, mem_fp16 = _bench(
                _rampde_solver, y0, self.beta, self.t_end, self.step_size,
                self.device, use_fp16=True, hidden_dim=self.hidden_dim,
            )

            saving_pct = 100.0 * (mem_fp32 - mem_fp16) / mem_fp32
            rows.append({
                "dim": dim,
                "fp32_mb": mem_fp32 / 1e6,
                "fp16_mb": mem_fp16 / 1e6,
                "saving_pct": saving_pct,
            })

            if not QUIET:
                print(
                    f"  dim={dim:5d}: torchfde FP32 = {mem_fp32/1e6:7.1f} MB  "
                    f"rampde FP16 = {mem_fp16/1e6:7.1f} MB  "
                    f"saving = {saving_pct:.1f}%"
                )

            self.assertGreater(
                saving_pct, self.min_mem_saving_pct,
                f"dim={dim}: expected ≥{self.min_mem_saving_pct}% memory saving, "
                f"got {saving_pct:.1f}%",
            )

        if not QUIET:
            print(
                f"\n  {'Dim':<8} {'FP32 (MB)':<14} {'FP16 (MB)':<14} {'Saving %'}"
            )
            print("  " + "-" * 44)
            for r in rows:
                print(
                    f"  {r['dim']:<8} {r['fp32_mb']:<14.1f} {r['fp16_mb']:<14.1f} "
                    f"{r['saving_pct']:.1f}%"
                )

    # ------------------------------------------------------------------
    # Speed comparison (informational — no hard assertion on speedup
    # since timings are noisy on shared hardware, but we print the table)
    # ------------------------------------------------------------------

    def test_speed_comparison(self):
        """Report forward/backward speedup of rampde FP16 vs torchfde FP32."""
        rows = []

        for dim in self.dimensions:
            torch.cuda.empty_cache()
            y0 = self._make_y0(dim)

            torch.cuda.empty_cache()
            fwd32, bwd32, _ = _bench(
                _torchfde_solver, y0, self.beta, self.t_end, self.step_size,
                self.device, use_fp16=False, hidden_dim=self.hidden_dim,
            )

            torch.cuda.empty_cache()
            fwd16, bwd16, _ = _bench(
                _rampde_solver, y0, self.beta, self.t_end, self.step_size,
                self.device, use_fp16=True, hidden_dim=self.hidden_dim,
            )

            total32 = fwd32 + bwd32
            total16 = fwd16 + bwd16
            speedup = total32 / max(total16, 1e-9)
            rows.append({
                "dim": dim,
                "fwd32": fwd32, "bwd32": bwd32,
                "fwd16": fwd16, "bwd16": bwd16,
                "speedup": speedup,
            })

            if not QUIET:
                print(
                    f"  dim={dim:5d}: torchfde FP32 fwd={fwd32:.4f}s bwd={bwd32:.4f}s  "
                    f"rampde FP16 fwd={fwd16:.4f}s bwd={bwd16:.4f}s  "
                    f"speedup={speedup:.2f}x"
                )

        if not QUIET:
            print(
                f"\n  {'Dim':<8} {'FP32 total (s)':<16} {'FP16 total (s)':<16} {'Speedup'}"
            )
            print("  " + "-" * 50)
            for r in rows:
                print(
                    f"  {r['dim']:<8} {r['fwd32']+r['bwd32']:<16.4f} "
                    f"{r['fwd16']+r['bwd16']:<16.4f} {r['speedup']:.2f}x"
                )

        # Sanity: at least the larger dimension should show some speedup
        if rows:
            largest = max(rows, key=lambda r: r["dim"])
            self.assertGreater(
                largest["speedup"], 0.5,
                "rampde FP16 should not be more than 2x slower than torchfde FP32 "
                f"even on this hardware (got {largest['speedup']:.2f}x)",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
