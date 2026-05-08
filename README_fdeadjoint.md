# `fdeadjoint.py` Guide

This document explains how [`fdeadjoint.py`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py) works, where gradient calls happen, and where VJPs are computed/used.

## What This File Does

`fdeadjoint.py` implements memory-efficient backpropagation for FDE solves by:

1. Running the forward solve with history capture (under `torch.no_grad()`).
2. Defining a custom autograd node (`FDEAdjointMethod`).
3. Re-integrating an augmented adjoint system backward in time.
4. Computing vector-Jacobian products (VJPs) on demand instead of storing full forward graphs.

Main entrypoint:
- [`fdeint_adjoint(...)`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:10)

## High-Level Call Flow

1. `fdeint_adjoint` validates/wraps inputs and collects module parameters.
- Input normalization and method validation happen via `_check_inputs(...)` at [`fdeadjoint.py:34`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:34).
- Parameters are collected by `find_parameters(...)` at [`fdeadjoint.py:44`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:44).

2. It calls `FDEAdjointMethod.apply(...)`.
- See invocation at [`fdeadjoint.py:48`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:48).

3. `FDEAdjointMethod.forward` runs the chosen forward solver.
- Solver dispatch is through `SOLVERS_Forward[method]` at [`fdeadjoint.py:89`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:89).
- It stores `tspan`, final state `ans`, forward `yhistory`, `func`, `beta`, `method`, and `func_params` in `ctx` when gradients are needed (`fdeadjoint.py:91-105`).

4. `FDEAdjointMethod.backward` reconstructs gradient dynamics.
- It builds `AugDynamics` at [`fdeadjoint.py:144`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:144).
- It flips time with `tspan.flip(0)` at [`fdeadjoint.py:185`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:185) and reverses history with `ReversedListView` at [`fdeadjoint.py:188`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:188).
- It dispatches to `SOLVERS_Backward[method]` at [`fdeadjoint.py:203`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:203).

5. Backward returns gradients for `*y0` and `*params`.
- Return signature is at [`fdeadjoint.py:222`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:222).
- Gradients for `func`, `n_state`, `n_params`, `beta`, `tspan`, `method`, and `options` are returned as `None`.

## Where Gradient Calls Happen

There is one explicit autograd gradient call site in this file:

- VJP construction: [`torch.autograd.grad(...)` at `fdeadjoint.py:159-166`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:159)

Details:
- Outputs differentiated: `func_eval`
- Inputs differentiated: `y + self.f_params`
- Vector for VJP: `tuple(adj_y)`
- `allow_unused=True` so missing paths produce `None` (later zero-filled).

## Where VJPs Are Computed

VJPs are computed inside `AugDynamics.__call__`:

- State/parameter prep with grads enabled: [`fdeadjoint.py:153-156`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:153)
- VJP call: [`fdeadjoint.py:159-166`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:159)
- Split into state and parameter components:
  - `vjp_y` at [`fdeadjoint.py:168`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:168)
  - `vjp_params` at [`fdeadjoint.py:169`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:169)
- `None` gradients are replaced by zeros at:
  - [`fdeadjoint.py:172-175`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:172)
  - [`fdeadjoint.py:177-180`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:177)

## Where VJPs Are Consumed

Each backward solver repeatedly evaluates:
- `func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))`

Call sites:
- Predictor fixed-point backward: [`fdeadjoint.py:344`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:344)
- GL fixed-point backward: [`fdeadjoint.py:501`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:501)
- Trap fixed-point backward: [`fdeadjoint.py:659`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:659)
- L1 fixed-point backward: [`fdeadjoint.py:846`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:846)
- One-step backward (`-o` methods): [`fdeadjoint.py:932`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:932)

State-adjoint update from `vjp_y`:
- Predictor: [`fdeadjoint.py:363-364`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:363)
- GL: [`fdeadjoint.py:530`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:530)
- Trap: [`fdeadjoint.py:704`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:704)
- L1: [`fdeadjoint.py:889`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:889)
- One-step (`-o`): [`fdeadjoint.py:960`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:960)

Parameter-adjoint accumulation from `vjp_params`:
- Predictor: [`fdeadjoint.py:387-389`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:387)
- GL: [`fdeadjoint.py:536-538`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:536)
- Trap: [`fdeadjoint.py:711-713`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:711)
- L1: [`fdeadjoint.py:895-897`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:895)
- One-step (`-o`): [`fdeadjoint.py:965-967`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:965)

## Solver Routing For Adjoint Methods

Forward solver mapping:
- [`SOLVERS_Forward` at `fdeadjoint.py:1012-1022`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:1012)

Backward solver mapping:
- [`SOLVERS_Backward` at `fdeadjoint.py:1024-1033`](/Users/tylercatoe/Developer/repo-comparison/torchfde/torchfde/fdeadjoint.py:1024)

Notable behavior:
- `*-f` methods use method-specific fractional adjoint solvers (`backward_predictor`, `backward_gl`, `backward_trap`, `backward_l1`).
- `*-o` methods route to `backward_euler_w_history`, which applies a one-step adjoint update with stored forward history.

## Practical Notes

- `func` must be an `nn.Module` (`fdeadjoint.py:14-15`) so parameters can be discovered and differentiated.
- Forward solves run under `torch.no_grad()` for memory savings; gradients are recovered via VJPs in backward.
- For tensor inputs, `fdeint_adjoint` wraps/unpacks to keep a uniform tuple-based internal API.
