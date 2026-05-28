#!/usr/bin/env python
"""
STL10 comparison runner for direct backprop, adjoint, and mixed-precision adjoint.

This script follows the multi-mode benchmark flow from mp_fde_mnist.py while
keeping the network architecture close to rampde/paper/stl10/ode_stl10.py.
"""

import argparse
import logging
import os
import random
import time
#from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List#, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as transforms


# =============================================================================
# Configuration
# =============================================================================

DIRECT_METHODS = {
    'predictor',
    'l1',
    'gl',
    'trap',
    'glmulti',
}
ADJOINT_METHODS = {
    'predictor-f',
    'predictor-o',
    'l1-f',
    'l1-o',
    'gl-f',
    'gl-o',
    'trap-f',
    'trap-o',
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/benchmark FDE-based STL10 model with multiple backprop modes")

    # Training and Network Settings
    parser.add_argument("--width", type=int, default=64, help="Base channel width")
    parser.add_argument("--nepochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Validation/eval batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--time-bins", type=int, default=4, help="Piecewise-constant time intervals per FDE block")
    parser.add_argument("--unstable", action="store_true", help="Use unstable (+) dynamics instead of stable (-) dynamics")

    # FDE parameters
    parser.add_argument("--beta", type=float, default=0.6, help="Fractional order")
    parser.add_argument("--T", type=float, default=1.0, help="End time for FDE integration")
    parser.add_argument("--step_size", type=float, default=0.1, help="FDE integration step size")
    parser.add_argument("--memory", type=int, default=-1, help="Memory for FDE adjoint (-1 for full)")
    parser.add_argument("--return_history", action="store_true", help="Return full state history from FDE solver")

    # Multi-term FDE parameters
    parser.add_argument("--multi_beta", type=float, nargs="+", default=None, help="Fractional orders for multi-term FDE")
    parser.add_argument("--multi_coefficient", type=float, nargs="+", default=None, help="Coefficients for multi-term FDE")
    parser.add_argument("--learn_coefficient", action="store_true", help="Learn coefficients for multi-term FDE")

    # Mode controls
    parser.add_argument("--mode", type=str, default='adjoint', choices=["direct", "adjoint", "adjoint-mixed", "adjoint-mixed-bfloat"], help="Training modes to run")
    parser.add_argument('--adjoint_method', type=str, default='predictor-f', choices=sorted(ADJOINT_METHODS), help='Adjoint method to use for training')
    parser.add_argument('--direct_method', type=str, default='predictor', choices=sorted(DIRECT_METHODS), help='Direct method to use for FDE integration')
    parser.add_argument('--mp_dtype', type=str, default='float32', choices = ['float16', 'bfloat16', 'float32'], help='Datatype to use for multi-precision training (e.g., float16, bfloat16, or float32)')
    parser.add_argument('--mp_loss_scaler', type=str, default='auto', choices=['auto','dynamic','false'], help='Loss scaler to use for multi-precision training (e.g., auto, dynamic, or a fixed float value)')
    
    # Data/system settings
    parser.add_argument("--data-root", type=str, default=".data/stl10", help="STL10 root directory")
    parser.add_argument("--download-data", action="store_true", default=True, help="Download STL10 if missing")
    parser.add_argument("--no-download-data", action="store_false", dest="download_data", help="Do not download STL10")
    parser.add_argument("--train-size", type=int, default=4000, help="Size of train subset from STL10 train split")
    parser.add_argument("--save", type=str, default="./exp_mp_stl10", help="Directory for logs and outputs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()

@dataclass
class FDEConfig:
    beta: float = 0.5
    T: float = 1.0
    step_size: float = 0.1
    method: str = "predictor-f"
    memory: int = -1
    return_history: bool = False
    dtype_hi: Optional[torch.dtype] = None
    mp_dtype: Optional[torch.dtype] = None

    # Multi-term FDE settings
    multi_beta: Optional[List[float]] = None
    multi_coefficient: Optional[List[float]] = None
    learn_coefficient: bool = False

@dataclass
class ModeConfig:
    name: str
    use_adjoint: bool
    method: str
    autocast_dtype: Optional[torch.dtype] = None
    loss_scaler: Any = False
    dtype_hi: Optional[torch.dtype] = None
    mp_dtype: Optional[torch.dtype] = None



# def validate_args(args: argparse.Namespace) -> None:
#     if args.T <= 0:
#         raise ValueError("--T must be positive.")
#     if args.step_size <= 0:
#         raise ValueError("--step_size must be positive.")
#     if args.time_bins < 1:
#         raise ValueError("--time-bins must be at least 1.")
#     if args.train_size < 1:
#         raise ValueError("--train-size must be at least 1.")

#     has_multi_beta = args.multi_beta is not None
#     has_multi_coeff = args.multi_coefficient is not None
#     if has_multi_beta != has_multi_coeff:
#         raise ValueError("Provide both --multi_beta and --multi_coefficient together.")

#     if has_multi_beta and len(args.multi_beta) != len(args.multi_coefficient):
#         raise ValueError("--multi_beta and --multi_coefficient must have the same length.")


# =============================================================================
# Model Components (rampde-like architecture, FDE blocks)
# =============================================================================


class ODEFunc(nn.Module):
    """Time-dependent dynamics with piecewise-constant weights."""

    def __init__(self, ch: int, t_grid: torch.Tensor, is_stable: bool = True):
        super().__init__()

        n_steps = int(len(t_grid) - 1)
        if n_steps < 1:
            raise ValueError("t_grid must define at least one interval.")

        init_weight = torch.randn(ch, ch, 3, 3).mul_(0.1)
        init_bias = torch.zeros(ch)

        self.weight_bank = nn.Parameter(init_weight.unsqueeze(0).repeat(n_steps, 1, 1, 1, 1))
        self.bias_bank = nn.Parameter(init_bias.unsqueeze(0).repeat(n_steps, 1))

        # Single conv whose weight/bias buffers are switched by time interval.
        self.A = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.A._parameters.pop("weight")
        self.A._parameters.pop("bias")
        self.A.register_buffer("weight", self.weight_bank[0])
        self.A.register_buffer("bias", self.bias_bank[0])

        self.A_T = nn.ConvTranspose2d(ch, ch, 3, padding=1, bias=False)
        self.A_T._parameters.pop("weight")
        self.A_T.register_buffer("weight", self.weight_bank[0])

        self.norms = nn.ModuleList([nn.InstanceNorm2d(ch, affine=False) for _ in range(n_steps)])

        self.t_start = float(t_grid[0])
        self.t_end = float(t_grid[-1])
        self.n_intervals = n_steps

        self.act = nn.ReLU(inplace=True)
        self.factor = -1.0 if is_stable else 1.0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.t_end == self.t_start:
            idx = 0
        else:
            t_normalized = (float(t.item()) - self.t_start) / (self.t_end - self.t_start)
            idx = max(0, min(int(t_normalized * self.n_intervals), self.n_intervals - 1))

        self.A._buffers["weight"] = self.weight_bank[idx]
        self.A._buffers["bias"] = self.bias_bank[idx]
        self.A_T._buffers["weight"] = self.weight_bank[idx]

        x = self.A(x)
        x = self.act(x)
        x = self.norms[idx](x)
        x = self.A_T(x)
        return self.factor * x


class FDEBlock(nn.Module):
    """Fractional Differential Equation block."""

    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, fdeint_solver: Any):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.fdeint_solver = fdeint_solver
        # self._setup_multi_term()

    # def _setup_multi_term(self) -> None:
    #     cfg = self.fde_config
    #     if cfg.multi_coefficient is not None:
    #         coeff_tensor = torch.tensor(cfg.multi_coefficient, dtype=torch.float32)
    #         beta_tensor = torch.tensor(cfg.multi_beta, dtype=torch.float32)

    #         if cfg.learn_coefficient:
    #             self.multi_coefficient = nn.Parameter(coeff_tensor)
    #         else:
    #             self.register_buffer("multi_coefficient", coeff_tensor)

    #         self.register_buffer("multi_beta", beta_tensor)
    #     else:
    #         self.multi_coefficient = None
    #         self.multi_beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        options = {
            "memory": cfg.memory,
            "return_history": cfg.return_history,
            "dtype_hi": cfg.dtype_hi,
            "mp_dtype": cfg.mp_dtype,
        }

        # if self.multi_coefficient is not None:
        #     beta = self.multi_beta.to(x.device)
        #     options["multi_coefficient"] = self.multi_coefficient.to(x.device)
        # else:
        beta = torch.tensor(cfg.beta, device=x.device, dtype=x.dtype)

        out = self.fdeint_solver(
            self.odefunc,
            x,
            beta,
            t=cfg.T,
            step_size=cfg.step_size,
            method=cfg.method,
            options=options,
        )
        return out

    def extra_repr(self) -> str:
        cfg = self.fde_config
        base = f"beta={cfg.beta}, T={cfg.T}, step_size={cfg.step_size}, method='{cfg.method}'"
        base += f", memory={cfg.memory}, return_history={cfg.return_history}"
        # if self.multi_coefficient is not None:
        #     base = f"multi_term=True, beta={list(cfg.multi_beta)}, " + base
        return base


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class MPFDE_STL10(nn.Module):
    """STL10 architecture mirroring rampde ODE model, but with FDE blocks."""

    def __init__(self, width: int, fde_config: FDEConfig, fdeint_solver: Any, is_stable: bool = True, time_bins: int = 4):
        super().__init__()
        self.ch = int(width)

        self.t_grid = torch.linspace(0.0, float(fde_config.T), int(time_bins) + 1)

        # 1) Stem
        self.stem = nn.Conv2d(3, self.ch, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(self.ch, affine=True)

        # 2) FDE block #1
        self.fde1 = FDEBlock(ODEFunc(self.ch, self.t_grid, is_stable=is_stable), fde_config, fdeint_solver)

        # 3) Downsample + channel lift
        self.conn1 = nn.Conv2d(self.ch, 2 * self.ch, 1, padding=0, bias=True)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.norm3 = nn.InstanceNorm2d(2 * self.ch, affine=True)

        # 4) FDE block #2
        self.fde2 = FDEBlock(ODEFunc(2 * self.ch, self.t_grid, is_stable=is_stable), fde_config, fdeint_solver)

        self.conn2 = nn.Conv2d(2 * self.ch, 4 * self.ch, 1, padding=0, bias=True)
        self.avg2 = nn.AvgPool2d(2, stride=2)
        self.norm4 = nn.InstanceNorm2d(4 * self.ch, affine=True)

        # 5) FDE block #3
        self.fde3 = FDEBlock(ODEFunc(4 * self.ch, self.t_grid, is_stable=is_stable), fde_config, fdeint_solver)

        self.act = nn.ReLU(inplace=True)

        # 6) Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(4 * self.ch, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f'\nData passed to forward is {x.dtype}')
        x = self.stem(x)
        x = self.norm1(x)
        x = self.act(x)
        #print(f'\nData after stem is {x.dtype}')

        x = self.fde1(x)
        #print(f'\nData after FDE block 1 is {x.dtype}')
        x = self.conn1(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.avg1(x)
        #print(f'\nData after avg1 is {x.dtype}')

        x = self.fde2(x)
        #print(f'\nData after FDE block 2 is {x.dtype}')
        x = self.conn2(x)
        x = self.norm4(x)
        x = self.act(x)
        x = self.avg2(x)
        #print(f'\nData after avg2 is {x.dtype}')

        x = self.fde3(x)
        #print(f'\nData after FDE block 3 is {x.dtype}')
        x = self.head(x)
        #print(f'\nData being returned is {x.dtype}')

        return x


# =============================================================================
# Data
# =============================================================================

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def get_stl10_loaders(data_root: str, download_data: bool, batch_size: int, test_batch_size: int, num_workers: int, train_size: int, seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train_loader, val_loader, train_eval_loader for STL10."""

    mean = (0.4467, 0.4398, 0.4066)
    std = (0.2241, 0.2210, 0.2239)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_aug = STL10(root=data_root, split="train", download=download_data, transform=transform_train)
    full_train_eval = STL10(root=data_root, split="train", download=download_data, transform=transform_eval)

    if train_size >= len(full_train_aug):
        raise ValueError(
            f"--train-size ({train_size}) must be less than STL10 train size ({len(full_train_aug)})."
        )

    split_seed = seed if seed is not None else 42
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(full_train_aug), generator=generator)

    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size:].tolist()
    if not val_idx:
        raise ValueError("Validation split is empty. Reduce --train-size.")

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)
    train_eval_set = Subset(full_train_eval, train_idx)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, train_eval_loader


# =============================================================================
# Utilities
# =============================================================================


def inf_generator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


# def make_autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
#     if device.type != "cuda" or dtype is None:
#         return nullcontext()
#     return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(x)
                predictions = logits.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
        else:
            logits = model(x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def reset_peak_memory(device: torch.device) -> None:
#     if device.type == "cuda":
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats(device)
#         torch.cuda.synchronize(device)

# def get_peak_memory_mb(device: torch.device) -> float:
#     if device.type == 'cuda':
#         return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
#     else:
#         return 0.0

def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    elif name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


# def resolve_direct_method(adjoint_method: str, requested_direct_method: str) -> str:
#     if requested_direct_method != "auto":
#         if requested_direct_method not in DIRECT_METHODS:
#             raise ValueError(f"Invalid direct method '{requested_direct_method}'.")
#         return requested_direct_method

#     base = adjoint_method.split("-")[0]
#     if base not in DIRECT_METHODS:
#         raise ValueError(
#             f"Cannot auto-map adjoint method '{adjoint_method}' to direct. "
#             "Set --direct_method explicitly."
#         )
#     return base


# =============================================================================
# Logging
# =============================================================================


def setup_logger(logpath: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    
    handler = logging.FileHandler(logpath, mode='w')
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.info(f'Logging to {logpath}')
    return logger


# =============================================================================
# Benchmark Modes
# =============================================================================


def build_mode_configs(args: argparse.Namespace, device: torch.device) -> ModeConfig:

    direct_method = args.direct_method
    mp_dtype = dtype_from_name(args.mp_dtype)

    if args.mp_loss_scaler == "auto":
        mp_scaler_mode = "dynamic" if mp_dtype == torch.float16 else "false"
    else:
        mp_scaler_mode = args.mp_loss_scaler

    mode = args.mode
    if mode == "direct":
        return ModeConfig(
            name="direct",
            use_adjoint=False,
            method=direct_method,
            autocast_dtype=None,
            loss_scaler=False,
            mp_dtype=mp_dtype,
        )
    elif mode == "adjoint":
        return ModeConfig(
            name="adjoint",
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=None,
            loss_scaler=False,
            mp_dtype=mp_dtype,
        )
    elif mode == "adjoint-mixed":
        autocast_dtype = mp_dtype if device.type == "cuda" else None
        scaler: Any = False
        if device.type == "cuda" and mp_scaler_mode == "dynamic" and mp_dtype == torch.float16:
            from torchfde import DynamicScaler
            
            scaler = DynamicScaler(dtype_low=torch.float16)

        return ModeConfig(
            name="adjoint-mixed",
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=autocast_dtype,
            loss_scaler=scaler,
            mp_dtype=mp_dtype,
        )
        
    elif mode == "adjoint-mixed-bfloat":
        autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
        return ModeConfig(
            name="adjoint-mixed-bfloat",
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=autocast_dtype,
            loss_scaler=False,
            mp_dtype=mp_dtype,
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'.")


def build_solver(mode_config: ModeConfig):
    if mode_config.use_adjoint:
        from torchfde import fdeint_adjoint

        def solver(func, y0, beta, t, step_size, method, options=None):
            return fdeint_adjoint(
                func,
                y0,
                beta=beta,
                t=t,
                step_size=step_size,
                method=method,
                options=options,
                loss_scaler=mode_config.loss_scaler,
            )
        return solver

    from torchfde import fdeint

    def solver(func, y0, beta, t, step_size, method, options=None):
        return fdeint(
            func,
            y0,
            beta=beta,
            t=t,
            step_size=step_size,
            method=method,
            options=options,
        )

    return solver


def measure_inference(model: nn.Module, val_loader: DataLoader, device: torch.device, autocast_dtype: Optional[torch.dtype]) -> Tuple[float, float, float]:
    model.eval()
    #reset_peak_memory(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    acc = evaluate_accuracy(model, val_loader, device, autocast_dtype=autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device)
    else: 
        peak_mb = 0.0
    #peak_mb = get_peak_memory_mb(device)
    return elapsed, peak_mb, acc


def train(args: argparse.Namespace, mode_cfg: ModeConfig, device: torch.device, train_loader: DataLoader, val_loader: DataLoader, train_eval_loader: DataLoader) -> Dict[str, float]:
    mode_save_dir = os.path.join(args.save, mode_cfg.name)
    os.makedirs(mode_save_dir, exist_ok=True)

    logger = setup_logger(
        logpath=os.path.join(mode_save_dir, "training.log"),
        logger_name=f"mp_fde_stl10.{mode_cfg.name}"
    )

    logger.info(f"Using device: {device}")
    logger.info(f'Mode config: {mode_cfg}')
    if mode_cfg.use_adjoint:
        logger.info("Using adjoint backpropagation")
    else:
        logger.info(f'Using standard backprop (no adjoint)')

    if mode_cfg.autocast_dtype is not None:
        logger.info(f"Using MP autocast with dtype: {mode_cfg.autocast_dtype}")
    else:
        logger.info("Using full precision (no autocast)")

    if mode_cfg.loss_scaler:
        logger.info(f'Using loss scaler: {mode_cfg.loss_scaler}')
    else:
        logger.info('No loss scaler will be used')

    data_gen = inf_generator(train_loader)
    best_acc = 0.0
    last_val_acc = float("nan")

    fde_config = FDEConfig(
        beta=args.beta,
        T=args.T,
        step_size=args.step_size,
        method=mode_cfg.method,
        memory=args.memory,
        return_history=args.return_history,
        dtype_hi=train_loader.dataset[0][0].dtype,
        mp_dtype=mode_cfg.mp_dtype,
    )

    logger.info(
        f"FDE Config: "
        f"  beta={fde_config.beta},"
        f"  T={fde_config.T}, "
        f"  step_size={fde_config.step_size},"
        f"  method={fde_config.method}"
    )

    fdeint_solver = build_solver(mode_cfg)
    model = MPFDE_STL10(width=args.width, fde_config=fde_config, fdeint_solver=fdeint_solver, is_stable=not args.unstable, time_bins=args.time_bins).to(device)

    logger.info(f"\n\nModel architecture:\n{model}")
    logger.info(f"\nTotal parameters: {count_parameters(model):,}\n")

    batches_per_epoch = len(train_loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs * batches_per_epoch, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training samples: {len(train_loader.dataset)}, batches/epoch: {batches_per_epoch}")

    logger.info(f"Starting training for {args.nepochs} epochs...")

    #reset_peak_memory(device)
    train_step_peak_mem_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    epoch_start_time = time.perf_counter()
    train_start = time.perf_counter()

    for iteration in range(args.nepochs * batches_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        x, y = next(data_gen)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if mode_cfg.autocast_dtype is not None:
            #print('Using autocast with dtype:', mode_cfg.autocast_dtype)
            with torch.autocast(device_type="cuda", dtype=mode_cfg.autocast_dtype):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            #print('Not using autocast')
            logits = model(x)
            loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep parameter values in a bounded range like the rampde STL10 script.
        for param in model.parameters():
            param.data.clamp_(-1, 1)
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
            train_step_peak_mem_mb = max(train_step_peak_mem_mb, peak_memory)

        

        if (iteration + 1) % batches_per_epoch == 0:
            epoch = (iteration + 1) // batches_per_epoch
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                #train_step_peak_mem_mb = max(train_step_peak_mem_mb, get_peak_memory_mb(device))
            epoch_time = time.perf_counter() - epoch_start_time
            if mode_cfg.autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=mode_cfg.autocast_dtype):
                    model.eval()
                    train_acc = evaluate_accuracy(model, train_eval_loader, device, autocast_dtype=mode_cfg.autocast_dtype)
                    val_acc = evaluate_accuracy(model, val_loader, device, autocast_dtype=mode_cfg.autocast_dtype)
            else:
                model.eval()
                train_acc = evaluate_accuracy(model, train_eval_loader, device)
                val_acc = evaluate_accuracy(model, val_loader, device)
            model.train()
            best_acc = max(best_acc, val_acc)
            last_val_acc = val_acc

            lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d} | "
                f"Time {epoch_time:.2f}s | "
                f"Peak Mem {train_step_peak_mem_mb:.2f} MB | "
                f"LR {lr:.4e} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"Best {best_acc:.4f}"
            )
            # print(
            #     f"Epoch {epoch:03d} | "
            #     f"Time {epoch_time:.2f}s | "
            #     f"Peak Mem {train_step_peak_mem_mb:.2f} MB | "
            #     f"LR {lr:.4e} | "
            #     f"Train Acc {train_acc:.4f} | "
            #     f"Val Acc {val_acc:.4f} | "
            #     f"Best {best_acc:.4f}"
            # )

            # if device.type == "cuda":
            #     torch.cuda.synchronize(device)
            #     torch.cuda.reset_peak_memory_stats(device)

            epoch_start_time = time.perf_counter()

    if not np.isfinite(last_val_acc):
        model.eval()
        last_val_acc = evaluate_accuracy(model, val_loader, device, autocast_dtype=mode_cfg.autocast_dtype)
        best_acc = max(best_acc, last_val_acc)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_s = time.perf_counter() - train_start
    train_peak_mem_mb = train_step_peak_mem_mb if device.type == "cuda" else 0.0

    if mode_cfg.autocast_dtype is not None: 
        with torch.autocast(device_type="cuda", dtype=mode_cfg.autocast_dtype):
            model.eval()
            inference_time_s, inference_peak_mem_mb, _ = measure_inference(
                model,
                val_loader,
                device,
                mode_cfg.autocast_dtype,
            )
    else:
        model.eval()
        inference_time_s, inference_peak_mem_mb, _ = measure_inference(
            model,
            val_loader,
            device,
            autocast_dtype=None,
        )

    logger.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    logger.info(
        "Final metrics | "
        f"Final Val Error {1.0 - last_val_acc:.4f} | "
        f"Best Val Error {1.0 - best_acc:.4f} | "
        f"Train Mem {train_peak_mem_mb:.2f} MB | "
        f"Train Time {train_time_s:.2f}s | "
        f"Infer Time {inference_time_s:.2f}s"
        f"Infer Peak Mem {inference_peak_mem_mb:.2f} MB | "
    )

    return {
        "mode": mode_cfg.name,
        "final_val_error": float(1.0 - last_val_acc),
        "best_val_error": float(1.0 - best_acc),
        "final_val_acc": float(last_val_acc),
        "train_gpu_memory_mb": float(train_peak_mem_mb),
        "train_time_s": float(train_time_s),
        "inference_gpu_memory_mb": float(inference_peak_mem_mb),
        "inference_time_s": float(inference_time_s),
    }


# =============================================================================
# Summary / Runner
# =============================================================================


# def format_summary_table(results: Sequence[Dict[str, float]]) -> str:
#     headers = [
#         "Mode",
#         "Val Error",
#         "Train GPU Mem (MB)",
#         "Train Time (s)",
#         "Infer GPU Mem (MB)",
#         "Infer Time (s)",
#     ]

#     rows: List[List[str]] = []
#     for row in results:
#         rows.append([
#             str(row["mode"]),
#             f"{row['final_val_error']:.6f}",
#             f"{row['train_gpu_memory_mb']:.2f}",
#             f"{row['train_time_s']:.2f}",
#             f"{row['inference_gpu_memory_mb']:.2f}",
#             f"{row['inference_time_s']:.2f}",
#         ])

#     widths = [len(h) for h in headers]
#     for row in rows:
#         for i, cell in enumerate(row):
#             widths[i] = max(widths[i], len(cell))

#     def format_row(cells: List[str]) -> str:
#         return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

#     line = "-+-".join("-" * w for w in widths)
#     parts = [format_row(headers), line]
#     parts.extend(format_row(row) for row in rows)
#     return "\n".join(parts)


# def run_all_modes(args: argparse.Namespace) -> List[Dict[str, float]]:
#     validate_args(args)
#     set_seed(args.seed)

#     os.makedirs(args.save, exist_ok=True)

#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     if device.type == "cuda":
#         torch.backends.cudnn.benchmark = True

#     mode_configs = build_mode_configs(args, device)

#     train_loader, val_loader, train_eval_loader = get_stl10_loaders(
#         data_root=args.data_root,
#         download_data=args.download_data,
#         batch_size=args.batch_size,
#         test_batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         train_size=args.train_size,
#         seed=args.seed,
#     )

#     results: List[Dict[str, float]] = []
#     for mode_cfg in mode_configs:
#         results.append(
#             train_one_mode(
#                 args=args,
#                 mode_cfg=mode_cfg,
#                 device=device,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 train_eval_loader=train_eval_loader,
#             )
#         )

#     table = format_summary_table(results)
#     print("\nSummary Table")
#     print(table)

#     summary_log = os.path.join(args.save, "summary.log")
#     with open(summary_log, "a", encoding="utf-8") as file_obj:
#         file_obj.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
#         file_obj.write(table)
#         file_obj.write("\n")

#     return results


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mode_cfg = build_mode_configs(args, device)
    train_loader, val_loader, train_eval_loader = get_stl10_loaders(
        data_root=args.data_root,
        download_data=args.download_data,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        train_size=args.train_size,
        seed=args.seed,
    )
    #print('Data loaders obtained')
    train(args, mode_cfg, device, train_loader, val_loader, train_eval_loader)

