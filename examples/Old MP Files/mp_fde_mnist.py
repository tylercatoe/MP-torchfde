#!/usr/bin/env python
"""
MNIST comparison runner for direct backprop, adjoint backprop, and mixed-precision adjoint.

This mirrors the training flow and epoch logging style of fde_mnist.py, and adds
mode-wise benchmarking plus a final summary table.

Default benchmark modes:
    1) direct          -> fdeint (standard backprop)
    2) adjoint         -> fdeint_adjoint (fp32)
    3) adjoint-mixed   -> fdeint_adjoint with autocast mixed precision
    4) adjoint-mixed-bfloat -> fdeint_adjoint with bfloat16 autocast (if GPU supports)

Typical run:
    python mp_fde_mnist.py

The defaults are tuned for your planned SLURM runs:
    - beta = 0.5
    - T = 5
    - step_size = 0.1
    - batch_size = 128
    - conv downsampling (two stride-2 reductions)
"""

import argparse
import logging
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FDEConfig:
    """Configuration for the FDE solver."""
    beta: float = 0.5
    T: float = 20.0
    step_size: float = 0.1
    method: str = "predictor"
    memory: int = -1  # -1 for full history
    return_history: bool = False

    # Multi-term FDE settings
    multi_beta: Optional[List[float]] = None
    multi_coefficient: Optional[List[float]] = None
    learn_coefficient: bool = False


DIRECT_METHODS = {"predictor", "l1", "gl", "trap", "glmulti"}
ADJOINT_METHODS = {
    "predictor-f", "predictor-o",
    "l1-f", "l1-o",
    "gl-f", "gl-o",
    "trap-f", "trap-o",
}


@dataclass
class ModeConfig:
    name: str
    use_adjoint: bool
    method: str
    autocast_dtype: Optional[torch.dtype]
    loss_scaler: Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/benchmark FDE-based neural network on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Network architecture
    parser.add_argument("--network", type=str, default="odenet", choices=["resnet", "odenet"], help="Network architecture")
    parser.add_argument("--downsampling-method", type=str, default="conv", choices=["conv", "res"], help="Downsampling method before FDE block")

    # Training parameters
    parser.add_argument("--nepochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Test/evaluation batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--data_aug", action="store_true", default=True, help="Enable data augmentation")
    parser.add_argument("--no_data_aug", action="store_false", dest="data_aug", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--data-root", type=str, default="data/mnist", help="Root directory for MNIST data")
    parser.add_argument("--download-data", action="store_true", default=True, help="Download MNIST if missing")
    parser.add_argument("--no-download-data", action="store_false", dest="download_data", help="Do not attempt MNIST download")

    # Benchmark mode controls
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["direct", "adjoint", "adjoint-mixed", "adjoint-mixed-bfloat"],
        choices=["direct", "adjoint", "adjoint-mixed", "adjoint-mixed-bfloat"],
        help="Training modes to run sequentially",
    )
    parser.add_argument(
        "--direct-method",
        type=str,
        default="auto",
        choices=["auto", "predictor", "l1", "gl", "trap", "glmulti"],
        help="Method for direct mode. 'auto' maps from adjoint method base name",
    )
    parser.add_argument(
        "--adjoint-method",
        type=str,
        default="predictor-o",
        choices=sorted(ADJOINT_METHODS),
        help="Method for adjoint and adjoint-mixed* modes",
    )

    # FDE solver parameters
    parser.add_argument("--beta", type=float, default=0.5, help="Fractional order (0 < beta <= 1)")
    parser.add_argument("--T", type=float, default=20.0, help="Integration terminal time")
    parser.add_argument("--step_size", type=float, default=0.1, help="Integration step size")
    parser.add_argument("--memory", type=int, default=-1, help="Memory length for history truncation (-1 for full history)")
    parser.add_argument("--return_history", action="store_true", default=False, help="Return full trajectory history from FDE solver")

    # Multi-term FDE parameters
    parser.add_argument("--multi_beta", type=float, nargs="+", default=None, help="Fractional orders for multi-term FDE")
    parser.add_argument("--multi_coefficient", type=float, nargs="+", default=None, help="Coefficients for multi-term FDE")
    parser.add_argument("--learn_coefficient", action="store_true", help="Make multi-term coefficients learnable")

    # Mixed precision controls (for adjoint-mixed mode)
    parser.add_argument("--mp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Autocast dtype for adjoint-mixed mode")
    parser.add_argument("--mp-loss-scaler", type=str, default="auto", choices=["auto", "dynamic", "false"], help="Loss scaler for adjoint-mixed mode")

    # System settings
    parser.add_argument("--save", type=str, default="./exp_mp_mnist", help="Directory to save logs and outputs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


# =============================================================================
# Neural Network Components
# =============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim: int) -> nn.GroupNorm:
    """Group normalization layer."""
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    """Residual block with pre-activation."""

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):
    """Convolution that concatenates time as an extra channel."""

    def __init__(self, dim_in: int, dim_out: int, ksize: int = 3,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], dim=1)
        return self._layer(ttx)


class ODEFunc(nn.Module):
    """Right-hand-side model f(t, z) for D^beta z = f(t, z)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(x))
        out = self.conv1(t, out)
        out = self.relu(self.norm2(out))
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEFuncDeep(nn.Module):
    """Deeper ODEFunc with residual connection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = norm(dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.conv3 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm4 = norm(dim)
        self.conv4 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm5 = norm(dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(x))
        out = self.conv1(t, out)
        out = self.act(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(t, out)

        residual = out

        out = self.act(self.norm3(out))
        out = self.conv3(t, out)
        out = self.act(self.norm4(out))
        out = self.dropout(out)
        out = self.conv4(t, out)

        out = out + residual
        out = self.norm5(out)

        return out


class FDEBlock(nn.Module):
    """Fractional Differential Equation block."""

    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, fdeint_solver):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.fdeint_solver = fdeint_solver
        self._setup_multi_term()

    def _setup_multi_term(self) -> None:
        cfg = self.fde_config

        if cfg.multi_coefficient is not None:
            coeff_tensor = torch.tensor(cfg.multi_coefficient, dtype=torch.float32)
            beta_tensor = torch.tensor(cfg.multi_beta, dtype=torch.float32)

            if cfg.learn_coefficient:
                self.multi_coefficient = nn.Parameter(coeff_tensor)
            else:
                self.register_buffer("multi_coefficient", coeff_tensor)

            self.register_buffer("multi_beta", beta_tensor)
        else:
            self.multi_coefficient = None
            self.multi_beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        options = {
            "memory": cfg.memory,
            "return_history": cfg.return_history,
        }

        if self.multi_coefficient is not None:
            beta = self.multi_beta.to(x.device)
            options["multi_coefficient"] = self.multi_coefficient.to(x.device)
        else:
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
        base_repr = f"beta={cfg.beta}, T={cfg.T}, step_size={cfg.step_size}, method='{cfg.method}'"
        base_repr += f", memory={cfg.memory}, return_history={cfg.return_history}"
        if self.multi_coefficient is not None:
            base_repr = f"multi_term=True, beta={list(cfg.multi_beta)}, " + base_repr
        return base_repr


class Flatten(nn.Module):
    """Flatten spatial dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


# =============================================================================
# Data Loading
# =============================================================================

def get_mnist_loaders(
    data_root: str = "data/mnist",
    download_data: bool = True,
    data_aug: bool = True,
    batch_size: int = 128,
    test_batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create MNIST data loaders."""
    if download_data:
        ensure_mnist_downloaded(data_root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4) if data_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=transform_train)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=transform_test)
    train_eval_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=transform_test)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_eval_loader


def ensure_mnist_downloaded(data_root: str) -> None:
    """
    Download MNIST with a cross-process lock to avoid concurrent corruption.

    Multiple SLURM jobs can start at once. This lock ensures only one process
    performs download/extract while others wait.
    """
    os.makedirs(data_root, exist_ok=True)
    lock_path = os.path.join(data_root, ".mnist_download.lock")

    try:
        import fcntl  # Linux/Unix
    except ImportError:
        fcntl = None

    if fcntl is None:
        datasets.MNIST(root=data_root, train=True, download=True)
        datasets.MNIST(root=data_root, train=False, download=True)
        return

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            datasets.MNIST(root=data_root, train=True, download=True)
            datasets.MNIST(root=data_root, train=False, download=True)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# =============================================================================
# Training Utilities
# =============================================================================

def inf_generator(iterable):
    """Infinite iterator over a dataloader."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_lr_scheduler(
    initial_lr: float,
    batch_size: int,
    batches_per_epoch: int,
    boundary_epochs: Tuple[int, ...],
    decay_rates: Tuple[float, ...],
) -> Callable[[int], float]:
    """Create a step learning rate scheduler."""
    scaled_lr = initial_lr * batch_size / 128

    boundaries = [batches_per_epoch * epoch for epoch in boundary_epochs]
    lrs = [scaled_lr * decay for decay in decay_rates]

    def lr_fn(iteration: int) -> float:
        for boundary, lr in zip(boundaries, lrs):
            if iteration < boundary:
                return lr
        return lrs[-1]

    return lr_fn


def make_autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with make_autocast_context(device, autocast_dtype):
            logits = model(x)
        predictions = logits.argmax(dim=1)
        correct += (predictions == y).sum().item()
        total += y.size(0)

    model.train()
    return correct / total


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype name: {name}")


def resolve_direct_method(adjoint_method: str, requested_direct_method: str) -> str:
    if requested_direct_method != "auto":
        if requested_direct_method not in DIRECT_METHODS:
            raise ValueError(f"Invalid direct method '{requested_direct_method}'.")
        return requested_direct_method

    base = adjoint_method.split("-")[0]
    if base not in DIRECT_METHODS:
        raise ValueError(
            f"Cannot auto-map adjoint method '{adjoint_method}' to a direct method. "
            "Set --direct-method explicitly."
        )
    return base


# =============================================================================
# Model Construction
# =============================================================================

def build_model(
    network_type: str,
    downsampling: str,
    fde_config: FDEConfig,
    fdeint_solver,
    dim: int = 64,
) -> nn.Module:
    """Build the complete model."""
    if downsampling == "conv":
        # Two stride-2 convs perform the two downsampling stages.
        downsampling_layers = [
            nn.Conv2d(1, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
        ]
    else:
        downsampling_layers = [
            nn.Conv2d(1, dim, 3, 1),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
        ]

    if network_type == "odenet":
        feature_layers = [FDEBlock(ODEFuncDeep(dim), fde_config, fdeint_solver)]
    else:
        feature_layers = [ResBlock(dim, dim) for _ in range(6)]

    fc_layers = [
        norm(dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(dim, 10),
    ]

    return nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)


# =============================================================================
# Logging
# =============================================================================

def setup_logger(logpath: str, filepath: str, logger_name: str, debug: bool = False) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(logpath, mode="a")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Source: {filepath}")
    return logger


# =============================================================================
# Benchmark Run
# =============================================================================

def build_mode_configs(args: argparse.Namespace, device: torch.device) -> List[ModeConfig]:
    from torchfde import DynamicScaler

    direct_method = resolve_direct_method(args.adjoint_method, args.direct_method)
    mp_dtype = dtype_from_name(args.mp_dtype)

    if args.mp_loss_scaler == "auto":
        mp_scaler_mode = "dynamic" if mp_dtype == torch.float16 else "false"
    else:
        mp_scaler_mode = args.mp_loss_scaler

    mode_configs: List[ModeConfig] = []
    for mode in args.modes:
        if mode == "direct":
            mode_configs.append(
                ModeConfig(
                    name="direct",
                    use_adjoint=False,
                    method=direct_method,
                    autocast_dtype=None,
                    loss_scaler=False,
                )
            )
        elif mode == "adjoint":
            mode_configs.append(
                ModeConfig(
                    name="adjoint",
                    use_adjoint=True,
                    method=args.adjoint_method,
                    autocast_dtype=None,
                    loss_scaler=False,
                )
            )
        elif mode == "adjoint-mixed":
            autocast_dtype = mp_dtype if device.type == "cuda" else None
            scaler: Any = False
            if device.type == "cuda" and mp_scaler_mode == "dynamic":
                if mp_dtype != torch.float16:
                    scaler = False
                else:
                    scaler = DynamicScaler(dtype_low=torch.float16)

            mode_configs.append(
                ModeConfig(
                    name="adjoint-mixed",
                    use_adjoint=True,
                    method=args.adjoint_method,
                    autocast_dtype=autocast_dtype,
                    loss_scaler=scaler,
                )
            )
        elif mode == "adjoint-mixed-bfloat":
            autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
            mode_configs.append(
                ModeConfig(
                    name="adjoint-mixed-bfloat",
                    use_adjoint=True,
                    method=args.adjoint_method,
                    autocast_dtype=autocast_dtype,
                    loss_scaler=False,
                )
            )

    return mode_configs


def build_solver(mode_cfg: ModeConfig):
    if mode_cfg.use_adjoint:
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
                loss_scaler=mode_cfg.loss_scaler,
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


def measure_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[float, float, float]:
    reset_peak_memory(device)
    start = time.perf_counter()
    acc = evaluate_accuracy(model, test_loader, device, autocast_dtype=autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = get_peak_memory_mb(device)
    return acc, elapsed, peak_mb


def train_one_mode(
    args: argparse.Namespace,
    mode_cfg: ModeConfig,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_eval_loader: DataLoader,
) -> Dict[str, float]:
    mode_save_dir = os.path.join(args.save, mode_cfg.name)
    os.makedirs(mode_save_dir, exist_ok=True)

    logger = setup_logger(
        logpath=os.path.join(mode_save_dir, "training.log"),
        filepath=os.path.abspath(__file__),
        logger_name=f"mp_fde_mnist.{mode_cfg.name}",
        debug=args.debug,
    )

    logger.info(f"Using device: {device}")
    if mode_cfg.use_adjoint:
        logger.info("Using adjoint backpropagation")
    else:
        logger.info("Using direct backpropagation")

    if mode_cfg.name == "direct" and args.adjoint_method.endswith("-o"):
        logger.info(
            "Direct mode uses the base forward scheme '%s' corresponding to adjoint '%s'.",
            mode_cfg.method,
            args.adjoint_method,
        )

    if mode_cfg.autocast_dtype is not None:
        logger.info(f"Using mixed precision autocast dtype: {mode_cfg.autocast_dtype}")
    else:
        logger.info("Using full precision (no autocast)")
    if mode_cfg.loss_scaler is False:
        logger.info("Using loss scaler: disabled")
    else:
        logger.info(f"Using loss scaler: {type(mode_cfg.loss_scaler).__name__}")

    fde_config = FDEConfig(
        beta=args.beta,
        T=args.T,
        step_size=args.step_size,
        method=mode_cfg.method,
        memory=args.memory,
        return_history=args.return_history,
        multi_beta=args.multi_beta,
        multi_coefficient=args.multi_coefficient,
        learn_coefficient=args.learn_coefficient,
    )

    logger.info(
        f"FDE Config: beta={fde_config.beta}, T={fde_config.T}, "
        f"step_size={fde_config.step_size}, method='{fde_config.method}', "
        f"memory={fde_config.memory}, return_history={fde_config.return_history}"
    )
    if fde_config.multi_beta:
        logger.info(
            f"Multi-term: beta={fde_config.multi_beta}, "
            f"coefficients={fde_config.multi_coefficient}, "
            f"learnable={fde_config.learn_coefficient}"
        )

    solver = build_solver(mode_cfg)
    model = build_model(
        network_type=args.network,
        downsampling=args.downsampling_method,
        fde_config=fde_config,
        fdeint_solver=solver,
    ).to(device)

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {count_parameters(model):,}")

    batches_per_epoch = len(train_loader)
    logger.info(
        f"Training samples: {len(train_loader.dataset)}, batches/epoch: {batches_per_epoch}"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_fn = get_lr_scheduler(
        initial_lr=args.lr,
        batch_size=args.batch_size,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=(60, 100, 140),
        decay_rates=(1.0, 0.1, 0.01, 0.001),
    )
    criterion = nn.CrossEntropyLoss()

    data_gen = inf_generator(train_loader)
    best_acc = 0.0
    last_test_acc = float("nan")

    logger.info("Starting training...")
    epoch_start_time = time.time()

    reset_peak_memory(device)
    train_start = time.perf_counter()
    train_step_peak_mem_mb = 0.0

    for iteration in range(args.nepochs * batches_per_epoch):
        # Match rampde-style memory profiling: reset peak at each training step.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        lr = lr_fn(iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        x, y = next(data_gen)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with make_autocast_context(device, mode_cfg.autocast_dtype):
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        # Capture per-step train peak memory after backward/step.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            train_step_peak_mem_mb = max(train_step_peak_mem_mb, get_peak_memory_mb(device))

        if (iteration + 1) % batches_per_epoch == 0:
            epoch = (iteration + 1) // batches_per_epoch
            epoch_time = time.time() - epoch_start_time

            train_acc = evaluate_accuracy(
                model,
                train_eval_loader,
                device,
                autocast_dtype=mode_cfg.autocast_dtype,
            )
            test_acc = evaluate_accuracy(
                model,
                test_loader,
                device,
                autocast_dtype=mode_cfg.autocast_dtype,
            )
            last_test_acc = test_acc

            if test_acc > best_acc:
                best_acc = test_acc

            logger.info(
                f"Epoch {epoch:03d} | "
                f"Time {epoch_time:.1f}s | "
                f"Peak Mem {train_step_peak_mem_mb:.2f} MB | "
                f"LR {lr:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"Best {best_acc:.4f}"
            )

            # Start fresh tracking for next epoch's training steps.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start_time = time.time()

    if not np.isfinite(last_test_acc):
        last_test_acc = evaluate_accuracy(
            model,
            test_loader,
            device,
            autocast_dtype=mode_cfg.autocast_dtype,
        )
        best_acc = max(best_acc, last_test_acc)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_s = time.perf_counter() - train_start
    train_peak_mem_mb = train_step_peak_mem_mb if device.type == "cuda" else 0.0

    inference_acc, inference_time_s, inference_peak_mem_mb = measure_inference(
        model,
        test_loader,
        device,
        mode_cfg.autocast_dtype,
    )

    logger.info(f"Training complete. Best test accuracy: {best_acc:.4f}")
    logger.info(
        "Final metrics | "
        f"Final Test Error {1.0 - last_test_acc:.4f} | "
        f"Train Mem {train_peak_mem_mb:.2f} MB | "
        f"Train Time {train_time_s:.2f}s | "
        f"Infer Mem {inference_peak_mem_mb:.2f} MB | "
        f"Infer Time {inference_time_s:.2f}s"
    )

    return {
        "mode": mode_cfg.name,
        "final_test_error": float(1.0 - last_test_acc),
        "final_test_acc": float(last_test_acc),
        "train_gpu_memory_mb": float(train_peak_mem_mb),
        "train_time_s": float(train_time_s),
        "inference_gpu_memory_mb": float(inference_peak_mem_mb),
        "inference_time_s": float(inference_time_s),
        "inference_test_acc": float(inference_acc),
    }


def format_summary_table(results: Sequence[Dict[str, float]]) -> str:
    headers = [
        "Mode",
        "Test Error",
        "Train GPU Mem (MB)",
        "Train Time (s)",
        "Infer GPU Mem (MB)",
        "Infer Time (s)",
    ]

    rows: List[List[str]] = []
    for r in results:
        rows.append([
            str(r["mode"]),
            f"{r['final_test_error']:.6f}",
            f"{r['train_gpu_memory_mb']:.2f}",
            f"{r['train_time_s']:.2f}",
            f"{r['inference_gpu_memory_mb']:.2f}",
            f"{r['inference_time_s']:.2f}",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    line = "-+-".join("-" * w for w in widths)
    parts = [fmt_row(headers), line]
    parts.extend(fmt_row(row) for row in rows)
    return "\n".join(parts)


def run_all_modes(args: argparse.Namespace) -> List[Dict[str, float]]:
    set_seed(args.seed)

    os.makedirs(args.save, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mode_configs = build_mode_configs(args, device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        data_root=args.data_root,
        download_data=args.download_data,
        data_aug=args.data_aug,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    results: List[Dict[str, float]] = []
    for mode_cfg in mode_configs:
        results.append(
            train_one_mode(
                args=args,
                mode_cfg=mode_cfg,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                train_eval_loader=train_eval_loader,
            )
        )

    table = format_summary_table(results)
    print("\nSummary Table")
    print(table)

    summary_log = os.path.join(args.save, "summary.log")
    with open(summary_log, "a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(table)
        f.write("\n")

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    cli_args = parse_args()
    run_all_modes(cli_args)
