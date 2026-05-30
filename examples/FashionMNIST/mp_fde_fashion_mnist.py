#!/usr/bin/env python
"""
FashionMNIST comparison runner for direct backprop, adjoint, and mixed-precision adjoint.

The defaults are tuned for planned SLURM runs:
    - beta = 0.3
    - T = 1.0
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
from typing import Any, Callable, Sequence, Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
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
    parser = argparse.ArgumentParser(description="Train/benchmark FDE-based FashionMNIST model with multiple backprop modes")

    # Training and Network Settings
    parser.add_argument("--width", type=int, default=64, help="Base channel width")
    parser.add_argument("--nepochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Validation/eval batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers") 

    # FDE parameters
    parser.add_argument("--beta", type=float, default=0.3, help="Fractional order")
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
    parser.add_argument('--dtype_hi', type=str, default='float32', choices = ['float16', 'bfloat16', 'float32'], help='Higher-precision datatype to use for FDE integration (e.g., float16, bfloat16, or float32)')
    parser.add_argument('--mp_dtype', type=str, default='None', choices = ['float16', 'bfloat16', 'float32'], help='Datatype to use for multi-precision training (e.g., float16, bfloat16, or float32)')
    parser.add_argument('--mp_loss_scaler', type=str, default='auto', choices=['auto','dynamic','false'], help='Loss scaler to use for multi-precision training (e.g., auto, dynamic, or a fixed float value)')
    
    # Data/system settings
    #parser.add_argument("--data-root", type=str, default=".data/fashion_mnist", help="FashionMNIST root directory")
    #parser.add_argument("--download-data", action="store_true", default=True, help="Download FashionMNIST if missing")
    #parser.add_argument("--no-download-data", action="store_false", dest="download_data", help="Do not download FashionMNIST")
    #parser.add_argument("--train-size", type=int, default=4000, help="Size of train subset from FashionMNIST    train split")
    parser.add_argument("--data-aug", action="store_true", help="Use data augmentation (random cropping)")
    parser.add_argument("--save", type=str, default="./exp_mp_fashion_mnist", help="Directory for logs and outputs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()

@dataclass
class FDEConfig:
    beta: float = 0.3
    T: float = 1.0
    step_size: float = 0.1
    method: str = "predictor-f"
    memory: int = -1
    return_history: bool = False
    dtype_hi: Optional[torch.dtype] = torch.float32
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
    dtype_hi: Optional[torch.dtype] = torch.float32
    mp_dtype: Optional[torch.dtype] = None
    loss_scaler: Any = False

def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    elif name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

# =============================================================================
# Model Components
# =============================================================================

class ConcatConv2d(nn.Module):
    """Convolution that concatenates time as an extra channel."""

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
    
def norm(ch: int) -> nn.GroupNorm:
    """Helper function for GroupNorm with 32 groups (or fewer if channels < 32)."""
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

class ODEFunc(nn.Module):
    """Right-hand side f of the ODE."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = norm(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(ch, ch, 3, 1, 1)
        self.norm2 = norm(ch)
        self.conv2 = ConcatConv2d(ch, ch, 3, 1, 1)
        self.norm3 = norm(ch)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(x))
        out = self.conv1(t, out)
        out = self.relu(self.norm2(out))
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class FDEBlock(nn.Module):

    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, fdeint_solver):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.fdeint_solver = fdeint_solver

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        options = {
            "memory": cfg.memory,
            "return_history": cfg.return_history,
            "dtype_hi": cfg.dtype_hi,
            "mp_dtype": cfg.mp_dtype,
        }

        beta = torch.tensor(cfg.beta, device=x.device, dtype=cfg.dtype_hi)
        
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

class Flatten(nn.Module):
    """Flatten spatial dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

# =============================================================================
# Data Loading
# =============================================================================

def get_fashion_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        FashionMNIST(root='data/fashion_mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        FashionMNIST(root='data/fashion_mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        FashionMNIST(root='data/fashion_mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

# =============================================================================
# Training Helpers
# =============================================================================

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    """Build a piecewise constant learning rate that decays at specified epochs."""
    initial_learning_rate = args.lr * (batch_size / batch_denom)

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    values = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(iter):
        lt = [iter < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return values[i]
    return learning_rate_fn

def accuracy(model, dataset_loader):
    total_correct = 0
    num_samples = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        num_samples += y.size(0)

        logits = model(x)
        predicted_class = logits.argmax(dim=1)
        total_correct += (predicted_class == y).sum().item()
    return total_correct / num_samples

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[], displaying=False, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    # with open(filepath, "r") as f:
    #     logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def build_mode_configs(args: argparse.Namespace, device: torch.device) -> ModeConfig:

    direct_method = args.direct_method
    mp_dtype = dtype_from_name(args.mp_dtype)
    dtype_hi = dtype_from_name(args.dtype_hi)

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
            dtype_hi=dtype_hi,
            mp_dtype=None,
            loss_scaler=False,
        )
    elif mode == "adjoint":
        return ModeConfig(
            name="adjoint",
            use_adjoint=True,
            method=args.adjoint_method,
            dtype_hi=dtype_hi,
            mp_dtype=None,
            loss_scaler=False,
        )
    elif mode == "adjoint-mixed":
        scaler: Any = False
        if device.type == "cuda" and mp_scaler_mode == "dynamic" and mp_dtype == torch.float16:
            from torchfde import DynamicScaler
    
            scaler = DynamicScaler(dtype_low=torch.float16)

        return ModeConfig(
            name="adjoint-mixed",
            use_adjoint=True,
            method=args.adjoint_method,
            dtype_hi=dtype_hi,
            mp_dtype=mp_dtype,
            loss_scaler=scaler,
        )
        
    elif mode == "adjoint-mixed-bfloat":
        return ModeConfig(
            name="adjoint-mixed-bfloat",
            use_adjoint=True,
            method=args.adjoint_method,
            dtype_hi=dtype_hi,
            mp_dtype=mp_dtype,
            loss_scaler=False, 
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

def measure_inference(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.time()
    acc = accuracy(model, test_loader)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    else: 
        peak_mb = 0.0
    
    return elapsed, peak_mb, acc

# =============================================================================
# Training 
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    seed_everything(args.seed)
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(f"Arguments: {args}")

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    mode_cfg = build_mode_configs(args, device)
    logger.info(f'Mode config: {mode_cfg}')

    fde_config = FDEConfig(
        beta=args.beta,
        T=args.T,
        step_size=args.step_size,
        method=mode_cfg.method,
        memory=args.memory,
        return_history=args.return_history,
        dtype_hi= dtype_from_name(args.dtype_hi),
        mp_dtype= dtype_from_name(args.mp_dtype),
    )

    if mode_cfg.use_adjoint:
        logger.info("Using adjoint backpropagation")
    else:
        logger.info(f'Using standard backprop (no adjoint)')

    if mode_cfg.mp_dtype is not None:
        logger.info(f"Using MP autocast with dtype: {mode_cfg.mp_dtype}")
    else:
        logger.info("Using full precision (no autocast)")

    if mode_cfg.loss_scaler:
        logger.info(f'Using loss scaler: {mode_cfg.loss_scaler}')
    else:
        logger.info('No loss scaler will be used')

    downsampling_layers = [
            nn.Conv2d(1, args.width, 3, 1),
            norm(args.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.width, args.width, 4, 2, 1),
            norm(args.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.width, args.width, 4, 2, 1),
        ]

    feature_layers = [FDEBlock(ODEFunc(args.width), fde_config, build_solver(mode_cfg))]

    fc_layers = [norm(args.width), 
                 nn.ReLU(inplace=True), 
                 nn.AdaptiveAvgPool2d((1, 1)), 
                 Flatten(), 
                 nn.Linear(args.width, 10)
                 ]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_fashion_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=args.batch_size, batches_per_epoch=batches_per_epoch, boundary_epochs=[10, 30, 60, 100, 150],
        decay_rates=[1, 0.1, 0.01, 0.001, .0001, 0.00005, 0.000001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    best_acc = 0.0
    epoch_time_meter = RunningAverageMeter()

    train_step_peak_mem_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logger.info(f"Starting training for {args.nepochs} epochs...")
    with torch.no_grad():
        model.eval()
        if mode_cfg.mp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=mode_cfg.mp_dtype):
                init_train_acc = accuracy(model, train_eval_loader)
                init_val_acc = accuracy(model, test_loader)
        else:
            init_train_acc = accuracy(model, train_eval_loader)
            init_val_acc = accuracy(model, test_loader)
        init_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Initial | "
            f"LR {init_lr:.4e} | "
            f"Train Acc {init_train_acc:.4f} | "
            f"Val Acc {init_val_acc:.4f}"
        )
        print(
            f"Initial | "
            f"LR {init_lr:.4e} | "
            f"Train Acc {init_train_acc:.4f} | "
            f"Val Acc {init_val_acc:.4f}"
        )
        model.train()

    end = time.time()
    train_start = time.time()

    for iter in range(args.nepochs * batches_per_epoch):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_fn(iter)
        
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if mode_cfg.mp_dtype is not None:
            #print('Using autocast with dtype:', mode_cfg.autocast_dtype)
            with torch.autocast(device_type="cuda", dtype=mode_cfg.mp_dtype):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            #print('Not using autocast')
            logits = model(x)
            loss = criterion(logits, y)

        # if mode_cfg.loss_scaler:
        #     scaled_loss = mode_cfg.loss_scaler.scale(loss)
        #     scaled_loss.backward()
        #     mode_cfg.loss_scaler.step(optimizer)
        #     mode_cfg.loss_scaler.update()
        # else:
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
            train_step_peak_mem_mb = max(train_step_peak_mem_mb, peak_memory)
        
        if (iter + 1) % batches_per_epoch == 0:
            
            with torch.no_grad():
                model.eval()
                epoch = (iter + 1) // batches_per_epoch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_time_meter.update(time.time() - end)
                if mode_cfg.mp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=mode_cfg.mp_dtype):
                        train_acc = accuracy(model, train_eval_loader)
                        val_acc = accuracy(model, test_loader)
                else:
                    train_acc = accuracy(model, train_eval_loader)
                    val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"Time {epoch_time_meter.val:.2f}s | "
                    f"Peak Mem {train_step_peak_mem_mb:.2f} MB | "
                    f"LR {lr:.4e} | "
                    f"Train Acc {train_acc:.4f} | "
                    f"Val Acc {val_acc:.4f} | "
                    f"Best {best_acc:.4f}"
                )
                print(
                    f"Epoch {epoch:03d} | "
                    f"Time {epoch_time_meter.val:.2f}s | "
                    f"Peak Mem {train_step_peak_mem_mb:.2f} MB | "
                    f"LR {lr:.4e} | "
                    f"Train Acc {train_acc:.4f} | "
                    f"Val Acc {val_acc:.4f} | "
                    f"Best {best_acc:.4f}"
                )
                model.train()
            end = time.time()
    
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_s = time.time() - train_start
    train_peak_mem_mb = train_step_peak_mem_mb if device.type == "cuda" else 0.0

    if mode_cfg.mp_dtype is not None: 
        with torch.autocast(device_type="cuda", dtype=mode_cfg.mp_dtype): 
            inference_time_s, inference_peak_mem_mb, acc = measure_inference(
                model,
                test_loader,
                device,
            )
    else:
        inference_time_s, inference_peak_mem_mb, acc = measure_inference(
            model,
            test_loader,
            device,
        )

    logger.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    logger.info(
        "Final metrics | "
        f"Final Val Error {1.0 - acc:.4f} | "
        f"Best Val Error {1.0 - best_acc:.4f} | "
        f"Train Mem {train_peak_mem_mb:.2f} MB | "
        f"Train Time {train_time_s:.2f}s | "
        f"Infer Time {inference_time_s:.2f}s |"
        f"Infer Peak Mem {inference_peak_mem_mb:.2f} MB | "
    )

    # return {
    #     "mode": mode_cfg.name,
    #     "final_val_error": float(1.0 - acc),
    #     "best_val_error": float(1.0 - best_acc),
    #     "final_val_acc": float(acc),
    #     "train_gpu_memory_mb": float(train_peak_mem_mb),
    #     "train_time_s": float(train_time_s),
    #     "inference_gpu_memory_mb": float(inference_peak_mem_mb),
    #     "inference_time_s": float(inference_time_s),
    # }
