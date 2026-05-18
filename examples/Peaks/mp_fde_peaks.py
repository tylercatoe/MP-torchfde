#!/usr/bin/env python
"""
Peaks example for FDEs with the multi-precision adjoint method.
"""

import argparse
import logging
import os
import random
import time
# from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple, Dict#, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, Subset

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FDEConfig: 
    beta: float = 0.5
    T: float = 1.0
    step_size: float = 0.1
    method: str = 'predictor-f'
    memory: int = -1
    return_history: bool = False
    dtype_hi: Optional[torch.dtype] = None

    # Multi-term FDE Settings
    multi_beta: Optional[List[float]] = None
    multi_coefficient: Optional[List[float]] = None
    learn_coefficient: bool = False

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

@dataclass
class ModeConfig:
    name: str
    use_adjoint: bool
    method: str
    autocast_dtype: Optional[torch.dtype] = None
    loss_scaler: Any = False
    dtype_hi: Optional[torch.dtype] = None

def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Peaks example for FDEs with the multi-precision adjoint method.")
    
    # Training Settings
    parser.add_argument('--width', type=int, default=64, help='Width of the hidden layers in the model')
    parser.add_argument('--nepochs', type=int, default=160, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=500, help='Batch size for testing')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Data Settings
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples to generate for the peaks dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')

    # FDE Settings
    parser.add_argument('--beta', type=float, default=0.5, help='Order of the fractional derivative')
    parser.add_argument('--T', type=float, default=1.0, help='End time for FDE integration')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for FDE integration')
    parser.add_argument('--memory', type=int, default=-1, help='Memory setting for FDE integration (-1 for full memory)')
    parser.add_argument('--return_history', action='store_true', help='Whether to return the full history of the solution during FDE integration')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the ODE function network for the FDE block')

    # Multi-term FDE Settings
    parser.add_argument('--multi_beta', type=float, nargs='+', default=None, help='Orders of the fractional derivatives for multi-term FDEs')
    parser.add_argument('--multi_coefficient', type=float, nargs='+', default=None, help='Coefficients for the fractional derivatives in multi-term FDEs')
    parser.add_argument('--learn_coefficient', action='store_true', help='Whether to learn the coefficients for the fractional derivatives in multi-term FDEs')

    # Mode Settings
    #parser.add_argument("--dtype_hi", type=str, default='float32', choices=[None, 'float16', 'bfloat16', 'float32'], help="High precision dtype to use for multi-precision training (e.g., float16, bfloat16)")
    parser.add_argument("--mode", type=str, default="adjoint", choices=["direct", "adjoint", "adjoint-mixed", "adjoint-mixed-bfloat"], help="Training mode to run")
    parser.add_argument('--adjoint_method', type=str, default='predictor-f', choices=sorted(ADJOINT_METHODS), help='Adjoint method to use for training')
    parser.add_argument('--direct_method', type=str, default='predictor', choices=sorted(DIRECT_METHODS), help='Direct method to use for FDE integration')
    parser.add_argument('--mp_dtype', type=str, default='float16', choices = ['float16', 'bfloat16'], help='Datatype to use for multi-precision training (e.g., float16, bfloat16)')
    parser.add_argument('--mp_loss_scaler', type=str, default='auto', choices=['auto','dynamic','false'], help='Loss scaler to use for multi-precision training (e.g., auto, dynamic, or a fixed float value)')

    # Other Settings
    parser.add_argument('--save', type=str, default='./exp_mp_peaks', help="Directory for logs and outputs")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use for training (e.g., 0, 1, etc.)')

    return parser.parse_args()

class ODEFunc(nn.Module):
    def __init__(self, width: int, num_layers: int = 3):
        super().__init__()
        self.width = width
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.width, self.width),
                    nn.Tanh(),
                )
                for _ in range(self.num_layers)
            ]
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(self.num_layers):
            out = self.layers[i](out)
        return out

class FDEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, fdeint_solver: Any):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.fdeint_solver = fdeint_solver
        #self._setup_mulit_term()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        options = {
            'memory': cfg.memory,
            'return_history': cfg.return_history,
            'dtype_hi': cfg.dtype_hi,
        }
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
    
class MPFDE_Peaks(nn.Module):
    def __init__(self, width: int, num_layers: int, fde_config: FDEConfig, fdeint_solver: Any):
        super().__init__()
        self.dim_in = 2
        self.dim_out = 1

        self.fc_in = nn.Linear(self.dim_in, width)
        self.fde_block = FDEBlock(
            ODEFunc(width, num_layers),
            fde_config, 
            fdeint_solver
        )
        self.fc_out = nn.Linear(width, self.dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc_in(x)
        out = torch.tanh(out)
        out = self.fde_block(out)
        out = torch.tanh(out)
        out = self.fc_out(out)
        return out

# =============================================================================
# Data
# =============================================================================

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_peaks_data(num_samples: int) -> torch.Tensor:

    # Generate grid of (x, y) coordinates
    x = np.random.uniform(-3, 3, num_samples)
    y = np.random.uniform(-3, 3, num_samples)
    xy = np.stack([x, y], axis=1)

    # Compute peaks function values
    z = 3 * (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) \
        - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
        - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    
    return torch.tensor(xy, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)

@torch.no_grad()
def evaluate_mse(model: nn.Module, data: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    predictions = model(data).squeeze()
    mse = torch.mean((predictions - targets) ** 2).item()
    return mse

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reset_peak_memory(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

def get_peak_memory_usage(device: torch.device) -> float:
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    else:
        return 0
    
def dtype_from_name(name: str) -> torch.dtype:
    if name == 'float16':
        return torch.float16
    elif name == 'bfloat16':
        return torch.bfloat16
    elif name == 'float32':
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype name: {name}")

# =============================================================================
# Logging
# =============================================================================

def setup_logging(logpath: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(logpath, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
    logger.info(f"Logging to {logpath}")
    return logger

# =============================================================================
# Build Modes
# =============================================================================

def build_mode_configs(args: argparse.Namespace, device: torch.device) -> ModeConfig:
    direct_method = args.direct_method
    
    mp_dtype = dtype_from_name(args.mp_dtype)
    #dtype_hi = dtype_from_name(args.dtype_hi) if args.dtype_hi is not None else None
    if args.mp_loss_scaler == 'auto':
        mp_scaler_mode = 'dynamic' if mp_dtype == torch.float16 else "false"
    else:
        mp_scaler_mode = args.mp_loss_scaler
    
    mode = args.mode
    if mode == 'direct':
        return ModeConfig(
            name='direct',
            use_adjoint=False,
            method=direct_method,
            autocast_dtype=None,
            loss_scaler=False,
            #dtype_hi=dtype_hi,
        )
    elif mode == 'adjoint':
        return ModeConfig(
            name='adjoint',
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=None,
            loss_scaler=False,
            #dtype_hi=dtype_hi,
        )
    elif mode == 'adjoint-mixed':
        autocast_dtype = mp_dtype if device.type == 'cuda' else None
        scaler: Any = False
        if device.type == 'cuda' and mp_scaler_mode == 'dynamic' and mp_dtype == torch.float16:
            from torchfde import DynamicScaler

            scaler = DynamicScaler(dtype_low = torch.float16)

        return ModeConfig(
            name='adjoint-mixed',
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=autocast_dtype,
            loss_scaler=scaler,
            #dtype_hi=dtype_hi,
        )
    elif mode == 'adjoint-mixed-bfloat':
        autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
        return ModeConfig(
            name='adjoint-mixed-bfloat',
            use_adjoint=True,
            method=args.adjoint_method,
            autocast_dtype=autocast_dtype,
            loss_scaler=False,
            #dtype_hi=dtype_hi,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
def build_solver(mode_config: ModeConfig): 
    if mode_config.use_adjoint:
        from torchfde import fdeint_adjoint
        def solver(func, y0, beta, t, step_size, method, options=None):
            return fdeint_adjoint(
                func,
                y0,
                beta,
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

def measure_inference(model: nn.Module, device: torch.device, num_inf: int = 100) -> Tuple[float, float, float]:
    data = generate_peaks_data(num_inf)
    model.eval()
    reset_peak_memory(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    pred = model(data[0].to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    peak_mb = get_peak_memory_usage(device)
    elapsed_time = end - start
    mse = torch.mean((pred.squeeze().to(device) - data[1].to(device)) ** 2).item()
    
    return elapsed_time, peak_mb, mse

# Check the output, this seems unnecessary 
def train(
    args: argparse.Namespace,
    mode_config: ModeConfig,
    device: torch.device
) -> Dict[str, float]: 
    mode_save_dir = os.path.join(args.save, mode_config.name)
    os.makedirs(mode_save_dir, exist_ok=True)

    logger = setup_logging(
        logpath=os.path.join(mode_save_dir, 'training.log'),
        logger_name=f'mp_fde_peaks.{mode_config.name}'
    )

    logger.info(f"Using device: {device}")
    logger.info(f'Mode Config: {mode_config}')
    if mode_config.use_adjoint:
        logger.info(f'Using adjoint for backprop')
    else:
        logger.info(f'Using standard backprop (no adjoint)')
    
    if mode_config.autocast_dtype is not None:
        logger.info(f'Using MP autocast with dtype: {mode_config.autocast_dtype}')
    else:
        logger.info('Using full precision (no autocast)')

    if mode_config.loss_scaler:
        logger.info(f'Using loss scaler: {mode_config.loss_scaler}')
    else:
        logger.info('No loss scaler will be used')

    # Generate data
    data, targets = generate_peaks_data(args.num_samples)
    train_size = int(args.num_samples * (1 - args.test_split))
    train_data, train_targets = data[:train_size], targets[:train_size]
    test_data, test_targets = data[train_size:], targets[train_size:]
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    test_data, test_targets = test_data.to(device), test_targets.to(device)

    fde_config = FDEConfig(
        beta = args.beta,
        T = args.T,
        step_size = args.step_size,
        method = mode_config.method,
        dtype_hi = train_data.dtype,
    )

    logger.info(
        f'FDE Config: '
        f'  beta={fde_config.beta},'
        f'  T={fde_config.T}, '
        f'  step_size={fde_config.step_size}, '
        f'  method={fde_config.method}'
    )
    
    fdeint_solver = build_solver(mode_config)
    model = MPFDE_Peaks(width=args.width, num_layers=args.num_layers, fde_config=fde_config, fdeint_solver=fdeint_solver).to(device)

    logger.info(f'Model architecture:\n{model}')
    logger.info(f'Model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.nepochs,
        eta_min=1e-4,
    )
    criterion = nn.MSELoss()

    best_test_mse = float('inf')
    last_test_mse = float('inf')

    logger.info(f'Starting training for {args.nepochs} epochs...')

    reset_peak_memory(device)
    train_step_peak_mem_mb = 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    epoch_start_time = time.perf_counter()
    train_start = time.perf_counter()

    for iteration in range(args.nepochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        perm = torch.randperm(train_size)
        xy_epoch = train_data[perm]
        z_epoch = train_targets[perm]

        for start in range(0, train_size, args.batch_size):
            end = min(start + args.batch_size, train_size)
            xy_batch = xy_epoch[start:end].to(device, non_blocking=True)
            z_batch = z_epoch[start:end].to(device, non_blocking=True)
            print(f"Before autocast: xy_batch dtype={xy_batch.dtype}, z_batch dtype={z_batch.dtype}")

            optimizer.zero_grad(set_to_none=True)
            
            if mode_config.autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=mode_config.autocast_dtype):
                    print(f"Inside autocast: xy_batch dtype={xy_batch.dtype}, z_batch dtype={z_batch.dtype}")
                    pred = model(xy_batch)
                    loss = criterion(pred.squeeze(), z_batch)
            else:
                pred = model(xy_batch)
                loss = criterion(pred.squeeze(), z_batch)

            loss.backward()
            optimizer.step()
        scheduler.step()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_end = time.perf_counter()
        if device.type == 'cuda':
            train_step_peak_mem_mb = max(train_step_peak_mem_mb, get_peak_memory_usage(device))
        
        epoch_time = train_end - epoch_start_time
        train_mse = evaluate_mse(model, train_data, train_targets)
        test_mse = evaluate_mse(model, test_data, test_targets)
        model.train()
        best_test_mse = min(best_test_mse, test_mse)
        last_test_mse = test_mse

        lr = optimizer.param_groups[0]['lr']

        logger.info(
            f'Epoch {iteration:03d} | '
            f'Time {epoch_time:.2f}s | '
            f'Peak Mem {train_step_peak_mem_mb:.2f} MB | '
            f'LR {lr:.4e} | '
            f'Train MSE {train_mse:.6f} | '
            f'Test MSE {test_mse:.6f} | '
            f'Best Test MSE {best_test_mse:.6f}'
        )

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_accumulated_memory_stats(device)

        epoch_start_time = time.perf_counter()
    
    if not np.isfinite(last_test_mse):
        last_test_mse = evaluate_mse(
            model, 
            test_data.to(device), 
            test_targets.to(device)
        )
        best_test_mse = min(best_test_mse, last_test_mse)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_time_s = time.perf_counter() - train_start
    train_step_peak_mem_mb = train_step_peak_mem_mb if device.type == 'cuda' else 0.0

    inf_time, inf_mem, _ = measure_inference(model, device, 500)

    logger.info(
        f'Final Results | '
        f'Final Test MSE {last_test_mse:.6f} | '
        f'Best Test MSE {best_test_mse:.6f} | '
        f'Train Memory {train_step_peak_mem_mb:.2f} MB | '
        f'Train Time {train_time_s:.2f} s | '
        f'Inference Time {inf_time:.4f}s | '
        f'Inference Peak Mem {inf_mem:.2f} MB | '
     )
    
    return {
        "mode": mode_config.name,
        "final_test_mse": float(last_test_mse),
        "best_test_mse": float(best_test_mse),
        "train_gpu_peak_mem_mb": float(train_step_peak_mem_mb),
        "train_time_s": float(train_time_s),
        "inference_time": float(inf_time),
        "inference_peak_mem_mb": float(inf_mem),
    }
    
        
if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    mode_config = build_mode_configs(args, device)
    train(args, mode_config, device)