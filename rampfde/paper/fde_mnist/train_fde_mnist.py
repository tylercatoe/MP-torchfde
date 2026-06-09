#!/usr/bin/env python
"""
Neural FDE MNIST experiment: torchfde FP32 vs rampde FP16.

Mirrors the rampde paper's ODE benchmark methodology but for the FDE case.
Trains a CNN + FDE-block classifier on MNIST and reports:
  - Test accuracy per epoch
  - Peak GPU memory per epoch
  - Training time per epoch

Usage:
    # FP32 baseline (torchfde)
    python train_fde_mnist.py --solver torchfde_fp32 --nepochs 30

    # FP16 mixed-precision (rampde)
    python train_fde_mnist.py --solver rampde_fp16 --nepochs 30

    # Quick 3-epoch smoke test
    python train_fde_mnist.py --solver rampde_fp16 --nepochs 3 --batch_size 64

Results are saved to <save_dir>/results.json for later comparison.
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Solver imports
# ---------------------------------------------------------------------------

_RAMPDE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TORCHFDE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "MP-torchfde")
)
for _p in [_RAMPDE_DIR, _TORCHFDE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from torchfde import fdeint as _torchfde_fdeint
    TORCHFDE_OK = True
except ImportError:
    TORCHFDE_OK = False

try:
    from rampde import fdeint as _rampde_fdeint
    RAMPDE_OK = True
except ImportError:
    RAMPDE_OK = False


# ---------------------------------------------------------------------------
# Model components (same architecture as MP-torchfde/examples/fde_mnist.py)
# ---------------------------------------------------------------------------

def norm(dim: int) -> nn.Module:
    return nn.GroupNorm(min(32, dim), dim)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConcatConv2d(nn.Module):
    """Conv2d that concatenates a broadcast time channel before convolution."""

    def __init__(self, dim_in: int, dim_out: int, ksize: int = 3,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self._layer = nn.Conv2d(dim_in + 1, dim_out, kernel_size=ksize,
                                stride=stride, padding=padding, bias=True)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(x[:, :1, :, :]) * t
        return self._layer(torch.cat([tt, x], dim=1))


class ODEFunc(nn.Module):
    """f(t, z) for D^β z = f(t, z)."""

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
        return self.norm3(out)


class FDEBlockTorchfde(nn.Module):
    """FDE block using torchfde.fdeint (FP32, standard autograd)."""

    def __init__(self, func: nn.Module, beta: float, T: float, step_size: float):
        super().__init__()
        self.func = func
        self.beta = beta
        self.T = T
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.tensor(self.beta, device=x.device, dtype=x.dtype)
        return _torchfde_fdeint(self.func, x, beta, t=self.T,
                                step_size=self.step_size, method="l1")


class FDEBlockRampde(nn.Module):
    """FDE block using rampde.fdeint (FP16 autocast + adj_dtype=float16)."""

    def __init__(self, func: nn.Module, beta: float, T: float, step_size: float):
        super().__init__()
        self.func = func
        self.beta = beta
        self.T = T
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = _rampde_fdeint(
                self.func, x, self.beta, self.T, self.step_size,
                loss_scaler=False, adj_dtype=torch.float16,
            )
        return out.float()


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


def build_model(solver: str, dim: int, beta: float, T: float,
                step_size: float) -> nn.Module:
    func = ODEFunc(dim)
    if solver == "torchfde_fp32":
        fde_block = FDEBlockTorchfde(func, beta, T, step_size)
    elif solver == "rampde_fp16":
        fde_block = FDEBlockRampde(func, beta, T, step_size)
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    return nn.Sequential(
        # Downsampling: 28×28 → 6×6 feature maps
        nn.Conv2d(1, dim, 3, 1),
        norm(dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, dim, 4, 2, 1),
        norm(dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, dim, 4, 2, 1),
        # FDE block
        fde_block,
        # Classifier head
        norm(dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(dim, 10),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int, test_batch_size: int, data_aug: bool
                      ) -> Tuple[DataLoader, DataLoader]:
    tr_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4) if data_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])
    te_transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST("data/mnist", train=True, download=True,
                               transform=tr_transform)
    test_ds = datasets.MNIST("data/mnist", train=False, download=True,
                              transform=te_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False,
                             num_workers=2, drop_last=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def inf_gen(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple:
    """Returns (accuracy, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total#, total_loss / total


def lr_schedule(step: int, batches_per_epoch: int, initial_lr: float,
                batch_size: int) -> float:
    lr = initial_lr * batch_size / 128
    epochs = [60, 100, 140]
    rates = [1.0, 0.1, 0.01, 0.001]
    for ep, rate in zip(epochs, rates):
        if step < ep * batches_per_epoch:
            return lr * rate
    return lr * rates[-1]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.save, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Solver: {args.solver}  |  β={args.beta}  "
          f"T={args.T}  h={args.step_size}  N={int(args.T / args.step_size) + 1}")

    if args.solver == "torchfde_fp32" and not TORCHFDE_OK:
        raise RuntimeError("torchfde not found; check _TORCHFDE_DIR path")
    if args.solver == "rampde_fp16" and not RAMPDE_OK:
        raise RuntimeError("rampde not found; check _RAMPDE_DIR path")

    torch.manual_seed(args.seed)
    model = build_model(args.solver, args.dim, args.beta, args.T,
                        args.step_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    train_loader, test_loader = get_mnist_loaders(
        args.batch_size, args.test_batch_size, data_aug=True)
    bpe = len(train_loader)
    print(f"Batches/epoch: {bpe}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    data_gen = inf_gen(train_loader)

    results = {
        "solver": args.solver,
        "beta": args.beta, "T": args.T, "step_size": args.step_size,
        "dim": args.dim, "batch_size": args.batch_size, "n_params": n_params,
        "epochs": [],
    }

    best_acc = 0.0
    for iteration in range(args.nepochs * bpe):
        # LR update
        lr = lr_schedule(iteration, bpe, args.lr, args.batch_size)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = next(data_gen)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # End-of-epoch reporting
        if (iteration + 1) % bpe == 0:
            epoch = (iteration + 1) // bpe

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            acc = evaluate(model, test_loader, device)#test_acc(model, test_loader, device)
            torch.cuda.synchronize(device)
            eval_time = time.perf_counter() - t0

            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           os.path.join(args.save, "best.pth"))

            ep_rec = {
                "epoch": epoch, "test_acc": acc, "best_acc": best_acc,
                "peak_mem_mb": peak_mem, "eval_time_s": eval_time, "lr": lr,
            }
            results["epochs"].append(ep_rec)

            print(f"  Epoch {epoch:3d}/{args.nepochs}  "
                  f"acc={acc:.4f}  best={best_acc:.4f}  "
                  f"peak={peak_mem:.1f} MB  lr={lr:.5f}")

            # Save results incrementally
            with open(os.path.join(args.save, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")
    print(f"Results saved to {args.save}/results.json")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Neural FDE MNIST: torchfde FP32 vs rampde FP16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--solver", type=str, default="rampde_fp16",
                   choices=["torchfde_fp32", "rampde_fp16"],
                   help="Solver to use")
    p.add_argument("--beta", type=float, default=0.7,
                   help="Fractional order β ∈ (0,1)")
    p.add_argument("--T", type=float, default=2.0,
                   help="Integration end time")
    p.add_argument("--step_size", type=float, default=0.1,
                   help="L1 step size h")
    p.add_argument("--dim", type=int, default=64,
                   help="Feature channel width")
    p.add_argument("--nepochs", type=int, default=30,
                   help="Training epochs")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Training batch size")
    p.add_argument("--test_batch_size", type=int, default=256,
                   help="Test batch size")
    p.add_argument("--lr", type=float, default=0.1,
                   help="Base learning rate (scaled by batch_size/128)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--save", type=str, default=None,
                   help="Results directory (default: results/<solver>)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save is None:
        args.save = os.path.join("results", args.solver)
    train(args)
