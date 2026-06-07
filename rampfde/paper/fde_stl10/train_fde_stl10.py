#!/usr/bin/env python
"""
Neural FDE STL-10 experiment: torchfde FP32 vs rampde FP16.

Architecture mirrors the rampde paper STL-10 (Table 4):
  - Images upscaled to 128×128 for tensor core alignment
  - Stem: 3→ch channels
  - 3 FDE blocks at (ch×128², 2ch×64², 4ch×32²), each with N steps
  - Connectors: 1×1 conv + AvgPool2d between blocks
  - Global AvgPool → Linear(4ch, 10)

The key memory comparison:
  - torchfde FP32: stores ALL N×3_block intermediate activations in autograd graph
  - rampde FP16:  custom adjoint recomputes one step at a time; stores only yt+adj_buf

Usage:
    python train_fde_stl10.py --solver torchfde_fp32 --nepochs 30
    python train_fde_stl10.py --solver rampde_fp16   --nepochs 30
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as transforms

_RAMPDE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TORCHFDE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "MP-torchfde"))
for _p in [_RAMPDE_DIR, _TORCHFDE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_torchfde_fdeint = None
_rampde_fdeint   = None

try:
    from torchfde import fdeint as _torchfde_fdeint   # type: ignore[assignment]
    TORCHFDE_OK = True
except ImportError:
    TORCHFDE_OK = False

try:
    from rampde import fdeint as _rampde_fdeint       # type: ignore[assignment]
    RAMPDE_OK = True
except ImportError:
    RAMPDE_OK = False


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def norm(dim: int) -> nn.Module:
    return nn.InstanceNorm2d(dim, affine=True)


class ConcatConv2d(nn.Module):
    """3×3 conv that prepends a broadcast time channel."""
    def __init__(self, dim: int):
        super().__init__()
        self._layer = nn.Conv2d(dim + 1, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(x[:, :1]) * t
        return self._layer(torch.cat([tt, x], dim=1))


class ODEFunc(nn.Module):
    """f(t, y) for D^β y = f(t, y) — 2-layer conv with InstanceNorm."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = norm(dim)
        self.conv1 = ConcatConv2d(dim)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim)
        self.norm3 = norm(dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(x))
        out = self.conv1(t, out)
        out = F.relu(self.norm2(out))
        out = self.conv2(t, out)
        return self.norm3(out)


class FDEBlockTorchfde(nn.Module):
    """FDE block: torchfde.fdeint (FP32, standard autograd)."""
    def __init__(self, func: nn.Module, beta: float, T: float, step_size: float):
        super().__init__()
        self.func = func
        self.beta = beta
        self.T = T
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.tensor(self.beta, device=x.device, dtype=x.dtype)
        return _torchfde_fdeint(self.func, x, beta,
                                t=self.T, step_size=self.step_size, method="l1")


class FDEBlockRampde(nn.Module):
    """FDE block: rampde.fdeint (FP16 autocast + adj_dtype=float16)."""
    def __init__(self, func: nn.Module, beta: float, T: float, step_size: float):
        super().__init__()
        self.func = func
        self.beta = beta
        self.T = T
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = _rampde_fdeint(self.func, x, self.beta, self.T, self.step_size,
                                 loss_scaler=False, adj_dtype=torch.float16)
        return out.float()


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


def _make_fde_block(solver: str, dim: int, beta: float,
                    T: float, step_size: float) -> nn.Module:
    func = ODEFunc(dim)
    if solver == "torchfde_fp32":
        return FDEBlockTorchfde(func, beta, T, step_size)
    return FDEBlockRampde(func, beta, T, step_size)


def build_model(solver: str, ch: int, beta: float,
                T: float, step_size: float) -> nn.Module:
    """
    3-stage FDE network matching rampde paper STL-10 architecture:
      128×128×ch  →  FDE1  →  64×64×2ch  →  FDE2  →  32×32×4ch  →  FDE3  →  FC
    """
    fde1 = _make_fde_block(solver, ch,    beta, T, step_size)
    fde2 = _make_fde_block(solver, ch*2,  beta, T, step_size)
    fde3 = _make_fde_block(solver, ch*4,  beta, T, step_size)

    return nn.Sequential(
        # Stem: 3×128×128 → ch×128×128
        nn.Conv2d(3, ch, 3, padding=1, bias=True),
        norm(ch), nn.ReLU(inplace=True),
        # FDE block 1: ch×128×128
        fde1,
        # Connector 1: ch→2ch, 128→64
        nn.Conv2d(ch, ch*2, 1, bias=True),
        norm(ch*2), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, stride=2),
        # FDE block 2: 2ch×64×64
        fde2,
        # Connector 2: 2ch→4ch, 64→32
        nn.Conv2d(ch*2, ch*4, 1, bias=True),
        norm(ch*4), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, stride=2),
        # FDE block 3: 4ch×32×32
        fde3,
        # Classifier head
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(ch*4, 10),
    )


# ---------------------------------------------------------------------------
# Data — mirrors rampde paper STL-10 preprocessing
# ---------------------------------------------------------------------------

def get_stl10_loaders(batch_size: int, test_batch_size: int) -> tuple:
    # Per-channel mean/std computed on training set
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)

    tr = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    te = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = STL10(root="data/stl10", split="train", download=True, transform=tr)
    test_ds  = STL10(root="data/stl10", split="test",  download=True, transform=te)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=4, drop_last=True, pin_memory=True),
        DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False,
                   num_workers=4, pin_memory=True),
    )


def inf_gen(loader):
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
    return correct / total, total_loss / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.save, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    N = int(args.T / args.step_size) + 1
    print(f"Device: {device}  |  Solver: {args.solver}  |  "
          f"β={args.beta}  T={args.T}  h={args.step_size}  N={N}  ch={args.ch}")

    if args.solver == "torchfde_fp32" and not TORCHFDE_OK:
        raise RuntimeError("torchfde not found")
    if args.solver == "rampde_fp16" and not RAMPDE_OK:
        raise RuntimeError("rampde not found")

    torch.manual_seed(args.seed)
    model = build_model(args.solver, args.ch, args.beta,
                        args.T, args.step_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    train_loader, test_loader = get_stl10_loaders(args.batch_size, args.test_batch_size)
    bpe = len(train_loader)
    print(f"Batches/epoch: {bpe}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.nepochs * bpe)
    criterion = nn.CrossEntropyLoss()
    data_gen = inf_gen(train_loader)

    results = {
        "solver": args.solver, "beta": args.beta, "T": args.T,
        "step_size": args.step_size, "ch": args.ch,
        "batch_size": args.batch_size, "n_params": n_params, "epochs": [],
    }

    best_acc = 0.0
    fwd_times: list = []
    bwd_times: list = []
    torch.cuda.reset_peak_memory_stats(device)

    for iteration in range(args.nepochs * bpe):
        x, y = next(data_gen)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Timed forward
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = model(x)
        loss = criterion(logits, y)
        torch.cuda.synchronize(device)
        fwd_times.append(time.perf_counter() - t0)

        # Timed backward
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        bwd_times.append(time.perf_counter() - t1)

        optimizer.step()
        scheduler.step()

        if (iteration + 1) % bpe == 0:
            epoch = (iteration + 1) // bpe
            test_acc, test_loss = evaluate(model, test_loader, device)
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)
            lr_now = scheduler.get_last_lr()[0]

            avg_fwd = sum(fwd_times) / len(fwd_times)
            avg_bwd = sum(bwd_times) / len(bwd_times)
            fwd_times.clear()
            bwd_times.clear()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), os.path.join(args.save, "best.pth"))

            ep_rec = {
                "epoch": epoch,
                "test_acc": test_acc, "test_loss": test_loss,
                "best_acc": best_acc,
                "avg_fwd_time_s": avg_fwd, "avg_bwd_time_s": avg_bwd,
                "peak_mem_mb": peak_mem_mb, "lr": lr_now,
            }
            results["epochs"].append(ep_rec)
            print(f"  Epoch {epoch:3d}/{args.nepochs}  "
                  f"acc={test_acc:.4f}  best={best_acc:.4f}  "
                  f"fwd={avg_fwd:.3f}s  bwd={avg_bwd:.3f}s  "
                  f"peak={peak_mem_mb:.0f} MB  lr={lr_now:.5f}")
            with open(os.path.join(args.save, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--solver",    type=str,   default="rampde_fp16",
                   choices=["torchfde_fp32", "rampde_fp16"])
    p.add_argument("--beta",      type=float, default=0.7)
    p.add_argument("--T",         type=float, default=1.0,
                   help="Integration end time per FDE block")
    p.add_argument("--step_size", type=float, default=0.25,
                   help="L1 step size (N = T/step_size + 1)")
    p.add_argument("--ch",        type=int,   default=64,
                   help="Base channel width (blocks use ch, 2ch, 4ch)")
    p.add_argument("--nepochs",   type=int,   default=30)
    p.add_argument("--batch_size",type=int,   default=16,
                   help="Match rampde paper: batch=16")
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--lr",        type=float, default=0.05,
                   help="Match rampde paper: lr=0.05 with cosine annealing")
    p.add_argument("--gpu",       type=int,   default=0)
    p.add_argument("--seed",      type=int,   default=25,
                   help="Match rampde paper seed=25")
    p.add_argument("--save",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save is None:
        args.save = os.path.join("results", args.solver)
    train(args)
