"""
compare_fde_precision.py
========================
Compares three training modes for a Neural FDE on MNIST:

  Mode A  float32  (full precision baseline)
  Mode B  bfloat16 (MP, adj history in float32 → 25% memory saving)
  Mode C  bfloat16 (MP, adj history in bfloat16 → 50% memory saving)

For each mode, reports:
  - Time per epoch (ms)
  - Peak GPU memory (MB)
  - Final train accuracy

Usage
-----
  cd rampde/
  python examples/compare_fde_precision.py            # 2 epochs (quick)
  python examples/compare_fde_precision.py --epochs 5 # longer run
  python examples/compare_fde_precision.py --T 5.0 --step_size 0.1  # more FDE steps
"""

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from rampde import fdeint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FDE precision comparison on MNIST")
    p.add_argument("--epochs",    type=int,   default=2,     help="Training epochs per mode")
    p.add_argument("--batch",     type=int,   default=128,   help="Batch size")
    p.add_argument("--width",     type=int,   default=32,    help="Feature map channels in FDE block")
    p.add_argument("--beta",      type=float, default=0.5,   help="Fractional order β")
    p.add_argument("--T",         type=float, default=2.0,   help="Integration end time")
    p.add_argument("--step_size", type=float, default=0.5,   help="Step size h")
    p.add_argument("--n_train",   type=int,   default=4000,  help="Training subset size (None = full)")
    p.add_argument("--data_root", type=str,   default=".data")
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConcatConv2d(nn.Module):
    """Conv2d that prepends a time channel to the input."""
    def __init__(self, ch_in, ch_out, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in + 1, ch_out, kernel, padding=padding)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(x[:, :1]) * t
        return self.conv(torch.cat([tt, x], dim=1))


class FDEFunc(nn.Module):
    """RHS of the fractional ODE: f(t, x) for D^β x = f."""
    def __init__(self, ch: int):
        super().__init__()
        groups = min(8, ch)
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = ConcatConv2d(ch, ch)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = ConcatConv2d(ch, ch)
        self.norm3 = nn.GroupNorm(groups, ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(x))
        out = self.conv1(t, out)
        out = self.act(self.norm2(out))
        out = self.conv2(t, out)
        return self.norm3(out)


class FDENet(nn.Module):
    """
    MNIST classifier using a Neural FDE block:
      Conv → GroupNorm → ReLU
      → FDE block (D^β x = FDEFunc(t, x))
      → AdaptiveAvgPool → Linear → logits
    """
    def __init__(self, ch: int, beta: float, T: float, step_size: float,
                 autocast_dtype: Optional[torch.dtype],
                 adj_dtype: Optional[torch.dtype]):
        super().__init__()
        self.beta       = beta
        self.T          = T
        self.step_size  = step_size
        self.autocast_dtype = autocast_dtype
        self.adj_dtype  = adj_dtype

        self.downsample = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.ReLU(inplace=True),
        )
        self.fde_func = FDEFunc(ch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)          # (B, ch, 28, 28)

        device_str = x.device.type
        ctx = (torch.autocast(device_type=device_str, dtype=self.autocast_dtype)
               if self.autocast_dtype is not None
               else contextlib.nullcontext())

        with ctx:
            x = fdeint(
                self.fde_func, x,
                beta=self.beta,
                t=self.T,
                step_size=self.step_size,
                adj_dtype=self.adj_dtype,
            )

        return self.classifier(x.float())   # classifier always in float32


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def run_mode(label, model, loader, device, epochs, reset_memory=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.type == 'cuda' and reset_memory:
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    acc = 0.0
    for ep in range(1, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        acc = one_epoch(model, loader, optimizer, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        epoch_times.append(elapsed)
        print(f"    epoch {ep}/{epochs}  acc={acc*100:.1f}%  t={elapsed:.0f}ms")

    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6
               if device.type == 'cuda' else float('nan'))
    avg_ms  = sum(epoch_times) / len(epoch_times)
    return avg_ms, peak_mb, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Data
    tfm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
    if args.n_train:
        train_ds = Subset(train_ds, range(args.n_train))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, pin_memory=(device.type == 'cuda'))

    N = int(round(args.T / args.step_size)) + 1
    print(f"FDE: β={args.beta}, T={args.T}, h={args.step_size}, N={N} steps")
    print(f"Network width: {args.width} channels")
    print(f"Training: {len(train_ds)} samples, {args.epochs} epochs\n")

    # Define modes
    @dataclass
    class Mode:
        label: str
        autocast_dtype: Optional[torch.dtype]
        adj_dtype: Optional[torch.dtype]
        desc: str

    modes = [
        Mode("float32  (baseline)",          None,              None,
             "No autocast. adj_history in fp32."),
        Mode("bfloat16 + adj fp32",           torch.bfloat16,   None,
             "Autocast bf16. adj_history in fp32 (25% mem saving)."),
        Mode("bfloat16 + adj bf16 (50% mem)", torch.bfloat16,   torch.bfloat16,
             "Autocast bf16. adj_history in bf16 (50% mem saving)."),
    ]

    results = []
    for mode in modes:
        print(f"── {mode.label}")
        print(f"   {mode.desc}")
        torch.manual_seed(args.seed)
        model = FDENet(
            ch=args.width,
            beta=args.beta,
            T=args.T,
            step_size=args.step_size,
            autocast_dtype=mode.autocast_dtype,
            adj_dtype=mode.adj_dtype,
        ).to(device)
        avg_ms, peak_mb, acc = run_mode(mode.label, model, loader, device, args.epochs)
        results.append((mode.label, avg_ms, peak_mb, acc))
        print()

    # Summary table
    base_ms, base_mb = results[0][1], results[0][2]
    print("=" * 72)
    print(f"{'Mode':<38} {'ms/epoch':>9} {'vs fp32':>8} {'peak MB':>9} {'acc':>7}")
    print("-" * 72)
    for label, ms, mb, acc in results:
        speedup  = f"{base_ms/ms:.2f}x" if ms > 0 else "—"
        mem_str  = f"{mb:.0f}" if not torch.isnan(torch.tensor(mb)) else "N/A"
        mem_save = (f"({(1-mb/base_mb)*100:.0f}% saved)" if base_mb > 0 and not torch.isnan(torch.tensor(mb))
                    else "")
        print(f"  {label:<36} {ms:>9.0f} {speedup:>8} {mem_str:>6} MB {mem_save}")
    print("=" * 72)
    if device.type != 'cuda':
        print("Note: memory stats only available on CUDA.")


if __name__ == "__main__":
    main()
