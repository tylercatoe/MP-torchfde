#!/usr/bin/env python3
"""MNIST mixed-precision experiment matrix for fractional adjoint training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchfde import DynamicScaler, fdeint_adjoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
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
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        ksize: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._layer = nn.Conv2d(
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


class ODEFuncDeep(nn.Module):
    def __init__(self, dim: int):
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
        self.dropout = nn.Dropout(0.1)
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


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


@dataclass
class FDEConfig:
    beta: float
    t_final: float
    step_size: float
    method: str
    memory: int


class FDEBlockMixed(nn.Module):
    """FDE block that explicitly forwards loss_scaler into fdeint_adjoint."""

    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, loss_scaler: Any):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.loss_scaler = loss_scaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        beta = torch.tensor(cfg.beta, device=x.device, dtype=x.dtype)
        options = {"memory": cfg.memory, "return_history": False}
        return fdeint_adjoint(
            self.odefunc,
            x,
            beta=beta,
            t=cfg.t_final,
            step_size=cfg.step_size,
            method=cfg.method,
            options=options,
            loss_scaler=self.loss_scaler,
        )


def build_model(
    network_type: str,
    downsampling_method: str,
    fde_config: FDEConfig,
    loss_scaler: Any,
    dim: int = 64,
) -> nn.Module:
    if downsampling_method == "conv":
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
        feature_layers = [FDEBlockMixed(ODEFuncDeep(dim), fde_config, loss_scaler)]
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


def get_mnist_loaders(
    data_root: str,
    batch_size: int,
    test_batch_size: int,
    data_aug: bool,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    if data_aug:
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)

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
    return train_loader, test_loader


@dataclass
class PrecisionConfig:
    name: str
    autocast_dtype: Optional[torch.dtype]
    loss_scaler_mode: str  # "false" or "dynamic"


def make_autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    max_batches: int,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    n_total = 0
    n_correct = 0
    for bidx, (x, y) in enumerate(loader):
        if max_batches > 0 and bidx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with make_autocast_context(device, autocast_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss_sum += loss.detach().float().item() * x.shape[0]
        n_total += x.shape[0]
        n_correct += (logits.argmax(dim=1) == y).sum().item()
    model.train()
    return loss_sum / max(1, n_total), n_correct / max(1, n_total)


def run_one_config(
    cfg: PrecisionConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    if cfg.loss_scaler_mode == "dynamic":
        loss_scaler: Any = DynamicScaler(dtype_low=torch.float16)
    else:
        loss_scaler = False

    fde_cfg = FDEConfig(
        beta=args.beta,
        t_final=args.t_final,
        step_size=args.step_size,
        method=args.method,
        memory=args.memory,
    )
    model = build_model(
        network_type=args.network,
        downsampling_method=args.downsampling_method,
        fde_config=fde_cfg,
        loss_scaler=loss_scaler,
        dim=args.dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, test_loader = get_mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        data_aug=args.data_aug,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    nan_inf_events = 0
    epoch_times: List[float] = []
    total_processed = 0
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    best_test_acc = 0.0
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        n_total = 0
        n_correct = 0

        for bidx, (x, y) in enumerate(train_loader):
            if args.max_train_batches > 0 and bidx >= args.max_train_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with make_autocast_context(device, cfg.autocast_dtype):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                nan_inf_events += 1
                continue

            loss.backward()

            grads_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                nan_inf_events += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            loss_sum += loss.detach().float().item() * x.shape[0]
            n_total += x.shape[0]
            n_correct += (logits.argmax(dim=1) == y).sum().item()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
        total_processed += n_total

        last_train_loss = loss_sum / max(1, n_total)
        last_train_acc = n_correct / max(1, n_total)
        last_test_loss, last_test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            autocast_dtype=cfg.autocast_dtype,
            max_batches=args.max_test_batches,
        )
        best_test_acc = max(best_test_acc, last_test_acc)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(last_train_loss),
                "train_acc": float(last_train_acc),
                "test_loss": float(last_test_loss),
                "test_acc": float(last_test_acc),
                "epoch_time_s": float(epoch_time),
            }
        )

        if args.verbose:
            print(
                f"[{cfg.name}] epoch {epoch + 1:02d}/{args.epochs} | "
                f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} | "
                f"test_loss={last_test_loss:.4f} test_acc={last_test_acc:.4f} | "
                f"time={epoch_time:.2f}s"
            )

    mean_epoch_s = float(np.mean(epoch_times))
    throughput = total_processed / max(1e-12, sum(epoch_times))
    peak_mem_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
    )

    if isinstance(loss_scaler, DynamicScaler):
        scale_steps = len(loss_scaler.scale_history)
    else:
        scale_steps = 0

    row = {
        "config": cfg.name,
        "final_train_loss": float(last_train_loss),
        "final_train_acc": float(last_train_acc),
        "final_test_loss": float(last_test_loss),
        "final_test_acc": float(last_test_acc),
        "best_test_acc": float(best_test_acc),
        "nan_inf_events": int(nan_inf_events),
        "mean_epoch_s": float(mean_epoch_s),
        "train_samples_per_s": float(throughput),
        "peak_mem_mib": float(peak_mem_mib),
        "dynamic_scale_steps": int(scale_steps),
        "status": "ok",
    }
    return row, history


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "config",
        "final_test_acc",
        "acc_vs_fp32_pct",
        "final_test_loss",
        "nan_inf_events",
        "mean_epoch_s",
        "train_samples_per_s",
        "peak_mem_mib",
        "dynamic_scale_steps",
        "status",
    ]

    fp32_acc = None
    for r in rows:
        if r["config"] == "fp32_unscaled" and r["status"] == "ok":
            fp32_acc = r["final_test_acc"]
            break

    for r in rows:
        if fp32_acc is None or fp32_acc == 0 or r["status"] != "ok":
            r["acc_vs_fp32_pct"] = float("nan")
        else:
            r["acc_vs_fp32_pct"] = 100.0 * (r["final_test_acc"] - fp32_acc) / fp32_acc

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6f}"
        return str(v)

    widths = {h: max(len(h), *(len(fmt(r.get(h, ""))) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(fmt(r.get(h, "")).ljust(widths[h]) for h in headers))


def build_precision_configs(device: torch.device) -> List[PrecisionConfig]:
    configs: List[PrecisionConfig] = [
        PrecisionConfig("fp32_unscaled", None, "false"),
    ]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        configs.append(PrecisionConfig("bf16_unscaled", torch.bfloat16, "false"))
    if device.type == "cuda":
        configs.append(PrecisionConfig("fp16_safe", torch.float16, "false"))
        configs.append(PrecisionConfig("fp16_dynamic", torch.float16, "dynamic"))
    return configs


def plot_test_accuracy_histories(
    histories: Dict[str, List[Dict[str, float]]],
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-out. Install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8.5, 5.2))
    plotted = 0
    for config_name, hist in histories.items():
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        test_acc = [h["test_acc"] for h in hist]
        plt.plot(epochs, test_acc, marker="o", linewidth=2.0, label=config_name)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No epoch histories available to plot.")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy Across Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST mixed-precision matrix for torchfde fractional adjoint.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=512)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default="data/mnist")
    p.add_argument("--data-aug", action="store_true", default=True)
    p.add_argument("--no-data-aug", action="store_false", dest="data_aug")
    p.add_argument("--network", type=str, default="odenet", choices=["odenet", "resnet"])
    p.add_argument("--downsampling-method", type=str, default="conv", choices=["conv", "res"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--method", type=str, default="predictor-f")
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--t-final", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--memory", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--csv-out", type=str, default="")
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--history-json-out", type=str, default="")
    p.add_argument("--plot-out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print(f"Device: {device}")
    configs = build_precision_configs(device)
    print("Running configs:", ", ".join(c.name for c in configs))

    results: List[Dict[str, Any]] = []
    histories: Dict[str, List[Dict[str, float]]] = {}
    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        try:
            row, hist = run_one_config(cfg, args, device)
        except Exception as exc:
            row = {
                "config": cfg.name,
                "final_train_loss": float("nan"),
                "final_train_acc": float("nan"),
                "final_test_loss": float("nan"),
                "final_test_acc": float("nan"),
                "best_test_acc": float("nan"),
                "nan_inf_events": -1,
                "mean_epoch_s": float("nan"),
                "train_samples_per_s": float("nan"),
                "peak_mem_mib": float("nan"),
                "dynamic_scale_steps": -1,
                "status": f"error: {type(exc).__name__}",
            }
            hist = []
        results.append(row)
        histories[cfg.name] = hist

    print("\nMNIST Mixed-Precision Matrix Results")
    print_table(results)

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV: {args.csv_out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.json_out}")

    if args.history_json_out:
        with open(args.history_json_out, "w") as f:
            json.dump(histories, f, indent=2)
        print(f"Saved history JSON: {args.history_json_out}")

    if args.plot_out:
        plot_test_accuracy_histories(histories, args.plot_out)
        print(f"Saved accuracy plot: {args.plot_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
