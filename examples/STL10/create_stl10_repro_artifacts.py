#!/usr/bin/env python3
"""Create the shared STL10 split and initial model parameters once."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# This helper only creates CPU artifacts; it should not use a login-node GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "rampfde"))
sys.path.insert(0, str(repo_root / "rampfde" / "paper" / "stl10"))

from stl10_reproducibility import make_split_file, save_initial_parameters
from ode_stl10 import MPNODE_STL10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--init-state", required=True)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--width", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_path = Path(args.split_file)
    init_path = Path(args.init_state)
    if split_path.exists() or init_path.exists():
        raise SystemExit(
            "Refusing to overwrite an existing reproducibility artifact. "
            "Reuse the existing files or choose a new output directory."
        )

    make_split_file(args.split_file, args.dataset_size, args.train_size, args.seed)

    torch.manual_seed(args.seed)
    model_args = SimpleNamespace(gpu=0, odeint="rampde", precision="float32", stable=True)
    model = MPNODE_STL10(
        args.width,
        model_args,
        torch.float32,
        odeint_func=None,
        ScalerClass=None,
        dynamic_scaler_enabled=False,
        grad_scaler_enabled=False,
    )
    save_initial_parameters(model, args.init_state, args.seed)
    print(f"Created split artifact: {args.split_file}")
    print(f"Created initial-state artifact: {args.init_state}")


if __name__ == "__main__":
    main()
