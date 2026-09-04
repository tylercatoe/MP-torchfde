"""Small, explicit reproducibility helpers for the STL10 comparison.

The comparison uses two separate training scripts.  This module gives both
scripts the same saved data split and the same saved model parameters without
touching the solver implementations.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch


def make_split_file(path: str, dataset_size: int, train_size: int, seed: int) -> None:
    """Create and save one deterministic train/validation split."""
    if not 0 < train_size < dataset_size:
        raise ValueError(
            f"train_size must be between 1 and dataset_size-1; got "
            f"train_size={train_size}, dataset_size={dataset_size}"
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)
    payload = {
        "format": 1,
        "seed": int(seed),
        "dataset_size": int(dataset_size),
        "train_size": int(train_size),
        "train_idx": indices[:train_size].cpu(),
        "val_idx": indices[train_size:].cpu(),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _torch_load(path: str) -> Dict[str, object]:
    """Load a helper artifact across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch versions do not have weights_only.
        return torch.load(path, map_location="cpu")


def load_split_indices(path: str, dataset_size: int, train_size: int) -> Tuple[List[int], List[int]]:
    """Load and validate the shared train/validation split."""
    payload = _torch_load(path)
    saved_dataset_size = int(payload.get("dataset_size", -1))
    saved_train_size = int(payload.get("train_size", -1))
    if saved_dataset_size != dataset_size or saved_train_size != train_size:
        raise ValueError(
            f"Split artifact {path} was created for dataset_size={saved_dataset_size}, "
            f"train_size={saved_train_size}, but this run expects "
            f"dataset_size={dataset_size}, train_size={train_size}."
        )

    train_idx = [int(index) for index in payload["train_idx"]]
    val_idx = [int(index) for index in payload["val_idx"]]
    if len(train_idx) != train_size or len(train_idx) + len(val_idx) != dataset_size:
        raise ValueError(f"Split artifact {path} has invalid index counts.")
    if len(set(train_idx).intersection(val_idx)) != 0:
        raise ValueError(f"Split artifact {path} has overlapping train/validation indices.")
    return train_idx, val_idx


def _canonical_parameter_name(name: str) -> str:
    """Map ODE/FDE block names to one shared naming scheme."""
    for block_number in (1, 2, 3):
        name = name.replace(f"ode{block_number}.func.", f"block{block_number}.")
        name = name.replace(f"fde{block_number}.odefunc.", f"block{block_number}.")
    return name


def save_initial_parameters(model: torch.nn.Module, path: str, seed: int) -> None:
    """Save trainable parameters in a name scheme shared by ODE and FDE."""
    parameters = {
        _canonical_parameter_name(name): parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    payload = {
        "format": 1,
        "seed": int(seed),
        "parameters": parameters,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_initial_parameters(model: torch.nn.Module, path: str) -> None:
    """Load shared initial parameters into either the ODE or FDE model."""
    payload = _torch_load(path)
    saved_parameters = payload.get("parameters")
    if not isinstance(saved_parameters, dict):
        raise ValueError(f"Initial-state artifact {path} is missing its parameters.")

    target_parameters = {
        _canonical_parameter_name(name): parameter
        for name, parameter in model.named_parameters()
    }
    missing = sorted(set(target_parameters) - set(saved_parameters))
    if missing:
        raise ValueError(
            f"Initial-state artifact {path} is missing {len(missing)} model parameters; "
            f"first missing parameter: {missing[0]}"
        )

    with torch.no_grad():
        for canonical_name, parameter in target_parameters.items():
            initial_value = saved_parameters[canonical_name]
            if tuple(initial_value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {canonical_name}: artifact has "
                    f"{tuple(initial_value.shape)}, model has {tuple(parameter.shape)}."
                )
            parameter.copy_(initial_value.to(device=parameter.device, dtype=parameter.dtype))
