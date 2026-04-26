"""Optimizer and scheduler builders for MT training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def _get_config(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _group_parameters_for_weight_decay(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """Build optimizer based on type (Adam, AdamW, SGD)."""
    optimizer_type = str(_get_config(config, "optimizer_type", "adamw")).lower()
    lr = float(_get_config(config, "learning_rate", _get_config(config, "lr", 5e-4)))
    weight_decay = float(_get_config(config, "weight_decay", 0.01))
    betas = _get_config(config, "betas", (0.9, 0.999))
    eps = float(_get_config(config, "adam_epsilon", _get_config(config, "eps", 1e-8)))
    group_weight_decay = bool(_get_config(config, "group_weight_decay", True))

    if group_weight_decay:
        params = _group_parameters_for_weight_decay(model, weight_decay=weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = float(_get_config(config, "momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'adam', 'adamw', or 'sgd'.")

    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def _resolve_warmup_steps(config: Any, num_training_steps: int) -> int:
    warmup_steps = _get_config(config, "warmup_steps", None)
    if warmup_steps is not None:
        return max(0, int(warmup_steps))
    warmup_ratio = float(_get_config(config, "warmup_ratio", 0.0))
    return max(0, int(num_training_steps * warmup_ratio))


def _build_torch_lambda_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler_type = scheduler_type.lower()

    def linear_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)),
        )

    def cosine_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    fn = cosine_lambda if scheduler_type == "cosine" else linear_lambda
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Any, num_training_steps: int
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LambdaLR | None:
    """Build linear or cosine scheduler with warmup."""
    if num_training_steps <= 0:
        return None

    scheduler_type = str(
        _get_config(config, "scheduler_type", _get_config(config, "lr_scheduler_type", "linear"))
    ).lower()
    if scheduler_type in {"none", "constant", "off", "disabled"}:
        return None

    warmup_steps = _resolve_warmup_steps(config, num_training_steps)

    try:
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

        if scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    except ImportError:
        return _build_torch_lambda_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

