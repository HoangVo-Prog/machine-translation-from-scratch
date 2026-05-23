"""
Low-level tensor math helpers (from-scratch equivalents for common high-level ops).
"""

from __future__ import annotations

from typing import Iterable

import torch


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return exp_x / denom


def log_softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True).clamp_min(1e-12))
    return shifted - logsumexp


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int = -1,
    mask_fill_value: float = -1e9,
) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(mask, mask_fill_value)
    probs = softmax_stable(scores, dim=dim)
    if mask is not None:
        probs = probs.masked_fill(mask, 0.0)
        denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        probs = probs / denom
    return probs


def clip_grad_norm_manual(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm_sq = torch.zeros(1, device=grads[0].device)
    for grad in grads:
        total_norm_sq = total_norm_sq + grad.detach().pow(2).sum()
    total_norm = torch.sqrt(total_norm_sq).item()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)
    return total_norm


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def topk_1d_manual(values: torch.Tensor, k: int):
    if values.dim() != 1:
        raise ValueError(f"topk_1d_manual expects 1D tensor, got shape={tuple(values.shape)}")
    k = max(1, min(int(k), values.numel()))
    data = values.detach().cpu().tolist()
    indexed = list(enumerate(data))
    indexed.sort(key=lambda x: x[1], reverse=True)
    selected = indexed[:k]
    top_values = [float(val) for _, val in selected]
    top_indices = [int(idx) for idx, _ in selected]
    return top_values, top_indices
