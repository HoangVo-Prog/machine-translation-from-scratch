"""Loss functions for machine translation training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def label_smoothed_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute label-smoothed and plain CE losses.

    Returns:
        (total_loss, ce_loss)
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq_len, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, seq_len], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Mismatch between logits and labels sequence dims: {tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
        )

    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)

    ce_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="mean",
    )

    if label_smoothing <= 0.0:
        return ce_loss, ce_loss

    log_probs = F.log_softmax(flat_logits, dim=-1)
    mask = flat_labels.ne(ignore_index)
    safe_labels = flat_labels.masked_fill(~mask, 0)

    nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    nll = nll[mask]
    smooth = smooth[mask]

    if nll.numel() == 0:
        return ce_loss, ce_loss

    total_loss = (1.0 - label_smoothing) * nll.mean() + label_smoothing * smooth.mean()
    return total_loss, ce_loss


def compute_mt_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute MT loss and return stable logging metrics."""
    total_loss, ce_loss = label_smoothed_cross_entropy_loss(
        logits=logits,
        labels=labels,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
    logs = {
        "train/loss_total": float(total_loss.detach().item()),
        "train/loss_ce": float(ce_loss.detach().item()),
        "train/label_smoothing": float(label_smoothing),
    }
    return total_loss, logs


def maybe_compute_loss_from_outputs(
    outputs: Any,
    labels: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use model-provided loss when available, else compute from logits/labels."""
    if hasattr(outputs, "loss") and outputs.loss is not None:
        model_loss = outputs.loss
        logs = {
            "train/loss_total": float(model_loss.detach().item()),
            "train/loss_ce": float(model_loss.detach().item()),
            "train/label_smoothing": float(label_smoothing),
        }
        return model_loss, logs

    logits = None
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        if torch.is_tensor(outputs[0]):
            logits = outputs[0]
    elif torch.is_tensor(outputs):
        logits = outputs

    if logits is None or labels is None:
        raise RuntimeError("Unable to compute loss: missing logits and/or labels.")
    
    if logits.ndim == 3 and labels.ndim == 2:
        # logits: (B, T-1, V), labels: (B, T)
        if labels.size(1) == logits.size(1) + 1:
            labels = labels[:, 1:]

    return compute_mt_loss(logits, labels, label_smoothing=label_smoothing, ignore_index=ignore_index)

