"""
Label-smoothed cross-entropy loss.
"""

import torch

from src.utils.tensor_ops import log_softmax_stable


class LabelSmoothedCrossEntropy:
    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0):
        self.smoothing = label_smoothing
        self.ignore_index = ignore_index

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.forward(logits, targets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, vocab_size)
        targets: (N,)
        """
        vocab_size = logits.size(-1)
        log_probs = log_softmax_stable(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Mask padding
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.new_tensor(0.0)
        return loss[mask].mean()
