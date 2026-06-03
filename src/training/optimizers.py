"""
From-scratch optimizers and LR scheduler.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


class BaseOptimizer:
    def __init__(self, params: Iterable[torch.Tensor], lr: float):
        self.params: List[torch.Tensor] = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self, set_to_none: bool = True):
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def step(self):
        raise NotImplementedError

    def set_lr(self, lr: float):
        self.lr = lr

    def get_lrs(self) -> List[float]:
        return [self.lr]

    def state_dict(self) -> Dict:
        return {"lr": self.lr}

    def load_state_dict(self, state_dict: Dict):
        self.lr = state_dict["lr"]


class SGDOptimizer(BaseOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params=params, lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
            update = grad
            if self.weight_decay > 0:
                update = update + self.weight_decay * param.data
            if self.momentum > 0:
                self.velocity[idx].mul_(self.momentum).add_(update)
                update = self.velocity[idx]
            param.data.add_(update, alpha=-self.lr)

    def state_dict(self) -> Dict:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "velocity": self.velocity,
        }

    def load_state_dict(self, state_dict: Dict):
        self.lr = state_dict["lr"]
        self.momentum = state_dict["momentum"]
        self.weight_decay = state_dict["weight_decay"]
        velocity = state_dict.get("velocity")
        if velocity is not None and len(velocity) == len(self.params):
            self.velocity = velocity


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ):
        super().__init__(params=params, lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.step_num = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.step_num += 1
        beta1_pow = self.beta1 ** self.step_num
        beta2_pow = self.beta2 ** self.step_num

        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
            if self.weight_decay > 0 and self.decoupled_weight_decay:
                param.data.mul_(1.0 - self.lr * self.weight_decay)

            update = grad
            if self.weight_decay > 0 and not self.decoupled_weight_decay:
                update = update + self.weight_decay * param.data

            self.m[idx].mul_(self.beta1).add_(update, alpha=1.0 - self.beta1)
            self.v[idx].mul_(self.beta2).addcmul_(update, update, value=1.0 - self.beta2)

            m_hat = self.m[idx] / (1.0 - beta1_pow)
            v_hat = self.v[idx] / (1.0 - beta2_pow)
            denom = torch.sqrt(v_hat) + self.eps
            param.data.addcdiv_(m_hat, denom, value=-self.lr)

    def state_dict(self) -> Dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "decoupled_weight_decay": self.decoupled_weight_decay,
            "step_num": self.step_num,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state_dict: Dict):
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.decoupled_weight_decay = state_dict["decoupled_weight_decay"]
        self.step_num = state_dict["step_num"]

        m = state_dict.get("m")
        v = state_dict.get("v")
        if m is not None and len(m) == len(self.params):
            self.m = m
        if v is not None and len(v) == len(self.params):
            self.v = v


class LinearWarmupDecayScheduler:
    def __init__(self, optimizer: BaseOptimizer, total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.step_num = 0
        self.base_lr = optimizer.lr
        self.last_lr = optimizer.lr

    def _scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 1.0 - progress)

    def step(self):
        self.step_num += 1
        scale = self._scale(self.step_num)
        lr = self.base_lr * scale
        self.optimizer.set_lr(lr)
        self.last_lr = lr

    def get_last_lr(self):
        return [self.last_lr]

    def state_dict(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "step_num": self.step_num,
            "base_lr": self.base_lr,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state_dict: Dict):
        self.total_steps = state_dict["total_steps"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.step_num = state_dict["step_num"]
        self.base_lr = state_dict["base_lr"]
        self.last_lr = state_dict["last_lr"]
        self.optimizer.set_lr(self.last_lr)


def build_optimizer(optimizer_type: str, params, learning_rate: float) -> BaseOptimizer:
    opt = optimizer_type.lower()
    if opt == "adam":
        return AdamOptimizer(params, lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    if opt == "adamw":
        return AdamOptimizer(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            decoupled_weight_decay=True,
        )
    return SGDOptimizer(params, lr=learning_rate, momentum=0.9)
