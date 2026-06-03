"""
Reusable layer building blocks without torch.nn.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch

def _collect_from_container(container, prefix: str) -> List[Tuple[str, torch.Tensor]]:
    """Recursively collect named parameters from nested list/dict containers."""
    params = []
    if isinstance(container, dict):
        for k, v in container.items():
            key = f"{prefix}.{k}"
            if isinstance(v, torch.Tensor) and v.requires_grad:
                params.append((key, v))
            elif isinstance(v, ManualModule):
                params.extend(v.named_parameters(key))
            elif isinstance(v, (list, dict)):
                params.extend(_collect_from_container(v, key))
    elif isinstance(container, (list, tuple)):
        for idx, item in enumerate(container):
            key = f"{prefix}.{idx}"
            if isinstance(item, torch.Tensor) and item.requires_grad:
                params.append((key, item))
            elif isinstance(item, ManualModule):
                params.extend(item.named_parameters(key))
            elif isinstance(item, (list, dict)):
                params.extend(_collect_from_container(item, key))
    return params

class ManualModule:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _child_modules(self) -> List["ManualModule"]:
        children = []
        for value in self.__dict__.values():
            if isinstance(value, ManualModule):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, ManualModule):
                        children.append(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, ManualModule):
                        children.append(item)
        return children

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
        params: List[Tuple[str, torch.Tensor]] = []
        for name, value in self.__dict__.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, torch.Tensor) and value.requires_grad:
                params.append((key, value))
            elif isinstance(value, ManualModule):
                params.extend(value.named_parameters(key))
            elif isinstance(value, (list, dict)):
                params.extend(_collect_from_container(value, key))
        return params

    def parameters(self) -> List[torch.Tensor]:
        return [param for _, param in self.named_parameters()]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.named_parameters()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        own = dict(self.named_parameters())
        missing = []
        for name, param in own.items():
            if name not in state_dict:
                missing.append(name)
                continue
            param.data.copy_(state_dict[name].to(param.device, dtype=param.dtype))
        if missing:
            raise KeyError(f"Missing keys in state_dict: {missing}")

    def train(self, mode: bool = True):
        self.training = mode
        for child in self._child_modules():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device: torch.device):
        for _, param in self.named_parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
        for child in self._child_modules():
            child.to(device)
        return self


def _xavier_uniform(shape: Tuple[int, ...]) -> torch.Tensor:
    if len(shape) < 2:
        fan_in = shape[0]
        fan_out = shape[0]
    else:
        fan_in = shape[1]
        fan_out = shape[0]
    bound = math.sqrt(6.0 / float(max(1, fan_in + fan_out)))
    tensor = torch.empty(shape).uniform_(-bound, bound)
    return tensor


class ManualLinear(ManualModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = _xavier_uniform((out_features, in_features)).requires_grad_()
        self.bias = torch.zeros(out_features).requires_grad_() if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class ManualDropout(ManualModule):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = max(0.0, min(1.0, p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        if self.p >= 1.0:
            return torch.zeros_like(x)
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype) / keep_prob
        return x * mask


class EmbeddingLayer(ManualModule):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int, dropout: float = 0.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.weight = torch.empty(vocab_size, embed_dim).normal_(mean=0.0, std=embed_dim ** -0.5).requires_grad_()
        if 0 <= pad_idx < vocab_size:
            self.weight.data[pad_idx].zero_()
        self.dropout = ManualDropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        embedded = self.weight.index_select(0, flat).reshape(*x.shape, self.weight.size(1))
        return self.dropout(embedded)


def _init_rnn_gate(in_size: int, hidden_size: int, gate_count: int) -> Dict[str, torch.Tensor]:
    return {
        "w_ih": _xavier_uniform((gate_count * hidden_size, in_size)).requires_grad_(),
        "w_hh": _xavier_uniform((gate_count * hidden_size, hidden_size)).requires_grad_(),
        "b_ih": torch.zeros(gate_count * hidden_size).requires_grad_(),
        "b_hh": torch.zeros(gate_count * hidden_size).requires_grad_(),
    }


class ManualRNNStack(ManualModule):
    def __init__(
        self,
        cell_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = ManualDropout(dropout if num_layers > 1 else 0.0)

        if self.cell_type == "rnn":
            gate_count = 1
        elif self.cell_type == "gru":
            gate_count = 3
        elif self.cell_type == "lstm":
            gate_count = 4
        else:
            raise ValueError(f"Unknown cell_type='{cell_type}'. Choose rnn/gru/lstm.")

        self.layers: List[List[Dict[str, torch.Tensor]]] = []
        current_input = input_size
        for _ in range(num_layers):
            layer_dirs = []
            for _ in range(self.num_directions):
                layer_dirs.append(_init_rnn_gate(current_input, hidden_size, gate_count))
            self.layers.append(layer_dirs)
            current_input = hidden_size * self.num_directions

    def _init_hidden(self, batch: int, device: torch.device):
        h = torch.zeros(self.num_layers * self.num_directions, batch, self.hidden_size, device=device)
        if self.cell_type == "lstm":
            c = torch.zeros(self.num_layers * self.num_directions, batch, self.hidden_size, device=device)
            return h, c
        return h

    def _rnn_step(self, x_t: torch.Tensor, h_prev: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        preact = x_t @ params["w_ih"].t() + params["b_ih"] + h_prev @ params["w_hh"].t() + params["b_hh"]
        return torch.tanh(preact)

    def _gru_step(self, x_t: torch.Tensor, h_prev: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        gates_x = x_t @ params["w_ih"].t() + params["b_ih"]
        gates_h = h_prev @ params["w_hh"].t() + params["b_hh"]
        i_r, i_z, i_n = gates_x.chunk(3, dim=-1)
        h_r, h_z, h_n = gates_h.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return (1.0 - update_gate) * new_gate + update_gate * h_prev

    def _lstm_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = x_t @ params["w_ih"].t() + params["b_ih"] + h_prev @ params["w_hh"].t() + params["b_hh"]
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def _direction_pass(
        self,
        layer_input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        h0: torch.Tensor,
        c0: Optional[torch.Tensor],
        reverse: bool,
    ):
        batch, seq_len, _ = layer_input.shape
        outputs = [None] * seq_len
        h_t = h0
        c_t = c0
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in indices:
            x_t = layer_input[:, t, :]
            if self.cell_type == "rnn":
                h_t = self._rnn_step(x_t, h_t, params)
            elif self.cell_type == "gru":
                h_t = self._gru_step(x_t, h_t, params)
            else:
                h_t, c_t = self._lstm_step(x_t, h_t, c_t, params)
            outputs[t] = h_t
        outputs = torch.stack(outputs, dim=1)
        return outputs, h_t, c_t

    def forward(self, x: torch.Tensor, hidden=None):
        batch, _, _ = x.shape
        device = x.device
        if hidden is None:
            hidden = self._init_hidden(batch, device)

        if self.cell_type == "lstm":
            h_all, c_all = hidden
        else:
            h_all, c_all = hidden, None

        layer_input = x
        final_h = []
        final_c = []

        for layer_idx in range(self.num_layers):
            outputs_per_dir = []
            for dir_idx in range(self.num_directions):
                flat_idx = layer_idx * self.num_directions + dir_idx
                h0 = h_all[flat_idx]
                c0 = c_all[flat_idx] if c_all is not None else None
                reverse = dir_idx == 1
                out_dir, h_last, c_last = self._direction_pass(
                    layer_input, self.layers[layer_idx][dir_idx], h0, c0, reverse
                )
                outputs_per_dir.append(out_dir)
                final_h.append(h_last)
                if c_all is not None:
                    final_c.append(c_last)

            if self.num_directions == 1:
                layer_output = outputs_per_dir[0]
            else:
                layer_output = torch.cat(outputs_per_dir, dim=-1)

            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            layer_input = layer_output

        final_h_tensor = torch.stack(final_h, dim=0)
        if c_all is not None:
            final_c_tensor = torch.stack(final_c, dim=0)
            return layer_input, (final_h_tensor, final_c_tensor)
        return layer_input, final_h_tensor


def build_rnn_cell(
    cell_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool = False,
) -> ManualRNNStack:
    return ManualRNNStack(
        cell_type=cell_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
