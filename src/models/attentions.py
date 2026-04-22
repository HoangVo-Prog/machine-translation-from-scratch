import math

import torch
import torch.nn as nn


class ManualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.matmul(self.weight.t())
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_projection = ManualLinear(encoder_hidden_dim, attention_dim, bias=False)
        self.decoder_projection = ManualLinear(decoder_hidden_dim, attention_dim, bias=False)
        self.energy_projection = ManualLinear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_features = self.decoder_projection(decoder_hidden).unsqueeze(1)
        encoder_features = self.encoder_projection(encoder_outputs)
        energy = torch.tanh(encoder_features + decoder_features)
        scores = self.energy_projection(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class LuongAttention(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        score_method: str = "general",
    ):
        super().__init__()
        valid_methods = {"dot", "general", "concat"}
        if score_method not in valid_methods:
            raise ValueError(f"score_method must be one of {valid_methods}")

        self.score_method = score_method
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        if score_method == "general":
            self.linear_in = ManualLinear(decoder_hidden_dim, encoder_hidden_dim, bias=False)
        elif score_method == "concat":
            self.linear_query = ManualLinear(
                encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim, bias=False
            )
            self.energy_projection = ManualLinear(decoder_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.score_method == "dot":
            if self.encoder_hidden_dim != self.decoder_hidden_dim:
                raise ValueError("Luong dot attention requires equal encoder and decoder hidden dimensions")
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(-1)).squeeze(-1)
        elif self.score_method == "general":
            projected_hidden = self.linear_in(decoder_hidden)
            scores = torch.bmm(encoder_outputs, projected_hidden.unsqueeze(-1)).squeeze(-1)
        else:
            repeated_hidden = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            concat_features = torch.cat([encoder_outputs, repeated_hidden], dim=-1)
            energy = torch.tanh(self.linear_query(concat_features))
            scores = self.energy_projection(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights
