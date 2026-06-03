"""
Attention mechanisms for Seq2Seq models.
Supports:
  - Bahdanau (additive) attention
  - Luong (multiplicative / dot / general / concat) attention
"""

import torch

from src.models.layers import ManualModule, ManualLinear, ManualDropout
from src.utils.tensor_ops import masked_softmax


class BahdanauAttention(ManualModule):
    """
    Bahdanau (additive) attention.
    score(h_t, h_s) = v^T * tanh(W1*h_t + W2*h_s)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.W1 = ManualLinear(hidden_size, hidden_size, bias=False)
        self.W2 = ManualLinear(hidden_size, hidden_size, bias=False)
        self.v = ManualLinear(hidden_size, 1, bias=False)
        self.dropout = ManualDropout(dropout)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
        src_mask: torch.Tensor = None,  # (batch, src_len) bool, True = pad
    ):
        query = self.W1(decoder_hidden).unsqueeze(1)
        keys = self.W2(encoder_outputs)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)
        attn_weights = masked_softmax(scores, src_mask, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class LuongAttention(ManualModule):
    """
    Luong attention with three score functions:
      - 'dot'    : score = h_t^T * h_s
      - 'general': score = h_t^T * W * h_s
      - 'concat' : score = v^T * tanh(W * [h_t; h_s])
    """

    def __init__(
        self,
        hidden_size: int,
        score_fn: str = "general",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert score_fn in ("dot", "general", "concat"), \
            f"score_fn must be 'dot', 'general', or 'concat', got '{score_fn}'"
        self.score_fn = score_fn
        self.dropout = ManualDropout(dropout)

        if score_fn == "general":
            self.W = ManualLinear(hidden_size, hidden_size, bias=False)
        elif score_fn == "concat":
            self.W = ManualLinear(hidden_size * 2, hidden_size, bias=False)
            self.v = ManualLinear(hidden_size, 1, bias=False)

    def _score(
        self,
        decoder_hidden: torch.Tensor,   # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
    ) -> torch.Tensor:
        if self.score_fn == "dot":
            return torch.bmm(
                decoder_hidden.unsqueeze(1),
                encoder_outputs.transpose(1, 2),
            ).squeeze(1)

        if self.score_fn == "general":
            energy = self.W(encoder_outputs)
            return torch.bmm(
                decoder_hidden.unsqueeze(1),
                energy.transpose(1, 2),
            ).squeeze(1)

        src_len = encoder_outputs.size(1)
        h_expand = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        cat = torch.cat([h_expand, encoder_outputs], dim=-1)
        return self.v(torch.tanh(self.W(cat))).squeeze(-1)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None,
    ):
        scores = self._score(decoder_hidden, encoder_outputs)
        attn_weights = masked_softmax(scores, src_mask, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


def build_attention(attention_type: str, hidden_size: int, dropout: float = 0.0):
    if attention_type == "none":
        return None
    if attention_type == "bahdanau":
        return BahdanauAttention(hidden_size, dropout=dropout)
    if attention_type == "luong":
        return LuongAttention(hidden_size, score_fn="general", dropout=dropout)
    raise ValueError(
        f"Unknown attention_type='{attention_type}'. "
        "Choose 'none', 'bahdanau' or 'luong'."
    )
