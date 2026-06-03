"""
RNN Decoder with Attention (Bahdanau or Luong).
"""

import torch
from typing import Optional, Tuple, Union

from src.models.layers import EmbeddingLayer, ManualDropout, ManualLinear, ManualModule, build_rnn_cell
from src.models.attention import build_attention


class RNNDecoder(ManualModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        attention_type: str = "luong",
        cell_type: str = "lstm",
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.attention_type = attention_type.lower()
        self.use_attention = self.attention_type != "none"

        self.embedding = EmbeddingLayer(vocab_size, embed_dim, pad_idx, dropout)
        self.attention = build_attention(self.attention_type, hidden_size, dropout)

        rnn_input_size = embed_dim + hidden_size if self.use_attention else embed_dim
        self.rnn = build_rnn_cell(
            cell_type,
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )

        out_input_size = hidden_size * 2 if self.use_attention else hidden_size
        self.out_proj = ManualLinear(out_input_size, hidden_size, bias=False)
        self.generator = ManualLinear(hidden_size, vocab_size)
        self.dropout = ManualDropout(dropout)

    def forward_step(
        self,
        token: torch.Tensor,            # (batch,)
        hidden,                          # decoder hidden state
        encoder_outputs: torch.Tensor,   # (batch, src_len, hidden)
        context: torch.Tensor,           # (batch, hidden)  prev context
        src_mask: Optional[torch.Tensor] = None,
    ):
        """
        Single decoder step.
        Returns:
            logits:    (batch, vocab_size)
            hidden:    updated hidden state
            context:   updated context vector (batch, hidden)
            attn_w:    attention weights (batch, src_len)
        """
        embedded = self.embedding(token)           # (batch, embed_dim)
        if self.use_attention:
            rnn_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # (batch,1,e+h)
        else:
            rnn_input = embedded.unsqueeze(1)  # (batch,1,e)

        rnn_out, hidden = self.rnn(rnn_input, hidden)
        rnn_out = rnn_out.squeeze(1)               # (batch, hidden)

        # Attention uses top-layer hidden
        if self.cell_type == "lstm":
            top_hidden = hidden[0][-1]             # (batch, hidden)
        else:
            top_hidden = hidden[-1]                # (batch, hidden)

        if self.use_attention:
            context, attn_w = self.attention(top_hidden, encoder_outputs, src_mask)
            out_input = torch.cat([rnn_out, context], dim=-1)
        else:
            context = torch.zeros_like(rnn_out)
            attn_w = None
            out_input = rnn_out

        out = self.dropout(torch.tanh(self.out_proj(out_input)))
        logits = self.generator(out)               # (batch, vocab)
        return logits, hidden, context, attn_w

    def forward(
        self,
        tgt: torch.Tensor,               # (batch, tgt_len)
        encoder_outputs: torch.Tensor,
        hidden,
        src_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:                   # (batch, tgt_len, vocab)
        batch_size, tgt_len = tgt.shape
        device = tgt.device

        # Init context to zeros
        context = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []

        token = tgt[:, 0]  # <BOS>
        for t in range(1, tgt_len):
            logits, hidden, context, _ = self.forward_step(
                token, hidden, encoder_outputs, context, src_mask
            )
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                token = tgt[:, t]
            else:
                token = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)           # (batch, tgt_len-1, vocab)
