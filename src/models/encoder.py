"""
RNN Encoder (supports RNN / GRU / LSTM, optional bidirectional).
"""

import torch
from typing import Tuple, Union

from src.models.layers import EmbeddingLayer, ManualModule, ManualDropout, ManualLinear, build_rnn_cell
from src.utils.tensor_ops import lengths_to_padding_mask


class RNNEncoder(ManualModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()
        self.num_directions = 2 if bidirectional else 1

        self.embedding = EmbeddingLayer(vocab_size, embed_dim, pad_idx, dropout)
        self.rnn = build_rnn_cell(
            cell_type, embed_dim, hidden_size, num_layers, dropout, bidirectional
        )
        self.dropout = ManualDropout(dropout)

        # Project bidirectional hidden to decoder hidden_size
        if bidirectional:
            self.hidden_proj = ManualLinear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple]]:
        """
        Args:
            src: (batch, src_len) token ids
            src_lengths: (batch,) actual lengths

        Returns:
            encoder_outputs: (batch, src_len, hidden_size)  — always hidden_size (merged if bidir)
            hidden:          last hidden state, projected for decoder
        """
        embedded = self.embedding(src)  # (batch, src_len, embed_dim)

        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, src_len, hidden*num_directions)

        if self.bidirectional:
            # Merge forward/backward: (batch, src_len, hidden)
            batch, src_len, _ = outputs.shape
            outputs = outputs.view(batch, src_len, 2, self.hidden_size)
            outputs = outputs.sum(dim=2)          # sum forward+backward
            hidden = self._merge_hidden(hidden)

        src_mask = lengths_to_padding_mask(src_lengths, max_len=outputs.size(1))
        outputs = outputs.masked_fill(src_mask.unsqueeze(-1), 0.0)
        hidden = self._refresh_hidden_with_lengths(hidden, outputs, src_lengths)

        outputs = self.dropout(outputs)
        return outputs, hidden

    def _merge_hidden(self, hidden):
        """Merge bidirectional hidden states for decoder initialisation."""
        if self.cell_type == "lstm":
            h, c = hidden
            h = self._cat_directions(h)
            c = self._cat_directions(c)
            return h, c
        else:
            return self._cat_directions(hidden)

    def _cat_directions(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden)
        We concatenate the two directions then project.
        """
        # h shape: (num_layers * num_dirs, batch, hidden)
        num_layers = h.size(0) // 2
        batch = h.size(1)
        # (num_layers, 2, batch, hidden)
        h = h.view(num_layers, 2, batch, self.hidden_size)
        # (num_layers, batch, hidden*2)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
        # (num_layers, batch, hidden)
        h = self.hidden_proj(h)
        return h

    def _refresh_hidden_with_lengths(self, hidden, outputs, src_lengths):
        batch_size = outputs.size(0)
        valid = (src_lengths > 0).to(outputs.dtype).view(1, batch_size, 1)
        last_idx = (src_lengths - 1).clamp_min(0)
        top_hidden = outputs[torch.arange(batch_size, device=outputs.device), last_idx]

        if isinstance(hidden, tuple):
            h, c = hidden
            c = c * valid
            top_hidden_expanded = top_hidden.unsqueeze(0)
            if h.size(0) > 1:
                h = torch.cat([h[:-1], top_hidden_expanded], dim=0)
            else:
                h = top_hidden_expanded
            h = h * valid
            return h, c

        top_hidden_expanded = top_hidden.unsqueeze(0)
        if hidden.size(0) > 1:
            hidden = torch.cat([hidden[:-1], top_hidden_expanded], dim=0)
        else:
            hidden = top_hidden_expanded
        hidden = hidden * valid
        return hidden
