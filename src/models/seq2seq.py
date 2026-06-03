"""
Seq2Seq model wrapping Encoder + Decoder.
Supports beam-search and greedy decoding.
"""

import torch
from typing import Dict, List, Optional, Tuple

from src.models.layers import ManualModule
from src.models.encoder import RNNEncoder
from src.models.decoder import RNNDecoder
from src.utils.tensor_ops import log_softmax_stable, lengths_to_padding_mask, topk_1d_manual


class Seq2Seq(ManualModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 3,
        cell_type: str = "lstm",
        attention_type: str = "luong",
        bidirectional: bool = True,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type=cell_type,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=src_pad_idx,
        )
        self.decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_type=attention_type,
            cell_type=cell_type,
            dropout=dropout,
            pad_idx=tgt_pad_idx,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,          # (batch, src_len)
        src_lengths: torch.Tensor,  # (batch,)
        tgt: torch.Tensor,          # (batch, tgt_len)
        teacher_forcing_ratio: float = 1.0,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:              # (batch, tgt_len-1, tgt_vocab)
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        if src_mask is None:
            src_mask = lengths_to_padding_mask(src_lengths, max_len=src.size(1))

        # Adjust hidden depth to match decoder num_layers
        hidden = self._adjust_hidden(hidden)

        logits = self.decoder(
            tgt, encoder_outputs, hidden, src_mask, teacher_forcing_ratio
        )
        return logits

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------
    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,          # (1, src_len) or (batch, src_len)
        src_lengths: torch.Tensor,
        num_beams: int = 5,
        max_len: int = 150,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """
        Returns best token-id sequence for each item in the batch.
        """
        batch_size = src.size(0)
        device = src.device

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        hidden = self._adjust_hidden(hidden)
        src_mask = lengths_to_padding_mask(src_lengths, max_len=src.size(1))

        results = []
        for b in range(batch_size):
            enc_out_b = encoder_outputs[b].unsqueeze(0)       # (1, src_len, h)
            mask_b = src_mask[b].unsqueeze(0) if src_mask is not None else None

            # Init per-sample hidden
            h_b = self._slice_hidden(hidden, b)

            best = self._beam_decode_single(
                enc_out_b, h_b, mask_b, num_beams, max_len, length_penalty, device
            )
            results.append(best)
        return results

    def _beam_decode_single(self, enc_out, hidden, mask, num_beams, max_len, lp, device):
        """Beam search for a single sample."""
        context = torch.zeros(1, self.decoder.hidden_size, device=device)

        # Beams: (score, token_ids, hidden, context)
        beams = [(0.0, [self.sos_idx], hidden, context)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break
            candidates = []
            for score, tokens, h, ctx in beams:
                if tokens[-1] == self.eos_idx:
                    completed.append((score, tokens))
                    continue
                tok = torch.tensor([tokens[-1]], device=device)
                logits, new_h, new_ctx, _ = self.decoder.forward_step(
                    tok, h, enc_out, ctx, mask
                )
                log_probs = log_softmax_stable(logits[0], dim=-1)
                topk_probs, topk_ids = topk_1d_manual(log_probs, num_beams)
                for p, idx in zip(topk_probs, topk_ids):
                    candidates.append((score + p, tokens + [idx], new_h, new_ctx))

            # Keep top-k
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** lp), reverse=True)
            beams = candidates[:num_beams]

        # Also add unfinished beams
        completed += [(s, t) for s, t, *_ in beams]
        if not completed:
            return [self.eos_idx]

        completed.sort(key=lambda x: x[0] / (len(x[1]) ** lp), reverse=True)
        best_tokens = completed[0][1]
        # Strip <BOS> and everything after <EOS>
        if best_tokens[0] == self.sos_idx:
            best_tokens = best_tokens[1:]
        if self.eos_idx in best_tokens:
            best_tokens = best_tokens[: best_tokens.index(self.eos_idx)]
        return best_tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _adjust_hidden(self, hidden):
        """Trim or repeat encoder hidden to match decoder num_layers."""
        target = self.decoder.num_layers
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = self._trim_or_expand(h, target)
            c = self._trim_or_expand(c, target)
            return h, c
        return self._trim_or_expand(hidden, target)

    @staticmethod
    def _trim_or_expand(h: torch.Tensor, target: int) -> torch.Tensor:
        actual = h.size(0)
        if actual == target:
            return h
        if actual > target:
            return h[:target]
        # Repeat last layer
        repeats = target - actual
        return torch.cat([h, h[-1:].expand(repeats, -1, -1)], dim=0)

    def _slice_hidden(self, hidden, b: int):
        """Slice batch dimension b from hidden state."""
        if isinstance(hidden, tuple):
            h, c = hidden
            return h[:, b:b+1, :].contiguous(), c[:, b:b+1, :].contiguous()
        return hidden[:, b:b+1, :].contiguous()
