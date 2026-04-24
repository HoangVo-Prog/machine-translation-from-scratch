"""
src/models/seq2seq.py

Wrapper nối Encoder và Decoder theo đúng interface hiện có trong repo.
"""

import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Kết nối Encoder -> Decoder và tạo source mask từ pad token.

    Module này không tự cài lại RNN/Embedding/Decoder logic.
    Nó chỉ điều phối hai module đã có sẵn trong `src/models`.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor | None:
        if self.src_pad_idx is None:
            return None
        return src.eq(self.src_pad_idx)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
        max_length: int | None = None,
    ):
        """
        Parameters
        ----------
        src : LongTensor [batch, src_len]
        tgt : LongTensor [batch, tgt_len]
        src_lengths : optional, giữ tương thích với Encoder hiện có
        teacher_forcing_ratio : float
        max_length : optional, nếu None thì Decoder tự dùng mặc định của nó
        """
        src_mask = self.create_src_mask(src)
        encoder_outputs, final_hidden = self.encoder(src, src_lengths=src_lengths)

        decoder_kwargs = {
            "tgt": tgt,
            "final_hidden": final_hidden,
            "encoder_outputs": encoder_outputs,
            "src_mask": src_mask,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }
        if max_length is not None:
            decoder_kwargs["max_length"] = max_length

        return self.decoder(**decoder_kwargs)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_token_id: int,
        src_lengths: torch.Tensor | None = None,
        max_length: int = 50,
    ):
        """
        Decode tự hồi quy bằng chính `Decoder.forward`.
        `tgt` chỉ cần chứa token BOS ban đầu.
        """
        start_tokens = torch.full(
            (src.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=src.device,
        )
        return self.forward(
            src=src,
            tgt=start_tokens,
            src_lengths=src_lengths,
            teacher_forcing_ratio=0.0,
            max_length=max_length,
        )


__all__ = ["Seq2Seq"]
