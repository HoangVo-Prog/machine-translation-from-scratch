"""
src/models/decoder.py

Decoder cho mô hình Seq2Seq + Attention.
Tại mỗi bước thời gian:
    1. Embed token trước đó
    2. Tính context vector qua Attention (Bahdanau hoặc Luong)
    3. Đưa [embedded ; context] vào RNN cell
    4. Chiếu hidden state → logit phân phối từ vựng
    5. Kiểm tra stopping criteria: <EOS> hoặc max_length

Hỗ trợ 2 chế độ:
    - teacher_forcing : dùng ground-truth token làm input (lúc training)
    - autoregressive  : dùng token dự đoán làm input  (lúc inference)
"""

import torch
import torch.nn as nn
from src.models.layers import Embedding, VanillaRNN, LSTM, GRU
from src.models.activations import softmax


# Helpers 

def _build_cell(cell_type: str, input_size: int, hidden_size: int) -> nn.Module:
    """Factory RNN cell — giống encoder để tái sử dụng."""
    cell_type = cell_type.lower()
    if cell_type == "rnn":
        return VanillaRNN(input_size, hidden_size)
    elif cell_type == "lstm":
        return LSTM(input_size, hidden_size)
    elif cell_type == "gru":
        return GRU(input_size, hidden_size)
    else:
        raise ValueError(f"cell_type không hợp lệ: '{cell_type}'. Chọn 'rnn' | 'lstm' | 'gru'.")


# Decoder

class Decoder(nn.Module):
    """
    Decoder của Seq2Seq.

    Luồng dữ liệu tại mỗi bước t (theo diagram):
        token_(t-1)
            │
            ▼
        Embedding
            │  embedded_t  [batch, embed_dim]
            ▼
        ┌──────────────────────────────────┐
        │  Attention(hidden_(t-1),         │
        │            encoder_outputs)      │  ← Bahdanau hoặc Luong
        │  → context_t  [batch, hidden]    │
        └──────────────────────────────────┘
            │
            ▼  [embedded_t ; context_t]   [batch, embed_dim + hidden]
        RNN Cell  (VanillaRNN | LSTM | GRU)
            │
            ▼  hidden_t  [batch, hidden]
        Linear projection
            │
            ▼  logits_t  [batch, tgt_vocab_size]
        Softmax  →  token_t  (argmax khi inference)
            │
            ▼
        [Stopping: token_t == <EOS>  hoặc  t == max_length]

    Parameters
    ----------
    vocab_size   : int   – kích thước từ điển đích
    embed_dim    : int   – số chiều embedding
    hidden_size  : int   – phải bằng hidden_size của Encoder
    num_layers   : int   – số lớp RNN (phải bằng Encoder để khởi tạo từ final_hidden)
    cell_type    : str   – 'rnn' | 'lstm' | 'gru'
    attention    : nn.Module | None  – Bahdanau hoặc Luong (None = không dùng attention)
    dropout      : float
    eos_token_id : int   – index của token <EOS> trong từ điển đích
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "lstm",
        attention=None,
        dropout: float = 0.0,
        eos_token_id: int = 3,
    ):
        super().__init__()

        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.cell_type    = cell_type.lower()
        self.dropout_p    = dropout
        self.eos_token_id = eos_token_id
        self.vocab_size   = vocab_size
        self.attention    = attention

        # Embedding
        self.embedding = Embedding(vocab_size, embed_dim)

        # RNN cells
        # Input = [embedded ; context]  nên input_size = embed_dim + hidden_size
        # Nếu không có attention thì input_size = embed_dim
        rnn_input_size = embed_dim + hidden_size if attention is not None else embed_dim

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = rnn_input_size if i == 0 else hidden_size
            self.cells.append(_build_cell(cell_type, in_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

        # Projection: hidden → vocabulary logits
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    # private helpers

    def _unpack_hidden(self, final_hidden):
        """
        Chuyển final_hidden từ Encoder (shape [num_layers, batch, hidden])
        thành list of per-layer tuples để feed vào cells.
        """
        if self.cell_type == "lstm":
            h_n, c_n = final_hidden          # [num_layers, batch, hidden]
            return [(h_n[i], c_n[i]) for i in range(self.num_layers)]
        else:
            h_n = final_hidden               # [num_layers, batch, hidden]
            return [(h_n[i],) for i in range(self.num_layers)]

    def _step_rnn(self, x_t: torch.Tensor, states: list):
        """
        Chạy x_t qua toàn bộ RNN stack một bước.

        Returns
        -------
        h_top   : hidden state của layer trên cùng  [batch, hidden]
        states  : list trạng thái mới
        """
        new_states = []
        for layer_idx, cell in enumerate(self.cells):
            if self.cell_type == "lstm":
                h_prev, c_prev = states[layer_idx]
                h_t, c_t = cell.step(x_t, h_prev, c_prev)
                new_states.append((h_t, c_t))
            else:
                (h_prev,) = states[layer_idx]
                h_t = cell.step(x_t, h_prev)
                new_states.append((h_t,))

            x_t = self.dropout(h_t) if layer_idx < self.num_layers - 1 else h_t

        return x_t, new_states   # x_t == h_t của layer cuối

    def _get_top_hidden(self, states: list) -> torch.Tensor:
        """Lấy hidden state của layer trên cùng (dùng cho Attention query)."""
        if self.cell_type == "lstm":
            return states[-1][0]   # h của layer cuối
        else:
            return states[-1][0]

    def _pack_final_hidden(self, states: list):
        """Đóng gói states về dạng tensor [num_layers, batch, hidden]."""
        if self.cell_type == "lstm":
            h_n = torch.stack([s[0] for s in states], dim=0)
            c_n = torch.stack([s[1] for s in states], dim=0)
            return (h_n, c_n)
        else:
            return torch.stack([s[0] for s in states], dim=0)

    # Single step (dùng nội bộ và cho inference từng token)

    def step(
        self,
        token: torch.Tensor,
        states: list,
        encoder_outputs: torch.Tensor,
        src_mask=None,
    ):
        """
        Giải mã một bước thời gian.

        Parameters
        ----------
        token          : LongTensor [batch]          – token index của bước trước
        states         : list of per-layer tuples    – hidden state hiện tại
        encoder_outputs: FloatTensor [batch, src_len, hidden]
        src_mask       : BoolTensor [batch, src_len] | None  – True ở vị trí padding

        Returns
        -------
        logits         : FloatTensor [batch, vocab_size]
        new_states     : list of per-layer tuples
        attn_weights   : FloatTensor [batch, src_len] | None
        """
        # 1. Embed token
        embedded = self.embedding(token.unsqueeze(1)).squeeze(1)  # [batch, embed_dim]

        # 2. Attention → context vector
        attn_weights = None
        if self.attention is not None:
            query = self._get_top_hidden(states)               # [batch, hidden]
            context, attn_weights = self.attention(
                query, encoder_outputs, src_mask
            )                                                  # [batch, hidden]
            rnn_input = torch.cat([embedded, context], dim=-1) # [batch, embed+hidden]
        else:
            rnn_input = embedded

        # 3. RNN step
        h_top, new_states = self._step_rnn(rnn_input, states)

        # 4. Project → logits
        logits = self.output_projection(h_top)                 # [batch, vocab_size]

        return logits, new_states, attn_weights


    def forward(
        self,
        tgt: torch.Tensor,
        final_hidden,
        encoder_outputs: torch.Tensor,
        src_mask=None,
        teacher_forcing_ratio: float = 1.0,
        max_length: int = 50,
    ):
        """
        Parameters
        ----------
        tgt                   : LongTensor [batch, tgt_len]
                                Ground-truth target sequence (bao gồm <BOS> ở đầu).
                                Dùng để teacher forcing khi training.
                                Khi inference, chỉ cần truyền token <BOS>: shape [batch, 1].

        final_hidden          : output từ Encoder.forward() – dùng khởi tạo hidden state.

        encoder_outputs       : FloatTensor [batch, src_len, hidden]

        src_mask              : BoolTensor [batch, src_len] | None

        teacher_forcing_ratio : float [0, 1]
                                1.0 = luôn dùng ground-truth (training thuần)
                                0.0 = luôn autoregressive (inference)

        max_length            : int – số bước tối đa khi autoregressive
                                (stopping criteria 2: đủ max_length thì dừng)

        Returns
        -------
        all_logits   : FloatTensor [batch, decoded_len, vocab_size]
                       Logit phân phối từ vựng tại mỗi bước.

        all_attns    : FloatTensor [batch, decoded_len, src_len] | None
                       Attention weights tại mỗi bước (None nếu không dùng attention).
        """
        batch_size = encoder_outputs.size(0)
        device     = encoder_outputs.device

        # 1. Khởi tạo hidden từ Encoder final hidden
        states = self._unpack_hidden(final_hidden)

        # 2. Xác định số bước và token đầu vào
        #    - Training: tgt_len - 1 bước  (không dự đoán <BOS>)
        #    - Inference: max_length bước
        is_training   = tgt.size(1) > 1
        decode_steps  = (tgt.size(1) - 1) if is_training else max_length

        # Token đầu tiên luôn là <BOS> (cột 0 của tgt)
        current_token = tgt[:, 0]   # [batch]

        all_logits = []
        all_attns  = []

        # Theo dõi stopping criteria 1: câu nào đã gặp <EOS>
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(decode_steps):
            # Giải mã một bước
            logits, states, attn_w = self.step(
                current_token, states, encoder_outputs, src_mask
            )

            all_logits.append(logits)
            if attn_w is not None:
                all_attns.append(attn_w)

            # Chọn token tiếp theo
            use_teacher_forcing = (
                is_training
                and torch.rand(1).item() < teacher_forcing_ratio
                and t + 1 < tgt.size(1)
            )

            if use_teacher_forcing:
                # Teacher forcing: dùng ground-truth token
                current_token = tgt[:, t + 1]
            else:
                # Autoregressive: dùng token dự đoán (argmax)
                current_token = logits.argmax(dim=-1)  # [batch]

            # Stopping criteria 1: tất cả câu trong batch đã gặp <EOS>
            finished = finished | (current_token == self.eos_token_id)
            if finished.all():
                # Stopping criteria 2 (max_length) sẽ dừng vòng lặp for tự nhiên
                break

        # 3. Stack kết quả
        all_logits = torch.stack(all_logits, dim=1)   # [batch, decoded_len, vocab]
        all_attns  = (
            torch.stack(all_attns, dim=1)             # [batch, decoded_len, src_len]
            if all_attns else None
        )

        return all_logits, all_attns