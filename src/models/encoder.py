"""
src/models/encoder.py

Encoder cho mô hình Seq2Seq.
Nhận một chuỗi token indices, embed chúng, rồi chạy qua stack RNN cells
để tạo ra toàn bộ hidden states và final hidden state (dùng để khởi tạo Decoder).

Hỗ trợ: VanillaRNN | LSTM | GRU  (chọn qua tham số cell_type)
"""

import torch
import torch.nn as nn
from src.models.layers import Embedding, VanillaRNN, LSTM, GRU


# Helpers 

def _build_cell(cell_type: str, input_size: int, hidden_size: int) -> nn.Module:
    """Factory: trả về một RNN cell theo cell_type."""
    cell_type = cell_type.lower()
    if cell_type == "rnn":
        return VanillaRNN(input_size, hidden_size)
    elif cell_type == "lstm":
        return LSTM(input_size, hidden_size)
    elif cell_type == "gru":
        return GRU(input_size, hidden_size)
    else:
        raise ValueError(f"cell_type không hợp lệ: '{cell_type}'. Chọn 'rnn' | 'lstm' | 'gru'.")


# Encoder

class Encoder(nn.Module):
    """
    Encoder của Seq2Seq.

    Luồng dữ liệu (theo diagram):
        token indices  →  Embedding  →  embedding vectors
                       →  RNN stack  →  all hidden states  +  final hidden state

    Parameters
    ----------
    vocab_size   : int   – kích thước từ điển nguồn
    embed_dim    : int   – số chiều embedding
    hidden_size  : int   – số chiều hidden state của mỗi RNN cell
    num_layers   : int   – số lớp RNN xếp chồng (stacked RNN)
    cell_type    : str   – 'rnn' | 'lstm' | 'gru'  (mặc định 'lstm')
    dropout      : float – xác suất dropout giữa các layer (chỉ áp dụng khi num_layers > 1)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.cell_type   = cell_type.lower()
        self.dropout_p   = dropout

        # Embedding layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Stack RNN cells
        # Layer đầu nhận embed_dim, các layer sau nhận hidden_size
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size
            self.cells.append(_build_cell(cell_type, in_size, hidden_size))

        # Dropout giữa các layer - không ảnh hưởng logic RNN
        self.dropout = nn.Dropout(dropout)


    def _init_hidden(self, batch_size: int, device: torch.device):
        """
        Khởi tạo hidden state bằng zeros cho mỗi layer.

        Returns
        -------
        list of (h_0,) hoặc (h_0, c_0) tùy cell_type
        """
        states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            if self.cell_type == "lstm":
                c = torch.zeros(batch_size, self.hidden_size, device=device)
                states.append((h, c))
            else:
                states.append((h,))   # tuple 1 phần tử cho đồng nhất interface
        return states


    def forward(self, src: torch.Tensor, src_lengths=None):
        """
        Parameters
        ----------
        src        : LongTensor [batch, src_len]  – token indices chuỗi nguồn
        src_lengths: (tuỳ chọn) LongTensor [batch] – độ dài thực của mỗi câu
                     (dùng để mask padding trong attention sau này)

        Returns
        -------
        encoder_outputs : FloatTensor [batch, src_len, hidden_size]
                          Hidden state của layer cuối tại mỗi bước thời gian.
                          Attention sẽ dùng tensor này.

        final_hidden    : tuple  –  trạng thái cuối của từng layer
            - GRU / RNN : tuple of Tensor  [num_layers, batch, hidden_size]
            - LSTM       : (h_n, c_n) mỗi cái shape [num_layers, batch, hidden_size]
        """
        batch_size, src_len = src.size()
        device = src.device

        # 1. Embedding: [batch, src_len] → [batch, src_len, embed_dim]
        embedded = self.embedding(src)          # gọi forward của Embedding from scratch

        # 2. Khởi tạo hidden states
        states = self._init_hidden(batch_size, device)

        # 3. Chạy qua từng bước thời gian
        #    encoder_outputs lưu hidden state của layer CUỐI tại mỗi step
        encoder_outputs = []

        for t in range(src_len):
            x_t = embedded[:, t, :]             # [batch, embed_dim]

            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                # Lấy hidden (và cell) state của layer này
                if self.cell_type == "lstm":
                    h_prev, c_prev = states[layer_idx]
                    h_t, c_t = cell.step(x_t, h_prev, c_prev)
                    new_states.append((h_t, c_t))
                else:
                    (h_prev,) = states[layer_idx]
                    h_t = cell.step(x_t, h_prev)
                    new_states.append((h_t,))

                # Input của layer tiếp theo là h_t hiện tại (qua dropout nếu không phải layer cuối)
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(h_t)
                else:
                    x_t = h_t   # không dropout ở layer cuối

            states = new_states
            encoder_outputs.append(x_t)         # x_t == h_t của layer cuối

        # 4. Stack encoder_outputs: list[src_len × [batch, hidden]] → [batch, src_len, hidden]
        encoder_outputs = torch.stack(encoder_outputs, dim=1)

        # 5. Đóng gói final_hidden để truyền sang Decoder
        final_hidden = self._pack_final_hidden(states)

        return encoder_outputs, final_hidden

    def _pack_final_hidden(self, states):
        """
        Gom hidden states của tất cả layer thành tensor để dễ truyền sang Decoder.

        GRU / RNN  →  h_n  : [num_layers, batch, hidden_size]
        LSTM       →  (h_n, c_n) mỗi cái [num_layers, batch, hidden_size]
        """
        if self.cell_type == "lstm":
            h_list = [s[0] for s in states]    # h của từng layer
            c_list = [s[1] for s in states]    # c của từng layer
            h_n = torch.stack(h_list, dim=0)   # [num_layers, batch, hidden]
            c_n = torch.stack(c_list, dim=0)
            return (h_n, c_n)
        else:
            h_list = [s[0] for s in states]
            return torch.stack(h_list, dim=0)  # [num_layers, batch, hidden]