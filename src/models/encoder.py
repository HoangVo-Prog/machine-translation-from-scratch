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
        states = []
        for i in range(self.num_layers):
            # Gọi trực tiếp hàm init_hidden của cell tương ứng
            layer_state = self.cells[i].init_hidden(batch_size, device)
            # Đảm bảo trả về dạng tuple để nhất quán
            if not isinstance(layer_state, tuple):
                layer_state = (layer_state,)
            states.append(layer_state)
        return states


    def forward(self, src: torch.Tensor, src_lengths=None):
        batch_size, src_len = src.size()
        device = src.device

        embedded = self.embedding(src)
        states = self._init_hidden(batch_size, device)
        encoder_outputs = []
        # Lưu states tại mỗi bước để gather về sau (chỉ dùng khi src_lengths != None)
        all_states = []  # list[src_len] of states

        for t in range(src_len):
            x_t = embedded[:, t, :]
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
            states = new_states
            all_states.append(states)          # lưu snapshot tại t
            encoder_outputs.append(x_t)

        encoder_outputs = torch.stack(encoder_outputs, dim=1)

        # dùng src_lengths để chọn đúng snapshot
        if src_lengths is not None:
            final_hidden = self._gather_states_at(all_states, src_lengths, device)
        else:
            final_hidden = self._pack_final_hidden(states)

        return encoder_outputs, final_hidden


    def _gather_states_at(self, all_states, src_lengths, device):
        """
        Với mỗi sample i trong batch, lấy states tại bước t = src_lengths[i] - 1.
        all_states : list[src_len] of list[num_layers] of tuple(h, ?) 
        """
        lengths = src_lengths.to(device)
        batch_size = lengths.size(0)

        if self.cell_type == "lstm":
            h_layers, c_layers = [], []
            for layer_idx in range(self.num_layers):
                # Stack h của layer này theo thời gian: [src_len, batch, hidden]
                h_t_stack = torch.stack([all_states[t][layer_idx][0] for t in range(len(all_states))], dim=0)
                c_t_stack = torch.stack([all_states[t][layer_idx][1] for t in range(len(all_states))], dim=0)
                # Gather tại vị trí cuối thực của từng sample
                last_idx = (lengths - 1).clamp(min=0, max=h_t_stack.size(0) - 1)
                # idx: [batch] → [batch, 1, hidden] để gather trên dim=0
                # Dùng vòng lặp đơn giản cho rõ ràng
                h_last = h_t_stack[last_idx, torch.arange(batch_size, device=device)]
                c_last = c_t_stack[last_idx, torch.arange(batch_size, device=device)]
                h_layers.append(h_last)
                c_layers.append(c_last)
            h_n = torch.stack(h_layers, dim=0)  # [num_layers, batch, hidden]
            c_n = torch.stack(c_layers, dim=0)
            return (h_n, c_n)
        else:
            h_layers = []
            for layer_idx in range(self.num_layers):
                h_t_stack = torch.stack([all_states[t][layer_idx][0] for t in range(len(all_states))], dim=0)
                last_idx = (lengths - 1).clamp(min=0, max=h_t_stack.size(0) - 1)
                h_last = h_t_stack[last_idx, torch.arange(batch_size, device=device)]
                h_layers.append(h_last)
            return torch.stack(h_layers, dim=0)


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