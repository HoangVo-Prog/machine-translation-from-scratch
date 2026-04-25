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


def _build_cell(cell_type: str, input_size: int, hidden_size: int) -> nn.Module:
    cell_type = cell_type.lower()
    if cell_type == "rnn":
        return VanillaRNN(input_size, hidden_size)
    if cell_type == "lstm":
        return LSTM(input_size, hidden_size)
    if cell_type == "gru":
        return GRU(input_size, hidden_size)
    raise ValueError(f"cell_type không hợp lệ: '{cell_type}'. Chọn 'rnn' | 'lstm' | 'gru'.")


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "lstm",
        attention=None,
        dropout: float = 0.0,
        eos_token_id: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.dropout_p = dropout
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.attention = attention

        self.embedding = Embedding(vocab_size, embed_dim)

        rnn_input_size = embed_dim + hidden_size if attention is not None else embed_dim

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = rnn_input_size if i == 0 else hidden_size
            self.cells.append(_build_cell(cell_type, in_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def _unpack_hidden(self, final_hidden):
        if self.cell_type == "lstm":
            h_n, c_n = final_hidden
            return [(h_n[i], c_n[i]) for i in range(self.num_layers)]

        h_n = final_hidden
        return [(h_n[i],) for i in range(self.num_layers)]

    def _step_rnn(self, x_t: torch.Tensor, states: list):
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

        return x_t, new_states

    def _get_top_hidden(self, states: list) -> torch.Tensor:
        return states[-1][0]

    def _pack_final_hidden(self, states: list):
        if self.cell_type == "lstm":
            h_n = torch.stack([s[0] for s in states], dim=0)
            c_n = torch.stack([s[1] for s in states], dim=0)
            return h_n, c_n

        return torch.stack([s[0] for s in states], dim=0)

    def step(
        self,
        token: torch.Tensor,
        states: list,
        encoder_outputs: torch.Tensor,
        src_mask=None,
    ):
        embedded = self.embedding(token.unsqueeze(1)).squeeze(1)

        attn_weights = None

        if self.attention is not None:
            query = self._get_top_hidden(states)
            context, attn_weights = self.attention(query, encoder_outputs, src_mask)
            rnn_input = torch.cat([embedded, context], dim=-1)
        else:
            rnn_input = embedded

        h_top, new_states = self._step_rnn(rnn_input, states)
        logits = self.output_projection(h_top)

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
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        states = self._unpack_hidden(final_hidden)

        is_training = tgt.size(1) > 1
        decode_steps = (tgt.size(1) - 1) if is_training else max_length

        current_token = tgt[:, 0]

        all_logits = []
        all_attns = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(decode_steps):
            logits, states, attn_w = self.step(
                current_token,
                states,
                encoder_outputs,
                src_mask,
            )

            all_logits.append(logits)

            if attn_w is not None:
                all_attns.append(attn_w)

            use_teacher_forcing = (
                is_training
                and torch.rand(1, device=device).item() < teacher_forcing_ratio
                and t + 1 < tgt.size(1)
            )

            if use_teacher_forcing:
                current_token = tgt[:, t + 1]
            else:
                current_token = logits.argmax(dim=-1)

            # Quan trọng:
            # Không early-stop khi training, vì sẽ làm logits ngắn hơn labels.
            if not is_training:
                finished = finished | (current_token == self.eos_token_id)
                if finished.all():
                    break

        all_logits = torch.stack(all_logits, dim=1)

        all_attns = (
            torch.stack(all_attns, dim=1)
            if all_attns
            else None
        )

        return all_logits, all_attns