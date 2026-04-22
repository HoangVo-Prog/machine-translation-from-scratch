import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attentions import BahdanauAttention, LuongAttention, ManualLinear


class ManualEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, pad_idx: int | None = None):
        super().__init__()
        self.pad_idx = pad_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class CustomGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.w_ir = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.w_iz = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.w_in = nn.Parameter(torch.empty(hidden_dim, input_dim))

        self.w_hr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.w_hz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.w_hn = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        self.b_ir = nn.Parameter(torch.empty(hidden_dim))
        self.b_iz = nn.Parameter(torch.empty(hidden_dim))
        self.b_in = nn.Parameter(torch.empty(hidden_dim))

        self.b_hr = nn.Parameter(torch.empty(hidden_dim))
        self.b_hz = nn.Parameter(torch.empty(hidden_dim))
        self.b_hn = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.hidden_dim)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset_gate = torch.sigmoid(
            inputs.matmul(self.w_ir.t()) + self.b_ir + hidden.matmul(self.w_hr.t()) + self.b_hr
        )
        update_gate = torch.sigmoid(
            inputs.matmul(self.w_iz.t()) + self.b_iz + hidden.matmul(self.w_hz.t()) + self.b_hz
        )
        candidate = torch.tanh(
            inputs.matmul(self.w_in.t())
            + self.b_in
            + reset_gate * (hidden.matmul(self.w_hn.t()) + self.b_hn)
        )
        next_hidden = (1.0 - update_gate) * candidate + update_gate * hidden
        return next_hidden


class CustomGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList() if bidirectional else None

        layer_input_dim = input_dim
        for _ in range(num_layers):
            self.forward_cells.append(CustomGRUCell(layer_input_dim, hidden_dim))
            if bidirectional:
                self.backward_cells.append(CustomGRUCell(layer_input_dim, hidden_dim))
            layer_input_dim = hidden_dim * self.num_directions

    def _run_direction(
        self,
        cell: CustomGRUCell,
        sequence_inputs: torch.Tensor,
        initial_hidden: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = range(sequence_inputs.size(1) - 1, -1, -1) if reverse else range(sequence_inputs.size(1))
        hidden = initial_hidden
        outputs = []

        for step in time_steps:
            hidden = cell(sequence_inputs[:, step, :], hidden)
            outputs.append(hidden.unsqueeze(1))

        if reverse:
            outputs.reverse()

        return torch.cat(outputs, dim=1), hidden

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.size(0)
        if hidden is None:
            hidden = inputs.new_zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_dim
            )

        layer_inputs = inputs
        final_hidden_states = []

        for layer_idx in range(self.num_layers):
            forward_hidden = hidden[layer_idx * self.num_directions]
            forward_outputs, forward_final = self._run_direction(
                self.forward_cells[layer_idx], layer_inputs, forward_hidden, reverse=False
            )
            direction_outputs = [forward_outputs]
            final_hidden_states.append(forward_final)

            if self.bidirectional:
                backward_hidden = hidden[layer_idx * self.num_directions + 1]
                backward_outputs, backward_final = self._run_direction(
                    self.backward_cells[layer_idx], layer_inputs, backward_hidden, reverse=True
                )
                direction_outputs.append(backward_outputs)
                final_hidden_states.append(backward_final)

            layer_outputs = torch.cat(direction_outputs, dim=-1)
            if layer_idx < self.num_layers - 1 and self.dropout > 0.0:
                layer_outputs = F.dropout(layer_outputs, p=self.dropout, training=self.training)
            layer_inputs = layer_outputs

        hidden_outputs = torch.stack(final_hidden_states, dim=0)
        return layer_inputs, hidden_outputs


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        pad_idx: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout

        self.embedding = ManualEmbedding(input_dim, embedding_dim, pad_idx=pad_idx)
        self.rnn = CustomGRU(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * self.num_directions

    def forward(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src_tokens)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training)
        encoder_outputs, hidden = self.rnn(embedded)
        return encoder_outputs, hidden


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        encoder_output_dim: int,
        decoder_hidden_dim: int,
        attention_type: str = "bahdanau",
        attention_dim: int | None = None,
        luong_score_method: str = "general",
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = ManualEmbedding(output_dim, embedding_dim, pad_idx=pad_idx)

        attention_type = attention_type.lower()
        if attention_type == "bahdanau":
            if attention_dim is None:
                attention_dim = decoder_hidden_dim
            self.attention = BahdanauAttention(
                encoder_hidden_dim=encoder_output_dim,
                decoder_hidden_dim=decoder_hidden_dim,
                attention_dim=attention_dim,
            )
        elif attention_type == "luong":
            self.attention = LuongAttention(
                encoder_hidden_dim=encoder_output_dim,
                decoder_hidden_dim=decoder_hidden_dim,
                score_method=luong_score_method,
            )
        else:
            raise ValueError("attention_type must be either 'bahdanau' or 'luong'")

        self.rnn = CustomGRU(
            input_dim=embedding_dim + encoder_output_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )
        self.output_projection = ManualLinear(
            decoder_hidden_dim + encoder_output_dim + embedding_dim, output_dim
        )

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_token)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training).unsqueeze(1)
        decoder_state = hidden[-1]
        context, attention_weights = self.attention(decoder_state, encoder_outputs, src_mask)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.output_projection(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        )
        return logits, hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

        self.hidden_bridge = ManualLinear(
            encoder.hidden_dim * encoder.num_directions, decoder.decoder_hidden_dim
        )

    def _init_decoder_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_hidden.size(1)
        encoder_hidden = encoder_hidden.view(
            self.encoder.num_layers,
            self.encoder.num_directions,
            batch_size,
            self.encoder.hidden_dim,
        )
        encoder_hidden = encoder_hidden.transpose(1, 2).reshape(
            self.encoder.num_layers,
            batch_size,
            self.encoder.hidden_dim * self.encoder.num_directions,
        )
        return torch.tanh(self.hidden_bridge(encoder_hidden))

    def create_src_mask(self, src_tokens: torch.Tensor) -> torch.Tensor:
        return src_tokens.eq(self.pad_idx)

    def forward(
        self,
        src_tokens: torch.Tensor,
        trg_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, trg_length = trg_tokens.shape
        vocab_size = self.decoder.output_dim
        device = src_tokens.device

        outputs = torch.zeros(batch_size, trg_length, vocab_size, device=device)
        attention_scores = torch.zeros(batch_size, trg_length - 1, src_tokens.size(1), device=device)

        src_mask = self.create_src_mask(src_tokens)
        encoder_outputs, encoder_hidden = self.encoder(src_tokens)
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)

        input_token = trg_tokens[:, 0]
        for timestep in range(1, trg_length):
            logits, decoder_hidden, attention_weights = self.decoder(
                input_token=input_token,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            outputs[:, timestep] = logits
            attention_scores[:, timestep - 1] = attention_weights

            teacher_force = random.random() < teacher_forcing_ratio
            next_token = logits.argmax(dim=-1)
            input_token = trg_tokens[:, timestep] if teacher_force else next_token

        return outputs, attention_scores

    @torch.no_grad()
    def greedy_decode(
        self,
        src_tokens: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_length: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = src_tokens.size(0)
        device = src_tokens.device

        src_mask = self.create_src_mask(src_tokens)
        encoder_outputs, encoder_hidden = self.encoder(src_tokens)
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)

        generated_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        collected_attentions = []
        input_token = generated_tokens[:, 0]

        for _ in range(max_length):
            logits, decoder_hidden, attention_weights = self.decoder(
                input_token=input_token,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            collected_attentions.append(attention_weights.unsqueeze(1))
            input_token = next_token.squeeze(1)

            if torch.all(input_token == eos_idx):
                break

        if collected_attentions:
            attention_scores = torch.cat(collected_attentions, dim=1)
        else:
            attention_scores = torch.empty(batch_size, 0, src_tokens.size(1), device=device)

        return generated_tokens, attention_scores
