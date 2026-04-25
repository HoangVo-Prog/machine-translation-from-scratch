import torch
import torch.nn as nn
from src.models.activations import sigmoid, tanh


class Embedding(nn.Module):
    """
    Embedding from scratch:
    - không dùng nn.Embedding
    - tự lưu ma trận embedding W
    - lookup bằng indexing
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)

    def forward(self, inputs):
        """
        inputs:
            - int
            - list[int]
            - tensor shape ()
            - tensor shape (seq_len,)

        return:
            - 1 token -> shape (embedding_dim,)
            - nhiều token -> shape (seq_len, embedding_dim)
        """
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long, device=self.W.device)
        else:
            inputs = inputs.to(dtype=torch.long, device=self.W.device)

        return self.W[inputs]


class VanillaRNN(nn.Module):
    """
    Vanilla RNN from scratch:
        h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, 1))

    def init_hidden(self, device=None):
        if device is None:
            device = self.W_xh.device
        return torch.zeros((self.hidden_size, 1), device=device)

    def step(self, x_t, h_prev):
        x_t = x_t.view(-1, 1).to(self.W_xh.device)
        h_prev = h_prev.to(self.W_xh.device)

        h_t = tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)
        return h_t

    def forward(self, inputs, init_state=None):
        """
        inputs:
            - tensor shape (seq_len, input_size)
            - hoặc list các vector shape (input_size,) / (input_size, 1)

        return:
            outputs: list[h_t]
            final_state: h_T
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() != 2:
                raise ValueError("inputs tensor phải có shape (seq_len, input_size)")
            sequence = [inputs[t] for t in range(inputs.size(0))]
            device = inputs.device
        else:
            if len(inputs) == 0:
                raise ValueError("inputs không được rỗng")
            sequence = inputs
            device = sequence[0].device

        h_t = init_state if init_state is not None else self.init_hidden(device=device)

        outputs = []
        for x_t in sequence:
            h_t = self.step(x_t, h_t)
            outputs.append(h_t)

        return outputs, h_t


class LSTM(nn.Module):
    """
    LSTM from scratch:
        f_t = sigmoid(W_f [h_{t-1}; x_t] + b_f)
        i_t = sigmoid(W_i [h_{t-1}; x_t] + b_i)
        c_tilde = tanh(W_c [h_{t-1}; x_t] + b_c)
        c_t = f_t * c_{t-1} + i_t * c_tilde
        o_t = sigmoid(W_o [h_{t-1}; x_t] + b_o)
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = hidden_size + input_size

        self.W_f = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size, 1))

        self.W_i = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(hidden_size, 1))

        self.W_c = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_c = nn.Parameter(torch.zeros(hidden_size, 1))

        self.W_o = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(hidden_size, 1))

    def init_hidden(self, device=None):
        if device is None:
            device = self.W_f.device

        h0 = torch.zeros((self.hidden_size, 1), device=device)
        c0 = torch.zeros((self.hidden_size, 1), device=device)
        return (h0, c0)

    def step(self, x_t, prev_state):
        x_t = x_t.view(-1, 1).to(self.W_f.device)
        h_prev, c_prev = prev_state
        h_prev = h_prev.to(self.W_f.device)
        c_prev = c_prev.to(self.W_f.device)

        concat = torch.cat((h_prev, x_t), dim=0)

        f_t = sigmoid(self.W_f @ concat + self.b_f)
        i_t = sigmoid(self.W_i @ concat + self.b_i)
        c_tilde = tanh(self.W_c @ concat + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde

        o_t = sigmoid(self.W_o @ concat + self.b_o)
        h_t = o_t * tanh(c_t)

        return (h_t, c_t)

    def forward(self, inputs, init_state=None):
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() != 2:
                raise ValueError("inputs tensor phải có shape (seq_len, input_size)")
            sequence = [inputs[t] for t in range(inputs.size(0))]
            device = inputs.device
        else:
            if len(inputs) == 0:
                raise ValueError("inputs không được rỗng")
            sequence = inputs
            device = sequence[0].device

        state = init_state if init_state is not None else self.init_hidden(device=device)

        outputs = []
        for x_t in sequence:
            state = self.step(x_t, state)
            h_t, _ = state
            outputs.append(h_t)

        return outputs, state


class GRU(nn.Module):
    """
    GRU from scratch:
        z_t = sigmoid(W_z [h_{t-1}; x_t] + b_z)
        r_t = sigmoid(W_r [h_{t-1}; x_t] + b_r)
        h_tilde = tanh(W_h [r_t * h_{t-1}; x_t] + b_h)
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = hidden_size + input_size

        self.W_z = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size, 1))

        self.W_r = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size, 1))

        self.W_h = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, 1))

    def init_hidden(self, device=None):
        if device is None:
            device = self.W_z.device
        return torch.zeros((self.hidden_size, 1), device=device)

    def step(self, x_t, h_prev):
        x_t = x_t.to(self.W_z.device)
        h_prev = h_prev.to(self.W_z.device)

        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        concat = torch.cat((h_prev, x_t), dim=1)

        z_t = sigmoid(concat @ self.W_z.T + self.b_z.T)
        r_t = sigmoid(concat @ self.W_r.T + self.b_r.T)

        concat_reset = torch.cat((r_t * h_prev, x_t), dim=1)
        h_tilde = tanh(concat_reset @ self.W_h.T + self.b_h.T)

        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

    def forward(self, inputs, init_state=None):
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() != 2:
                raise ValueError("inputs tensor phải có shape (seq_len, input_size)")
            sequence = [inputs[t] for t in range(inputs.size(0))]
            device = inputs.device
        else:
            if len(inputs) == 0:
                raise ValueError("inputs không được rỗng")
            sequence = inputs
            device = sequence[0].device

        h_t = init_state if init_state is not None else self.init_hidden(device=device)

        outputs = []
        for x_t in sequence:
            h_t = self.step(x_t, h_t)
            outputs.append(h_t)

        return outputs, h_t