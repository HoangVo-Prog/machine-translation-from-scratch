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
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def init_hidden(self, batch_size, device=None):
        if device is None: device = self.W_xh.device
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def step(self, x_t, h_prev):
        # x_t: [batch, input_size], h_prev: [batch, hidden_size]
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
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
        # Khởi tạo trọng số gộp để tính toán nhanh hơn
        self.W_all = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size) * 0.01)
        self.b_all = nn.Parameter(torch.zeros(4 * hidden_size))

    def init_hidden(self, batch_size, device=None):
        if device is None: device = self.W_all.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h, c)

    def step(self, x_t, h_prev, c_prev=None):
        # Xử lý trường hợp truyền tuple (từ Encoder) hoặc truyền rời (từ Decoder)
        if c_prev is None and isinstance(h_prev, tuple):
            h_prev, c_prev = h_prev
            
        concat = torch.cat((x_t, h_prev), dim=1) # [batch, input + hidden]
        gates = concat @ self.W_all + self.b_all # [batch, 4 * hidden]
        
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
        
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

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
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        self.W_r = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        self.W_h = nn.Parameter(torch.randn(hidden_size, concat_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def init_hidden(self, batch_size, device=None): # Thêm batch_size
        if device is None:
            device = self.W_z.device
        # Khởi tạo shape [batch_size, hidden_size]
        return torch.zeros((batch_size, self.hidden_size), device=device)

    def step(self, x_t, h_prev):
        x_t = x_t.to(self.W_z.device)
        h_prev = h_prev.to(self.W_z.device)

        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        concat = torch.cat((h_prev, x_t), dim=1)

        z_t = sigmoid(concat @ self.W_z.T + self.b_z)
        r_t = sigmoid(concat @ self.W_r.T + self.b_r)

        concat_reset = torch.cat((r_t * h_prev, x_t), dim=1)
        h_tilde = tanh(concat_reset @ self.W_h.T + self.b_h)

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