import torch

def sigmoid(x):
    """
    Compute the sigmoid of x.
    """
    return 1 / (1 + torch.exp(-x))

def relu(x):
    """
    Compute the ReLU of x.
    """
    return torch.maximum(torch.tensor(0.0, device=x.device), x)

def tanh(x):
    """
    Compute the hyperbolic tangent of x.
    """
    return torch.tanh(x)

def softmax(x):
    """
    Compute the softmax of x.
    """
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)


def log_softmax(x):
    """
    Compute the log-softmax of x.
    """
    shifted = x - torch.max(x, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    return shifted - log_sum_exp

__all__ = [
    'sigmoid',
    'relu',
    'tanh',
    'softmax',
    'log_softmax'
]
