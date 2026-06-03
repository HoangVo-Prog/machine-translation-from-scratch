"""
Miscellaneous utilities.
"""

import importlib
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_from_string(dotted_path: str):
    """
    Import an object from a dotted path string like 'src.factories:build_model'.
    Supports both '.' and ':' as module/attribute separators.
    """
    if ":" in dotted_path:
        module_path, attr = dotted_path.rsplit(":", 1)
    else:
        module_path, attr = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
