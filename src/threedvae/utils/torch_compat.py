from __future__ import annotations


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    HAS_TORCH = False


def require_torch():
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for learned instance tokenizer training. "
            "Install the optional training dependency, e.g. `pip install -e .[train]`."
        )
    return torch, nn, F, Dataset, DataLoader

