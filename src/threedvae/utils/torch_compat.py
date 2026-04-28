from __future__ import annotations


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
    try:
        import torch_npu  # noqa: F401

        HAS_TORCH_NPU = True
    except ImportError:  # pragma: no cover - exercised outside Ascend environments
        torch_npu = None
        HAS_TORCH_NPU = False
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    torch_npu = None
    HAS_TORCH = False
    HAS_TORCH_NPU = False


def require_torch():
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for learned instance tokenizer training. "
            "Install the optional training dependency, e.g. `pip install -e .[train]`."
        )
    return torch, nn, F, Dataset, DataLoader
