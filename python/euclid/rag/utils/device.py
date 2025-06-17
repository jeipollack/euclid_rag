"""Utility for loading current device type."""

import torch


def get_device() -> torch.device:
    """
    Return the torch device to use for embedding.

    Checks for available hardware acceleration in the following order:
    CUDA, MPS, then CPU.

    Returns
    -------
    torch.device
        The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
