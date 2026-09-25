"""Device selection shared by every entry point (standard library + torch only)."""

import torch


def get_device(requested: str = "auto") -> str:
    """Resolve a device name: "auto" picks the GPU when available, else the CPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
