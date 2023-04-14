import torch
from torch import Tensor


def is_empty(tensor: Tensor) -> bool:
    """Checks whether a tensor is empty."""
    if tensor.numel() == 0:
        return True
    else:
        return False


def is_batched(tensor: Tensor) -> bool:
    """Checks whether a tensor has shape (B, C, H, W)."""
    if tensor.dim() == 4:
        return True
    else:
        return False


def tensor_intersection(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the intersection of given tensors"""
    if a.size() != b.size():
        raise ValueError(
            f"A and b must have the same shape. However got {a.size()} and {b.size()} for a and b, respectively.")
    return torch.sum(a * b)


def tensor_union(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the union of given tensors"""
    if a.size() != b.size():
        raise ValueError(
            f"A and b must have the same shape. However got {a.size()} and {b.size()} for a and b, respectively.")
    return torch.sum(a + b)


def intersection_over_union(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the Intersection over Union (IoU)"""
    return tensor_intersection(a, b) / tensor_union(a, b)
