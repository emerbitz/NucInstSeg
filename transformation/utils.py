import torch


def remove_zero_stacks(t: torch.Tensor) -> torch.Tensor:
    """Discards all-zero stacks from a torch.Tensor"""
    return torch.stack([stack for stack in t if torch.is_nonzero(stack.any())])


def ceiling(a: int, b: int) -> int:
    """Performs the ceiling division of a by b."""
    return -(a // -b)