import torch
from torch import Tensor


def remove_zero_stacks(t: Tensor) -> Tensor:
    """Discards all-zero stacks from the tensor"""
    return torch.stack([stack for stack in t if torch.is_nonzero(stack.any())])


# def add_zero_stacks(t: Tensor, num_stacks: int) -> Tensor:
#     """Adds the specified number of all-zero stacks to the tensor"""
#     _, h, w = t.shape
#     zero_stacks = torch.zeros((num_stacks, h, w), dtype=torch.bool)
#     return torch.cat((t, zero_stacks))
#
#
# def fill_with_zero_stacks(t: Tensor, max_channels: int):
#     """Adds all-zero stacks to the tensor until the tensor has a shape of (max_channels, H, W)"""
#     channels, *_ = t.shape
#     if channels > max_channels:
#         raise ValueError(f"The channel number of t should be less or equal than max_channels. However, t has {channels} channels.")
#     elif channels == max_channels:
#         return t
#     else:
#         return add_zero_stacks(t, num_stacks=max_channels - channels)
#

def ceiling(a: int, b: int) -> int:
    """Performs the ceiling division of a by b."""
    return -(a // -b)