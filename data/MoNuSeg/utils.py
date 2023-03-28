from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate, collate, default_collate_fn_map


def comp_dirs(dir: Path, comp_dir: Path) -> List[Path]:
    """Returns the files that are in dir but not in comp_dir."""

    files = {f for f in dir.iterdir() if f.is_file()}
    comp_files = {f.stem for f in comp_dir.iterdir() if f.is_file()}
    missing_files = []
    for file in files:
        if file.stem not in comp_files:
            missing_files.append(file)
    return missing_files


def has_same_shape(batch: Tuple[Tensor]) -> bool:
    """Checks whether all tensors in a batch have the same shape."""
    shape = batch[0].shape
    for elem in batch[1:]:
        if elem.shape != shape:
            return False
    return True


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    """Adapted default collate_tensor_fn to handle tensors with varying shape (C, H, W)."""
    if has_same_shape(batch):
        # Default collate_tensor_fn
        elem = batch[0]
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    else:
        # Adapted collate_tensor_fn
        return batch  # batch: Tuple[Tensor]


def custom_collate(batch):
    """Allows batching of tensors with varying shapes.

    Background: The number of nuclei instances varies from image to image. Nuclei instances are encoded as a
    tensor with shape (#nuclei instances, H, W). Thus, default batching of nuclei instances fails due to
    varying tensor shapes."""
    default_collate_fn_map.update({Tensor: collate_tensor_fn})
    return collate(batch, collate_fn_map=default_collate_fn_map)
