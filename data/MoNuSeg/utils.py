from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch import Tensor
from torch.utils.data._utils.collate import default_collate, collate, default_collate_fn_map


def comp_dirs(dir: Path, comp_dir: Path, file_suffix: str) -> List[Path]:
    """Returns the files that are in dir but not in comp_dir. Only files with matching file_suffix are considered."""

    files = {f for f in dir.iterdir() if f.is_file() and f.suffix == file_suffix}
    comp_files = {f.stem for f in comp_dir.iterdir() if f.is_file() and f.suffix == file_suffix}
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


def get_bbox(instance: Union[np.ndarray, Tensor]) -> Tuple[int, int, int, int]:
    """Determines the bounding box (bbox) for the instance. The bbox coordinates are designed for indexing."""
    if isinstance(instance, np.ndarray):
        rows = instance.any(axis=1)
        cols = instance.any(axis=0)
        y = rows.nonzero()[0]
        x = cols.nonzero()[0]
        y_min, y_max = y[0], y[-1] + 1  # Designed for indexing -> +1
        x_min, x_max = x[0], x[-1] + 1  # Designed for indexing -> +1
        return y_min, y_max, x_min, x_max
    elif isinstance(instance, Tensor):
        rows = instance.any(dim=1)
        cols = instance.any(dim=0)
        y = rows.nonzero(as_tuple=True)[0]
        x = cols.nonzero(as_tuple=True)[0]
        y_min, y_max = int(y[0]), int(y[-1] + 1)  # Designed for indexing -> +1
        x_min, x_max = int(x[0]), int(x[-1] + 1)  # Designed for indexing -> +1
        return y_min, y_max, x_min, x_max
    else:
        raise TypeError(f"Instance should be of type np.ndarray or tensor. Got instead {type(instance)}")


def center_of_mass(mask: Union[np.ndarray, Tensor]) -> Tuple[float, float]:
    """
    Calculates the center of mass for a binary mask.
    """
    if isinstance(mask, np.ndarray):
        rows, cols = mask.nonzero()
    elif isinstance(mask, Tensor):
        rows, cols = mask.nonzero(as_tuple=True)
    else:
        raise TypeError(f"Instance should be of type np.ndarray or tensor. Got instead {type(mask)}")
    y = rows.sum() / rows.shape[0]
    x = cols.sum() / cols.shape[0]
    return float(y), float(x)


def prob_to_mask(prob: np.ndarray, thresh: Union[str, float] = 0.5) -> np.ndarray:
    """
    Converts probabilities into a binary mask.

    Threshold value is either inferred using the Otsu method (Otsu 1979) or can be given as a float value.
    """
    # Check if conversion to mask is necessary:
    if issubclass(prob.dtype.type, np.floating):
        if isinstance(thresh, (str, float)):
            if isinstance(thresh, str):
                if thresh == "otsu":
                    thresh = threshold_otsu(prob)
                else:
                    raise ValueError(f"Thresh as string should be 'otsu'. Got instead {thresh}.")
            mask = prob >= thresh
        else:
            raise TypeError(f"Thresh should be string or float. Got instead {type(thresh)}.")
    else:
        mask = prob

    return mask


def cuda_tensor_to_ndarray(tensor: Tensor) -> np.ndarray:
    """
    Converts a tensor on cuda into a ndarray on the cpu.
    """
    tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.squeeze().numpy()
