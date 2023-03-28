from typing import Tuple, Union
import torch
from torch import Tensor

from data.MoNuSeg.ground_truth import NucleiInstances
from evaluation.utils import is_batched

class NucleiSplitter:
    """Splits clustered nuclei through removal of the nuclei contours into nuclei instances.
    Parameters:
        seg: Segmentation as tensor with shape (B, C, H, W)
        cont: Contours as tensor with shape (B, C, H, W)
    """
    def __init__(self, seg: Tensor, cont: Tensor):
        seg_mask = seg > 0.5  # Thresholding operation to obtain the segmentation masks e [0, 1]^(B, C, H, W)
        cont_mask = cont > 0.5 # Thresholding operation to obtain the contour masks e [0, 1]^(B, C, H, W)
        self._instance_mask = self.split(seg_mask, cont_mask)

        if not is_batched(seg):
            raise ValueError("Seg should have shape (B, C, H, W).")
        if not is_batched(cont):
            raise ValueError("Cont should have shape (B, C, H, W).")
        self.batch_size = seg.shape[0]

    @staticmethod
    def split(seg_mask: Tensor, cont_mask: Tensor) -> Tensor:
        """Removes the contours from the segmented nuclei."""
        return torch.logical_and(seg_mask, torch.logical_not(cont_mask))

    @property
    def instance_mask(self):
        return self._instance_mask

    def to_instances(self) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Extracts the nuclei instances from the instance mask."""
        if self.batch_size == 1:
            return NucleiInstances.from_seg_mask(self._instance_mask).as_tensor()
        else:
            return tuple(NucleiInstances.from_seg_mask(mask).as_tensor() for mask in self._instance_mask)

