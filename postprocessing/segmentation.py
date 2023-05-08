from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
from torch import Tensor

from data.MoNuSeg.ground_truth import NucleiInstances
from postprocessing.region_growing import RegionGrower
from evaluation.utils import is_batched


class InstanceExtractor:
    """Postprocesses the segmentation and extracts instances."""

    def __init__(self, seg: Tensor, cont: Tensor = None) -> None:
        """
        :param seg: Segmentation as tensor with shape (B, C, H, W)
        :param cont: Contours as tensor with shape (B, C, H, W)
        """
        seg_mask = seg > 0.5  # Thresholding operation to obtain the segmentation masks e {0, 1}^(B, C, H, W)
        if not is_batched(seg):
            raise ValueError(f"Seg should have shape (B, C, H, W). Got instead {seg.shape}")

        if cont is not None:
            cont_mask = cont > 0.5  # Thresholding operation to obtain the contour masks e [0, 1]^(B, C, H, W)
            if not is_batched(cont):
                raise ValueError(f"Cont should have shape (B, C, H, W). Got instead {cont.shape}")
            # Removes the contours from the segmentation in order to split instances
            self.mask = torch.logical_and(seg_mask, torch.logical_not(cont_mask))
        else:
            self.mask = seg_mask

        self.cuda = seg.is_cuda
        self.batch_size = seg.shape[0]

    def get_instances(self, impl: str = "skimage") -> Union[Tensor, Tuple[Tensor, ...]]:
        """Extracts the instance.
        :param
        impl: Implementation of the region growing algorithm for instance extraction. Can be either custom or skimage.
        """
        if impl == "skimage":
            if self.batch_size == 1:
                if self.cuda:
                    return NucleiInstances.from_mask(self.mask.cpu()).as_tensor().cuda()
                else:
                    return NucleiInstances.from_mask(self.mask).as_tensor()
            else:
                if self.cuda:
                    return tuple(NucleiInstances.from_mask(mask.cpu()).as_tensor().cuda() for mask in self.mask)
                else:
                    return tuple(NucleiInstances.from_mask(mask).as_tensor() for mask in self.mask)

        elif impl == "custom":
            if self.batch_size == 1:
                return RegionGrower(self.mask).get_regions()
            else:
                return tuple(RegionGrower(mask).get_regions() for mask in self.mask)
        else:
            raise ValueError(f"Impl should be custom or skimage. Got instead {impl}.")
