from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from data.MoNuSeg.conversion import NucleiInstances
from postprocessing.postprocesses_base import Postprocess


class SegPostProcess(Postprocess):
    """
    Postprocessing based on the segmentation and/or the contour.
    """

    def __init__(self, seg_params: Optional[dict] = None, actv: bool = True,
                 mode: Literal["baseline", "contour", "yang"] = "contour"):

        self.actv = actv
        assert mode in ["baseline", "contour", "yang"], f"Mode should be baseline, contour or yang. Got instead {mode}."
        self.mode = mode
        self.requires_cont = True if mode == "contour" else False
        self.seg_params = seg_params

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg = pred["seg_mask"]
        if self.actv:
            seg = torch.sigmoid(seg)

        if self.requires_cont:
            cont = pred["cont_mask"]
            if self.actv:
                cont = torch.sigmoid(cont)
            return self.postprocess(seg, cont)
        else:
            return self.postprocess(seg)

    def postprocess_fn(self, seg: Tensor, cont: Optional[Tensor] = None) -> Tensor:
        return NucleiInstances.from_seg(seg, cont, seg_params=self.seg_params, mode=self.mode).as_tensor()


class DistPostProcess(Postprocess):
    """
    Postprocessing based on the distance map.

    The postprocessing strategy by Naylor et al. 2019 is used to identify nuclei instances.
    """

    def __init__(self, dist_params: Optional[dict] = None):
        self.dist_params = dist_params

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        return self.postprocess(pred["dist_map"])

    def postprocess_fn(self, dist: Tensor) -> Tensor:
        return NucleiInstances.from_dist_map(dist, self.dist_params).as_tensor()


class HVPostProcess(Postprocess):
    """
   Postprocessing based on the horizontal and vertical distance map.

   The postprocessing strategy by Graham et al. 2019 is used to identify nuclei instances.
   """

    def __init__(self, hv_params: Optional[dict] = None, actv: bool = True, exprmtl: bool = False):
        self.actv = actv
        self.exprmtl = exprmtl
        self.hv_params = hv_params

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg = pred["seg_mask"]
        if self.actv:
            seg = torch.sigmoid(seg)

        return self.postprocess(seg, pred["hv_map"])

    def postprocess_fn(self, seg: Tensor, hv_map: Tensor) -> Tensor:
        return NucleiInstances.from_hv_map(hv_map, seg, hv_params=self.hv_params, exprmtl=self.exprmtl).as_tensor()
