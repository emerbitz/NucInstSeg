from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor


class SegLoss:
    def __init__(self):
        self.l0 = nn.BCEWithLogitsLoss()
        self.l1 = nn.BCEWithLogitsLoss()

    def __call__(self, pred: Dict[str, Union[None, Tensor]], gt: Dict[str, Union[None, str, Tensor]]) -> Tensor:
        return self.l0(pred["seg_mask"], gt["seg_mask"]) + self.l1(pred["cont_mask"], gt["cont_mask"])


class DistLoss:
    def __init__(self):
        self.l0 = nn.MSELoss()

    def __call__(self, pred: Dict[str, Union[None, Tensor]], gt: Dict[str, Union[None, str, Tensor]]) -> Tensor:
        return self.l0(pred["dist_map"], gt["dist_map"])


class HVLoss:
    def __init__(self):
        self.l0 = nn.BCEWithLogitsLoss()
        self.l1 = nn.MSELoss()

    def __call__(self, pred: Dict[str, Union[None, Tensor]], gt: Dict[str, Union[None, str, Tensor]]) -> Tensor:
        return self.l0(pred["seg_mask"], gt["seg_mask"]) + self.l1(torch.tanh(pred["hv_map"]), gt["hv_map"])
