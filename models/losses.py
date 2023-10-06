from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveLoss:
    def __init__(self, double_main_task: bool = False, auxiliary_task: bool = True):
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Tensor:
        loss = torch.tensor(0., requires_grad=True)
        for name in pred.keys():
            name = name.replace("aux_", "")
            if name in ["seg_mask", "cont_mask"]:
                loss += self.bce(pred[name], gt[name])
            elif name in ["dist_map", "hv_map"]:
                loss += self.mse(pred[name], gt[name])
        return loss


# class PerformanceLoss:
#     def __init__(self, mode: str):
#         self.bce = nn.BCEWithLogitsLoss()
#         self.mse = nn.MSELoss()
#         self.mse_with_logits = MSEWithLogitsLoss()
#         self.mode = mode
#
#     def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Dict[str, Tensor]:
#         if self.mode in ["baseline", "yang", "noname"]:
#             basic = self.bce(pred["seg_mask"], gt["seg_mask"])
#             return {
#                 "basic_performance": basic,
#                 "advanced_performance": basic + self.bce(pred["cont_mask"], gt["cont_mask"])
#             }
#         elif self.mode == "naylor":
#             return {
#                 "basic_performance": self.mse(pred["dist_map"], gt["dist_map"]),
#             }
#         else:
#             return {
#                 "basic_performance": self.mse(pred["seg_mask"], gt["seg_mask"]) +
#                                      self.mse(pred["hv_map"], gt["hv_map"]),
#                 "advanced_performance": self.mse_with_logits(pred["seg_mask"], gt["seg_mask"]) +
#                                         self.mse_with_logits(pred["hv_map"], (gt["hv_map"] + 1) / 2)
#             }


class SegLoss:
    def __init__(self, double_main_task: bool = False, auxiliary_task: bool = True):
        self.double_main_task = double_main_task
        self.auxiliary_task = auxiliary_task
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Tensor:
        if self.double_main_task and self.auxiliary_task:
            return 2 * self.bce(pred["seg_mask"], gt["seg_mask"]) + \
                   self.bce(pred["aux_seg_mask"], gt["seg_mask"]) + \
                   self.bce(pred["cont_mask"], gt["cont_mask"])
        elif self.double_main_task and not self.auxiliary_task:
            return 2 * self.bce(pred["seg_mask"], gt["seg_mask"]) + \
                   self.bce(pred["aux_seg_mask"], gt["seg_mask"])
        elif not self.double_main_task and self.auxiliary_task:
            return self.bce(pred["seg_mask"], gt["seg_mask"]) + \
                   self.bce(pred["cont_mask"], gt["cont_mask"])
        else:
            return self.bce(pred["seg_mask"], gt["seg_mask"])


class DistLoss:
    def __init__(self, double_main_task: bool = False, auxiliary_task: bool = True):
        self.double_main_task = double_main_task
        self.auxiliary_task = auxiliary_task
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Tensor:
        if self.double_main_task and self.auxiliary_task:
            return 2 * self.mse(pred["dist_map"], gt["dist_map"]) + \
                   self.mse(pred["aux_dist_map"], gt["dist_map"]) + \
                   self.bce(pred["cont_mask"], gt["cont_mask"])
        elif self.double_main_task and not self.auxiliary_task:
            return 2 * self.mse(pred["dist_map"], gt["dist_map"]) + \
                   self.mse(pred["aux_dist_map"], gt["dist_map"])
        elif not self.double_main_task and self.auxiliary_task:
            return self.mse(pred["dist_map"], gt["dist_map"]) + \
                   self.bce(pred["cont_mask"], gt["cont_mask"])
        else:
            return self.mse(pred["dist_map"], gt["dist_map"])


class HVLoss:
    def __init__(self, double_main_task: bool = False):
        self.double_main_task = double_main_task
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Tensor:
        if self.double_main_task:
            return 2 * self.bce(pred["seg_mask"], gt["seg_mask"]) + \
                   self.bce(pred["aux_seg_mask"], gt["seg_mask"]) + \
                   self.mse(pred["hv_map"], gt["hv_map"])
        else:
            return self.bce(pred["seg_mask"], gt["seg_mask"]) + \
                   self.mse(pred["hv_map"], gt["hv_map"])


class GrahamLoss:
    """
    Calculates the loss as presented by Graham et al. 2019.

    Please note, that only the loss terms for the nuclear prediction and the HoVer branch are included.
    """

    def __init__(self, additional_loss_term: bool = False):
        self.additional_loss_term = additional_loss_term
        self.mse = nn.MSELoss()
        self.msge = MSGELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dsc = DSCWithLogitsLoss()

    def __call__(self, pred: Dict[str, Tensor], gt: Dict[str, Union[str, Tensor]]) -> Tensor:
        if self.additional_loss_term:
            seg_loss = self.bce(pred["seg_mask"], gt["seg_mask"]) + self.dsc(pred["seg_mask"], gt["seg_mask"]) + \
                       self.bce(pred["aux_seg_mask"], gt["seg_mask"]) + self.dsc(pred["aux_seg_mask"], gt["seg_mask"])
        else:
            seg_loss = self.bce(pred["seg_mask"], gt["seg_mask"]) + self.dsc(pred["seg_mask"], gt["seg_mask"])
        hv_loss = self.mse(pred["hv_map"], gt["hv_map"]) + 2 * self.msge(pred["hv_map"], gt["hv_map"], gt["seg_mask"])

        return hv_loss + seg_loss


class DSCLoss:
    def __init__(self, ignore_background: bool = True, invert: bool = True, smoothness_const: float = 1e-5):
        self.invert = invert
        self.ignore_background = ignore_background
        self.smoothness_const = smoothness_const

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        loss = self.dsc(pred, gt, smoothness_const=self.smoothness_const)
        if self.invert:
            loss = 1 - loss
        return loss

    # def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
    #     numerator = 2. * torch.sum(pred * gt, dim=(0, 2, 3)) + self.epsilon
    #     denominator = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(gt, dim=(0, 2, 3)) + self.epsilon
    #     loss = 1. - numerator / denominator
    #     return torch.sum(loss)

    @staticmethod
    def dsc(pred: Tensor, gt: Tensor, ignore_background: bool = True, smoothness_const: float = 1e-5) -> Tensor:
        """
        Calculates the dice score.

        Tensors are expected to be of shape (N, C, H, W). The dice score is calculated for the whole batch at once.
        Another option would be to take the mean of the individual dice score from each batch element.
        """
        intersection = torch.sum(pred * gt)
        denominator = pred.sum() + gt.sum() + smoothness_const
        if not ignore_background:
            intersection += torch.sum((1 - pred) * (1 - gt))
            denominator += torch.sum(1 - pred) + torch.sum(1 - gt)
        numerator = 2 * intersection + smoothness_const

        return numerator / denominator  # shape (1)


class DSCWithLogitsLoss(DSCLoss):
    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = torch.sigmoid(pred)
        return super().__call__(pred, gt)


class MSEWithLogitsLoss(nn.MSELoss):
    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = torch.sigmoid(pred)
        return super().__call__(pred, gt)


class MSGELoss:
    """
    Calculates the mean squared gradient error (MSGE) of the horizontal and vertical distance map.

    The MSGE loss was developed by Graham et al. 2019.

    Code is adapted from:
    https://github.com/vqdang/hover_net/blob/master/models/hovernet/utils.py
    """

    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size

    @staticmethod
    def make_sobel_kernel(size: int = 5) -> Tuple[Tensor, Tensor]:
        """Creates a sobel kernel of given size."""

        if not size % 2:
            raise ValueError(f"Size should be an odd number. Got instead {size}.")

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="xy")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_hv_grad(self, hv: Tensor) -> Tensor:
        """
        Calculates the gradient of the horizontal and vertical distance map.
        """
        kernel_h, kernel_v = self.make_sobel_kernel(self.kernel_size)
        kernel_h = kernel_h.view(1, 1, 5, 5)
        kernel_v = kernel_v.view(1, 1, 5, 5)

        h = hv[:, 0].unsqueeze(1)  # Shape (N,1,H,W)
        v = hv[:, 1].unsqueeze(1)
        dh = F.conv2d(h, kernel_h, padding=self.kernel_size // 2)
        dv = F.conv2d(v, kernel_v, padding=self.kernel_size // 2)
        return torch.cat((dh, dv), dim=1)

    def __call__(self, pred: Tensor, gt: Tensor, focus: Tensor) -> Tensor:
        focus = torch.cat((focus, focus), dim=1)
        grad_pred = self.get_hv_grad(pred)
        grad_gt = self.get_hv_grad(gt)
        error = grad_pred - grad_gt
        # Squared error in focus region:
        error = focus * (error * error)
        return error.sum() / (focus.sum() + 1.0e-8)
