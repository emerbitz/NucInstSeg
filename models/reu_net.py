from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.att_net import AttentionUNet


class REUNet(nn.Module):
    """
    REU-Net from Qin et al. 2022
    """

    def __init__(self, in_channels: int = 3, mode: str = "noname", net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "inter_channels_equal_out_channels": True,
            "min_inter_channels": 32,
            "aspp": True,
            "aspp_inter_channels": 16,
            "norm_actv": False,
            "base_channels": 32,
            "base_channels_factor": 1
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}

        self.norm_actv = net_params["norm_actv"]

        # Main branch
        self.mb = AttentionUNet(in_channels=in_channels, out_channels=1, net_params=net_params)

        if mode in ["baseline", "noname", "yang"]:
            # Main branch - normalization and activation
            if self.norm_actv:
                self.mb_actv_norm = nn.Sequential(
                    nn.BatchNorm2d(num_features=1),
                    nn.Sigmoid()
                )
            # Auxiliary branch
            self.ab = AttentionUNet(in_channels=in_channels, out_channels=1, net_params=net_params)

            # Auxiliary branch - normalization and activation
            if self.norm_actv:
                self.ab_actv_norm = nn.Sequential(
                    nn.BatchNorm2d(num_features=1),
                    nn.Sigmoid()
                )
            # Compression
            self.comp = None
            # Final branch
            net_params["num_ext_skip"] = 2
            net_params["base_channels_ext"] = net_params["base_channels"]
            net_params["base_channels"] *= net_params["base_channels_factor"]
            net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
            self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)

        elif mode == "naylor":
            # Main branch - normalization and activation
            if self.norm_actv:
                self.mb_actv_norm = nn.Sequential(
                    nn.BatchNorm2d(num_features=1),
                    nn.ReLU()
                )
            # Auxiliary branch
            self.ab = None
            # Final branch
            net_params["num_ext_skip"] = 1
            net_params["base_channels_ext"] = net_params["base_channels"]
            net_params["base_channels"] *= net_params["base_channels_factor"]
            net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
            self.fb = AttentionUNet(in_channels=2 * in_channels, out_channels=1, net_params=net_params)
        elif mode in ["graham", "exprmtl"]:
            # Main branch - normalization and activation
            if self.norm_actv:
                self.mb_actv_norm = nn.Sequential(
                    nn.BatchNorm2d(num_features=1),
                    nn.ReLU()
                )
            # Auxiliary branch
            self.ab = AttentionUNet(in_channels=in_channels, out_channels=2, net_params=net_params)
            # Auxiliary branch - normalization and activation
            if self.norm_actv:
                self.ab_actv_norm = nn.Sequential(
                    nn.BatchNorm2d(num_features=2),
                    nn.Tanh()
                )
            # Compression
            self.comp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
            # Final branch
            net_params["num_ext_skip"] = 2
            net_params["base_channels_ext"] = net_params["base_channels"]
            net_params["base_channels"] *= net_params["base_channels_factor"]
            net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
            self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)
        self.mode = mode  # Fetched by NetModel: Selects loss term and postprocessing scheme
        self.auxiliary_task = mode != "naylor"  # Fetched by NetModel: Adapts loss term
        if not self.auxiliary_task and mode not in ["baseline", "yang", "naylor"]:
            raise ValueError(f"No auxiliary task is only support for modes 'baseline', 'yang', 'naylor'. "
                             f"Got instead {self.mode}.")
        self.double_main_task = True  # Fetched by NetModel: Adapts loss term

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_main, skip_main = self.mb(x)
        x_main_actv_norm = self.mb_actv_norm(x_main) if self.norm_actv else x_main
        if self.ab is not None:
            x_aux, skip_aux = self.ab(x)
            x_aux_actv_norm = self.ab_actv_norm(x_aux) if self.norm_actv else x_aux
            if self.comp is not None:
                x = torch.cat((x, x_main_actv_norm * x, self.comp(x_aux_actv_norm) * x), dim=1)
            else:
                x = torch.cat((x, x_main_actv_norm * x, x_aux_actv_norm * x), dim=1)
            skip = (skip_main, skip_aux)
        else:
            x = torch.cat((x, x_main_actv_norm * x), dim=1)
            skip = [skip_main]
        x_final, _ = self.fb(x, skip)

        if self.mode in ["baseline", "noname", "yang"]:
            return {"seg_mask": x_final, "cont_mask": x_aux, "aux_seg_mask": x_main}
        elif self.mode == "naylor":
            return {"dist_map": x_final, "aux_dist_map": x_main}
        elif self.mode in ["graham", "exprmtl"]:
            return {"seg_mask": x_final, "hv_map": x_aux, "aux_seg_mask": x_main}

#
# class REUNet(nn.Module):
#
#     def __init__(self, in_channels: int = 3, mode: str = "noname", net_params: Optional[dict] = None):
#         super().__init__()
#         # Define default hyperparameters:
#         default_params = {
#             "inter_channels_equal_out_channels": True,
#             "min_inter_channels": 32,
#             "aspp": True,
#             "aspp_inter_channels": 1,
#             "norm_actv": False
#         }
#         if net_params is None:
#             net_params = {}
#         # Overwrite default hyperparameters:
#         net_params = {**default_params, **net_params}
#
#         self.norm_actv = net_params["norm_actv"]
#
#         # Main branch
#         self.mb = AttentionUNet(in_channels=in_channels, out_channels=1, aspp=True,
#                                 inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                 aspp_inter_channels=net_params["aspp_inter_channels"],
#                                 min_inter_channels=net_params["min_inter_channels"])
#
#         if mode in ["baseline", "noname", "yang"]:
#             # Main branch - normalization and activation
#             if self.norm_actv:
#                 self.mb_actv_norm = nn.Sequential(
#                     nn.BatchNorm2d(num_features=1),
#                     nn.Sigmoid()
#                 )
#             # Auxiliary branch
#             self.ab = AttentionUNet(in_channels=in_channels, out_channels=1, aspp=True,
#                                     inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                     aspp_inter_channels=net_params["aspp_inter_channels"],
#                                     min_inter_channels=net_params["min_inter_channels"])
#
#             # Auxiliary branch - normalization and activation
#             if self.norm_actv:
#                 self.ab_actv_norm = nn.Sequential(
#                     nn.BatchNorm2d(num_features=1),
#                     nn.Sigmoid()
#                 )
#             # Compression
#             self.comp = None
#             # Final branch
#             self.fb = AttentionUNet(in_channels=3*in_channels, out_channels=1, aspp=True, num_ext_skip=2,
#                                     inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                     aspp_inter_channels=net_params["aspp_inter_channels"],
#                                     min_inter_channels=net_params["min_inter_channels"])
#         elif mode == "naylor":
#             # Main branch - normalization and activation
#             if self.norm_actv:
#                 self.mb_actv_norm = nn.Sequential(
#                     nn.BatchNorm2d(num_features=1),
#                     nn.ReLU()
#                 )
#             # Auxiliary branch
#             self.ab = None
#             # Final branch
#             self.fb = AttentionUNet(in_channels=2*in_channels, out_channels=1, aspp=True, num_ext_skip=1,
#                                     inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                     aspp_inter_channels=net_params["aspp_inter_channels"],
#                                     min_inter_channels=net_params["min_inter_channels"])
#         elif mode in ["graham", "exprmtl"]:
#             # Main branch - normalization and activation
#             if self.norm_actv:
#                 self.mb_actv_norm = nn.Sequential(
#                     nn.BatchNorm2d(num_features=1),
#                     nn.ReLU()
#                 )
#             # Auxiliary branch
#             self.ab = AttentionUNet(in_channels=in_channels, out_channels=2, aspp=True,
#                                     inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                     aspp_inter_channels=["aspp_inter_channels"],
#                                     min_inter_channels=["min_inter_channels"])
#             # Auxiliary branch - normalization and activation
#             if self.norm_actv:
#                 self.ab_actv_norm = nn.Sequential(
#                     nn.BatchNorm2d(num_features=2),
#                     nn.Tanh()
#                 )
#             # Compression
#             self.comp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
#             # Final branch
#             self.fb = AttentionUNet(in_channels=3*in_channels, out_channels=1, aspp=True, num_ext_skip=2,
#                                     inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
#                                     aspp_inter_channels=net_params["aspp_inter_channels"],
#                                     min_inter_channels=["min_inter_channels"])
#         self.mode = mode   # Fetched by NetModel: Selects loss term and postprocessing scheme
#         self.additional_loss_term = True  # Fetched by NetModel: Adapts loss term
#
#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         x_main, skip_main = self.mb(x)
#         if self.norm_actv:
#             x_main = self.mb_actv_norm(x_main)
#         if self.ab is not None:
#             x_aux, skip_aux = self.ab(x)
#             if self.norm_actv:
#                 x_aux = self.ab_actv_norm(x_aux)
#             if self.comp is not None:
#                 x = torch.cat((x, x_main * x, self.comp(x_aux) * x), dim=1)
#             else:
#                 x = torch.cat((x, x_main * x, x_aux * x), dim=1)
#             skip = (skip_main, skip_aux)
#         else:
#             x = torch.cat((x, x_main * x), dim=1)
#             skip = [skip_main]
#         x_final, _ = self.fb(x, skip)
#
#         if self.mode in ["baseline", "noname", "yang"]:
#             return {"seg_mask": x_final, "cont_mask": x_aux, "aux_seg_mask": x_main}
#         elif self.mode == "naylor":
#             return {"dist_map": x_final, "aux_dist_map": x_main}
#         elif self.mode in ["graham", "exprmtl"]:
#             return {"seg_mask": x_final, "hv_map": x_aux, "aux_seg_mask": x_main}
