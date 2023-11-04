from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.att_net import AttentionUNet


class REUNet(nn.Module):
    """
    REU-Net from Qin et al. 2022
    """

    def __init__(self, in_channels: int = 3, mode: str = "contour", naylor_aux_task: Optional[str] = None,
                 net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "inter_channels_equal_out_channels": True,
            "min_inter_channels": 32,
            "aspp": True,
            "aspp_inter_channels": 256,
            "aspp_global_avg": False,
            "norm": False,
            "base_channels": 32,
            "base_channels_factor": 2,
            "depth": 4,
            "img_size": 256
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}

        self.norm = net_params["norm"]

        # Main branch
        self.mb = AttentionUNet(in_channels=in_channels, out_channels=1, net_params=net_params)

        if mode in ["baseline", "contour", "yang"]:
            self.auxiliary_task = True  # Fetched by NetModel: Adapts loss term
            # Auxiliary branch
            self.ab = AttentionUNet(in_channels=in_channels, out_channels=1, net_params=net_params)

            # Compression
            self.comp = None
            # Final branch
            net_params["num_ext_skip"] = 2
            net_params["base_channels_ext"] = net_params["base_channels"]
            net_params["base_channels"] *= net_params["base_channels_factor"]
            net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
            self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)

        elif mode == "naylor":
            if naylor_aux_task is None:
                self.auxiliary_task = False  # Fetched by NetModel: Adapts loss term
                # Auxiliary branch
                self.ab = None
                # Final branch
                net_params["num_ext_skip"] = 1
                net_params["base_channels_ext"] = net_params["base_channels"]
                net_params["base_channels"] *= net_params["base_channels_factor"]
                net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
                self.fb = AttentionUNet(in_channels=2 * in_channels, out_channels=1, net_params=net_params)
            elif naylor_aux_task == "cont_mask":
                self.auxiliary_task = True  # Fetched by NetModel: Adapts loss term
                # Auxiliary branch
                self.ab = AttentionUNet(in_channels=in_channels, out_channels=1, net_params=net_params)

                # Compression
                self.comp = None
                # Final branch
                net_params["num_ext_skip"] = 2
                net_params["base_channels_ext"] = net_params["base_channels"]
                net_params["base_channels"] *= net_params["base_channels_factor"]
                net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
                self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)
            elif naylor_aux_task == "hv_map":
                self.auxiliary_task = True  # Fetched by NetModel: Adapts loss term
                # Auxiliary branch
                self.ab = AttentionUNet(in_channels=in_channels, out_channels=2, net_params=net_params)

                # Compression
                self.comp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
                # Final branch
                net_params["num_ext_skip"] = 2
                net_params["base_channels_ext"] = net_params["base_channels"]
                net_params["base_channels"] *= net_params["base_channels_factor"]
                net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
                self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)
            else:
                raise ValueError(f"Naylor auxiliary task should be None, 'cont_mask' or 'hv_map'. "
                                 f"Got instead {naylor_aux_task}.")
            self.naylor_aux_task = naylor_aux_task  # Fetched by NetModel: Adapts loss term

        elif mode in ["graham", "exprmtl"]:
            self.auxiliary_task = True  # Fetched by NetModel: Adapts loss term
            # Auxiliary branch
            self.ab = AttentionUNet(in_channels=in_channels, out_channels=2, net_params=net_params)
            # Compression
            self.comp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
            # Final branch
            net_params["num_ext_skip"] = 2
            net_params["base_channels_ext"] = net_params["base_channels"]
            net_params["base_channels"] *= net_params["base_channels_factor"]
            net_params["aspp_inter_channels"] *= net_params["base_channels_factor"]
            self.fb = AttentionUNet(in_channels=3 * in_channels, out_channels=1, net_params=net_params)
        self.mode = mode  # Fetched by NetModel: Selects loss term and postprocessing scheme
        self.double_main_task = True  # Fetched by NetModel: Adapts loss term

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_main, skip_main = self.mb(x)
        if self.norm and self.mode == "naylor":
            x_main_norm = x_main / x_main.max()
        else:
            x_main_norm = x_main

        if self.ab is not None:
            x_aux, skip_aux = self.ab(x)
            if self.norm and x_aux.size(1) == 2:
                x_aux_norm = x_aux.abs()
            else:
                x_aux_norm = x_aux
            # x_aux_actv_norm = self.ab_actv_norm(x_aux) if self.norm_actv else x_aux
            if self.comp is not None:
                x = torch.cat((x, x_main_norm * x, self.comp(x_aux_norm) * x), dim=1)
            else:
                x = torch.cat((x, x_main_norm * x, x_aux_norm * x), dim=1)
            skip = (skip_main, skip_aux)
        else:
            x = torch.cat((x, x_main_norm * x), dim=1)
            skip = [skip_main]
        x_final, _ = self.fb(x, skip)

        if self.mode in ["baseline", "contour", "yang"]:
            return {"seg_mask": x_final, "cont_mask": x_aux, "aux_seg_mask": x_main}
        elif self.mode == "naylor":
            if self.naylor_aux_task is None:
                return {"dist_map": x_final, "aux_dist_map": x_main}
            elif self.naylor_aux_task == "cont_mask":
                return {"dist_map": x_final, "aux_dist_map": x_main, "cont_mask": x_aux}
            elif self.naylor_aux_task == "hv_map":
                return {"dist_map": x_final, "aux_dist_map": x_main, "hv_map": x_aux}
        elif self.mode in ["graham", "exprmtl"]:
            return {"seg_mask": x_final, "hv_map": x_aux, "aux_seg_mask": x_main}
