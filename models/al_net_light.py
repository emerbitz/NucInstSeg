from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from models.al_net import ALBridge, ALDecoderAuxiliary
from models.building_blocks import CELayer, Up, BLUpConvBNReLU, ASPP


class ALLightEncoder(nn.Module):
    """
    Consists of a pretrained ResNet34 with a CE layer before every pooling (except for the first pooling XD).

    The last down-sampling layer is omitted.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.resnet = models.resnet34(weights='IMAGENET1K_V1')
        # self.resnet.requires_grad_(False)
        self.resnet.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resnet.layer1[0].conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.ce_layer = nn.ModuleList([CELayer(channels=2 ** i * 32) for i in range(3)])
        self.resnet.fc = None
        self.resnet.layer4 = None

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        s0 = self.ce_layer[0](x)

        x = self.resnet.maxpool(s0)
        x = self.resnet.layer1(x)
        s1 = self.ce_layer[1](x)

        x = self.resnet.layer2(s1)
        s2 = self.ce_layer[2](x)

        x = self.resnet.layer3(s2)
        return x, s0, s1, s2


class ALLightDecoderMain(nn.Module):
    """
    AL-Net decoder for the main task.
    """

    def __init__(self, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.d3 = BLUpConvBNReLU(in_channels=8 * base_channels, out_channels=4 * base_channels)
        self.d2 = BLUpConvBNReLU(in_channels=8 * base_channels, out_channels=2 * base_channels)
        self.d1 = BLUpConvBNReLU(in_channels=4 * base_channels, out_channels=base_channels)
        self.d0 = Up(in_channels=2 * base_channels, out_channels=out_channels, kernel_size=3,
                     mode="up-convolution")  # Maybe Conv 1x1

    def forward(self, x: Tensor, skip: Tuple[Tensor, ...]) -> Tensor:
        x = self.d3(x)
        x = torch.cat((x, skip[-1]), dim=1)
        x = self.d2(x)
        x = torch.cat((x, skip[-2]), dim=1)
        x = self.d1(x)
        x = torch.cat((x, skip[-3]), dim=1)
        x = self.d0(x)
        return x


class ALNetLight(nn.Module):
    """
    Lightweight version of the Attention Learning Network (AL-Net) introduced by Zhao et al. 2022.

    The depth of the lightweight version is one less than that of the original version.
    """

    def __init__(self, in_channels: int = 3, mode: str = "noname", net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "auxiliary_task": True,  # Dis-/enables optional auxiliary task for modes 'baseline', 'yang' and 'naylor'
            "aspp": False,
            "aspp_inter_channels": 32,
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}

        self.e = ALLightEncoder(in_channels=in_channels)
        self.b = ALBridge(base_channels=32, num_branches=3)
        if net_params["aspp"]:
            self.aspp = ASPP(in_channels=256, inter_channels=net_params["aspp_inter_channels"])
        else:
            self.aspp = None
        self.dm = ALLightDecoderMain(base_channels=32, out_channels=1)
        if mode in ["graham", "exprmtl"]:
            self.da = ALDecoderAuxiliary(out_channels=2, num_branches=3)
        elif not net_params["auxiliary_task"]:
            self.da = None
        else:
            self.da = ALDecoderAuxiliary(out_channels=1, num_branches=3)

        self.mode = mode  # Fetched by NetModel: Selects loss term and postprocessing scheme
        self.auxiliary_task = net_params["auxiliary_task"]  # Fetched by NetModel: Adapts loss term
        if not self.auxiliary_task and mode not in ["baseline", "yang", "naylor"]:
            raise ValueError(f"No auxiliary task is only support for modes 'baseline', 'yang', 'naylor'. "
                             f"Got instead {self.mode}.")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, *skip = self.e(x)
        skip = self.b(skip)
        if self.aspp:
            x = self.aspp(x)
        x_main = self.dm(x, skip)
        if self.auxiliary_task:
            x_aux = self.da(skip)
            if self.mode in ["baseline", "noname", "yang"]:
                return {"seg_mask": x_main, "cont_mask": x_aux}
            elif self.mode == "naylor":
                return {"dist_map": x_main, "cont_mask": x_aux}
            elif self.mode in ["graham", "exprmtl"]:
                return {"seg_mask": x_main, "hv_map": x_aux}
        else:
            if self.mode in ["baseline", "yang"]:
                return {"seg_mask": x_main}
            elif self.mode == "naylor":
                return {"dist_map": x_main}


class ALNetLightDualDecoder(nn.Module):
    """
    Lightweight version of the Attention Learning Network (AL-Net) with a dual decoder.

    The depth of the lightweight version is one less than that of the original AL-Net.
    """

    def __init__(self, in_channels: int = 3, mode: str = "noname", net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "auxiliary_task": True,
            # Dis-/enables optional auxiliary task for modes 'baseline', 'yang' and 'naylor'
            "aspp": False,
            "aspp_inter_channels": 32,
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}

        self.e = ALLightEncoder(in_channels=in_channels)
        self.b = ALBridge(base_channels=32, num_branches=3)
        if net_params["aspp"]:
            self.aspp = ASPP(in_channels=256, inter_channels=net_params["aspp_inter_channels"])
        else:
            self.aspp = None
        self.dm = ALLightDecoderMain(base_channels=32, out_channels=1)
        if mode in ["graham", "exprmtl"]:
            self.da = ALLightDecoderMain(base_channels=32, out_channels=2)
        elif not net_params["auxiliary_task"]:
            self.da = None
        else:
            self.da = ALLightDecoderMain(base_channels=32, out_channels=1)

        self.mode = mode  # Fetched by NetModel: Selects loss term and postprocessing scheme
        self.auxiliary_task = net_params["auxiliary_task"]  # Fetched by NetModel: Adapts loss term
        if not self.auxiliary_task and mode not in ["baseline", "yang", "naylor"]:
            raise ValueError(f"No auxiliary task is only support for modes 'baseline', 'yang', 'naylor'. "
                             f"Got instead {self.mode}.")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, *skip = self.e(x)
        skip = self.b(skip)
        if self.aspp:
            x = self.aspp(x)
        x_main = self.dm(x, skip)
        if self.auxiliary_task:
            x_aux = self.da(x, skip)
            if self.mode in ["baseline", "noname", "yang"]:
                return {"seg_mask": x_main, "cont_mask": x_aux}
            elif self.mode == "naylor":
                return {"dist_map": x_main, "cont_mask": x_aux}
            elif self.mode in ["graham", "exprmtl"]:
                return {"seg_mask": x_main, "hv_map": x_aux}
        else:
            if self.mode in ["baseline", "yang"]:
                return {"seg_mask": x_main}
            elif self.mode == "naylor":
                return {"dist_map": x_main}
