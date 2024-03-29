from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from models.building_blocks import ALModule, CELayer, VolUp, Up, BLUpConvBNReLU


class ALEncoder(nn.Module):
    """Consists of a pretrained ResNet34 with a CE layer before every pooling (except for the first pooling XD)"""

    def __init__(self, in_channels: int, freeze_resnet: bool = False):
        super().__init__()
        self.resnet = models.resnet34(weights='IMAGENET1K_V1')
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resnet.layer1[0].conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.ce_layer = nn.ModuleList([CELayer(channels=2 ** i * 32) for i in range(4)])
        self.resnet.fc = None

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
        s3 = self.ce_layer[3](x)

        x = self.resnet.layer4(s3)
        return x, s0, s1, s2, s3


class ALBridge(nn.ModuleList):
    """Applies the Attention Learning (AL) module to the skip connections"""

    def __init__(self, base_channels: int = 32, num_branches: int = 4):
        super().__init__([ALModule(in_channels=2 ** i * base_channels) for i in range(num_branches)])

    def forward(self, skip: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return tuple([m(s) for s, m in zip(skip, self)])


class ALDecoderMain(nn.Module):
    """
    AL-Net decoder for the main task.
    """

    def __init__(self, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.d4 = VolUp(channel_factor=0.5, spatial_factor=2)
        # self.d4 = BLUpConvBNReLU(in_channels=16*base_channels, out_channels=8*base_channels)
        self.d3 = BLUpConvBNReLU(in_channels=16 * base_channels, out_channels=4 * base_channels)
        self.d2 = BLUpConvBNReLU(in_channels=8 * base_channels, out_channels=2 * base_channels)
        self.d1 = BLUpConvBNReLU(in_channels=4 * base_channels, out_channels=base_channels)
        self.d0 = Up(in_channels=2 * base_channels, out_channels=out_channels, kernel_size=3,
                     mode="up-convolution")  # Maybe Conv 1x1

    def forward(self, x: Tensor, skip: Tuple[Tensor, ...]) -> Tensor:
        x = self.d4(x)
        x = torch.cat((x, skip[-1]), dim=1)
        x = self.d3(x)
        x = torch.cat((x, skip[-2]), dim=1)
        x = self.d2(x)
        x = torch.cat((x, skip[-3]), dim=1)
        x = self.d1(x)
        x = torch.cat((x, skip[-4]), dim=1)
        x = self.d0(x)
        return x


class ALDecoderAuxiliary(nn.Module):
    def __init__(self, out_channels: int, base_channels: int = 32, num_branches: int = 4):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(1, num_branches + 1):
            branch = i * nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2))
            branch.append(
                nn.Conv2d(in_channels=2 ** (i - 1) * base_channels, out_channels=out_channels, kernel_size=1,
                          bias=1 == i)
            )
            self.decoder.append(branch)

    def forward(self, skip: Tuple[Tensor, ...]) -> Tensor:
        return sum([b(s) for s, b in zip(skip, self.decoder)])


class ALNet(nn.Module):
    """Attention Learning Network (AL-Net) as introduced by Zhao et al. 2022"""

    def __init__(self, in_channels: int = 3, mode: str = "contour", net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "auxiliary_task": True,  # Dis-/enables optional auxiliary task for modes 'baseline', 'yang' and 'naylor'
            "naylor_aux_task": "cont_mask",  # Auxiliary task for Naylor postprocessing
            "freeze_resnet": False,
            "bridge": True
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}

        self.e = ALEncoder(in_channels=in_channels, freeze_resnet=net_params["freeze_resnet"])
        if net_params["bridge"]:
            self.b = ALBridge(base_channels=32)
        else:
            self.b = None
        self.dm = ALDecoderMain(base_channels=32, out_channels=1)
        if mode in ["graham", "exprmtl"]:
            self.da = ALDecoderAuxiliary(out_channels=2)
        elif not net_params["auxiliary_task"]:
            self.da = None
        elif mode == "naylor" and net_params["auxiliary_task"] and net_params["naylor_aux_task"] == "hv_map":
            self.da = ALDecoderAuxiliary(out_channels=2)
        else:
            self.da = ALDecoderAuxiliary(out_channels=1)

        self.mode = mode  # Fetched by NetModel: Selects loss term and postprocessing scheme
        self.auxiliary_task = net_params["auxiliary_task"]  # Fetched by NetModel: Adapts loss term
        self.naylor_aux_task = net_params["naylor_aux_task"]  # Fetched by NetModel: Adapts loss term
        if not self.auxiliary_task and mode not in ["baseline", "yang", "naylor"]:
            raise ValueError(f"No auxiliary task is only support for modes 'baseline', 'yang', 'naylor'. "
                             f"Got instead {self.mode}.")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, *skip = self.e(x)
        if self.b is not None:
            skip = self.b(skip)
        x_main = self.dm(x, skip)
        if self.auxiliary_task:
            x_aux = self.da(skip)
            if self.mode in ["baseline", "contour", "yang"]:
                return {"seg_mask": x_main, "cont_mask": x_aux}
            elif self.mode == "naylor":
                if self.naylor_aux_task == "cont_mask":
                    return {"dist_map": x_main, "cont_mask": x_aux}
                else:
                    return {"dist_map": x_main, "hv_map": x_aux}
            elif self.mode in ["graham", "exprmtl"]:
                return {"seg_mask": x_main, "hv_map": x_aux}
        else:
            if self.mode in ["baseline", "yang"]:
                return {"seg_mask": x_main}
            elif self.mode == "naylor":
                return {"dist_map": x_main}
