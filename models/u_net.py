from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from models.building_blocks import ConvBlock, EncodBlock, DecodBlock


class OrigUNet(nn.Module):
    """
    Original U-Net as described by Ronneberger et al. 2015.

    Modifications:
    * BatchNorm instead of Dropout
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.e1 = EncodBlock(in_channels=in_channels, out_channels=base_channels)
        self.e2 = EncodBlock(in_channels=base_channels, out_channels=2 * base_channels)
        self.e3 = EncodBlock(in_channels=2 * base_channels, out_channels=4 * base_channels)
        self.e4 = EncodBlock(in_channels=4 * base_channels, out_channels=8 * base_channels)
        # Bottleneck
        self.b = ConvBlock(in_channels=8 * base_channels, out_channels=16 * base_channels)
        # Decoder
        self.d1 = DecodBlock(in_channels=16 * base_channels, out_channels=8 * base_channels, up_mode="up-convolution")
        self.d2 = DecodBlock(in_channels=8 * base_channels, out_channels=4 * base_channels, up_mode="up-convolution")
        self.d3 = DecodBlock(in_channels=4 * base_channels, out_channels=2 * base_channels, up_mode="up-convolution")
        self.d4 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels, up_mode="up-convolution")
        # Classifier
        self.c = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        # Bottleneck
        x = self.b(x)
        # Decoder
        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)
        # Classifier
        return self.c(x)


class UNet(nn.Module):
    """
    U-Net with three down/up-sampling blocks.

    Additional layer after the final U-Net layer:
    * Connector layer: 3x3 Conv, BN, ReLU
    * Output layer(s): 1x1 Conv
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, con_channels: int = 128, mode: str = "contour"):
        super().__init__()
        # Encoder
        self.e0 = EncodBlock(in_channels=in_channels, out_channels=base_channels)
        self.e1 = EncodBlock(in_channels=base_channels, out_channels=2 * base_channels)
        self.e2 = EncodBlock(in_channels=2 * base_channels, out_channels=4 * base_channels)
        # Bottleneck
        self.b = ConvBlock(in_channels=4 * base_channels, out_channels=4 * base_channels)
        # Decoder
        self.d0 = DecodBlock(in_channels=4 * base_channels, out_channels=2 * base_channels)
        self.d1 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels)
        self.d2 = DecodBlock(in_channels=base_channels, out_channels=base_channels)
        # Connector
        self.c = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=con_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=con_channels),
            nn.ReLU()
        )
        # Output
        if mode == "contour":
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.auxiliary_task = True
        elif mode in ["baseline", "yang", "naylor"]:
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.auxiliary_task = False
        elif mode in ["graham", "exprmtl"]:
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=con_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.mode = mode

    def forward(self, x) -> Dict[str, Tensor]:
        # Encoder
        x, s0 = self.e0(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        # Bottleneck
        x = self.b(x)
        # Decoder
        x = self.d0(x, s2)
        x = self.d1(x, s1)
        x = self.d2(x, s0)
        # Connector
        x = self.c(x)
        # Output
        if self.mode == "contour":
            return {"seg_mask": self.o0(x), "cont_mask": self.o1(x)}
        elif self.mode in ["baseline", "yang"]:
            return {"seg_mask": self.o0(x)}
        elif self.mode == "naylor":
            return {"dist_map": self.o0(x)}
        elif self.mode in ["graham", "exprmtl"]:
            return {"seg_mask": self.o0(x), "hv_map": self.o1(x)}


class UNetDualDecoder(nn.Module):
    """
    U-Net with one encoder and two decoder.

    The encoder and the decoder consist of three down-sampling and up-sampling blocks, respectively.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 con_channels: Optional[int] = 128, mode: str = "contour"):
        super().__init__()
        # Encoder
        self.e0 = EncodBlock(in_channels=in_channels, out_channels=base_channels, skip_con=True)
        self.e1 = EncodBlock(in_channels=base_channels, out_channels=2 * base_channels, skip_con=True)
        self.e2 = EncodBlock(in_channels=2 * base_channels, out_channels=4 * base_channels, skip_con=True)
        # Bottleneck
        self.b = ConvBlock(in_channels=4 * base_channels, out_channels=4 * base_channels)
        # Decoder a
        self.da0 = DecodBlock(in_channels=4 * base_channels, out_channels=2 * base_channels, skip_con=True)
        self.da1 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels, skip_con=True)
        self.da2 = DecodBlock(in_channels=base_channels, out_channels=base_channels, skip_con=True)
        # Decoder b
        self.db0 = DecodBlock(in_channels=4 * base_channels, out_channels=2 * base_channels, skip_con=True)
        self.db1 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels, skip_con=True)
        self.db2 = DecodBlock(in_channels=base_channels, out_channels=base_channels, skip_con=True)
        if con_channels is not None:
            # Connector a
            self.ca = nn.Sequential(
                nn.Conv2d(in_channels=base_channels, out_channels=con_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=con_channels),
                nn.ReLU()
            )
            # Connector b
            self.cb = nn.Sequential(
                nn.Conv2d(in_channels=base_channels, out_channels=con_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=con_channels),
                nn.ReLU()
            )
        else:
            # Connector a
            self.ca = None
            # Connector b
            self.cb = None
        # Output
        output_in_channels = base_channels if con_channels is None else con_channels
        self.oa = nn.Conv2d(in_channels=output_in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        if mode == "contour":
            self.ob = nn.Conv2d(in_channels=output_in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode in ["graham", "exprmtl"]:
            self.ob = nn.Conv2d(in_channels=output_in_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.mode = mode

    def forward(self, x) -> Dict[str, Tensor]:
        # Encoder
        x, s0 = self.e0(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        # Bottleneck
        x = self.b(x)
        # Decoder a
        xa = self.da0(x, s2)
        xa = self.da1(xa, s1)
        xa = self.da2(xa, s0)
        # Decoder b
        xb = self.db0(x, s2)
        xb = self.db1(xb, s1)
        xb = self.db2(xb, s0)
        # Connector a
        if self.ca is not None:
            xa = self.ca(xa)
        # Connector b
        if self.cb is not None:
            xb = self.cb(xb)
        # Output
        if self.mode == "contour":
            return {"seg_mask": self.oa(xa), "cont_mask": self.ob(xb)}
        elif self.mode in ["graham", "exprmtl"]:
            return {"seg_mask": self.oa(xa), "hv_map": self.ob(xb)}
