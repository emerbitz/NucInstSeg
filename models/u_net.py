from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=stride * (kernel_size // 2))
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=stride * (kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, mode: str = "bilinear"):
        super().__init__()
        if mode == "bilinear":
            self.up = nn.UpsamplingBilinear2d(scale_factor=(2., 2.))
        elif mode == "transposed":
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)
        elif mode == "up-convolution":
            # The original U-Net (Ronneberger el al. 2015) used upsampling followed by a convolution.
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=(2., 2.)),
                # Here 3x3 conv instead of 2x2 conv as in the original U-Net
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=1, padding=1)
            )
        else:
            raise ValueError(f"Mode should be bilinear, transposed or up-convolution. Got instead {mode}.")

    def forward(self, x):
        return self.up(x)


class EncodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv_block(x)
        return self.pool(x), x


class DecodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, up_mode: str = "bilinear"):
        super().__init__()
        self.up_conv = Up(in_channels=in_channels, mode=up_mode)
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


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

    def __init__(self, in_channels: int = 3, base_channels: int = 32, con_channels: int = 128, mode: str = "seg"):
        super().__init__()
        # Encoder
        self.e0 = EncodBlock(in_channels=in_channels, out_channels=base_channels)
        self.e1 = EncodBlock(in_channels=base_channels, out_channels=2 * base_channels)
        self.e2 = EncodBlock(in_channels=2 * base_channels, out_channels=4 * base_channels)
        # Bottleneck
        self.b = ConvBlock(in_channels=4 * base_channels, out_channels=4 * base_channels)
        # Decoder
        self.d0 = DecodBlock(in_channels=8 * base_channels, out_channels=2 * base_channels)
        self.d1 = DecodBlock(in_channels=4 * base_channels, out_channels=base_channels)
        self.d2 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels)
        # Connector
        self.c = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=con_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=con_channels),
            nn.ReLU()
        )
        # Output
        if mode == "seg":
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "dist":
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "hv":
            self.o0 = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=con_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError(f"Mode should be seg, dist or hv. Got instead {mode}.")
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
        if self.mode == "seg":
            return {"seg_mask": self.o0(x), "cont_mask": self.o1(x)}
        elif self.mode == "dist":
            return {"dist_map": self.o0(x)}
        elif self.mode == "hv":
            return {"seg_mask": self.o0(x), "hv_map": self.o1(x)}


class UNetDualDecoder(nn.Module):
    """
    U-Net with one encoder and two decoder.

    The encoder and the decoder consist of three down-sampling and up-sampling blocks, respectively.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, con_channels: int = 128, mode: str = "seg"):
        super().__init__()
        # Encoder
        self.e0 = EncodBlock(in_channels=in_channels, out_channels=base_channels)
        self.e1 = EncodBlock(in_channels=base_channels, out_channels=2 * base_channels)
        self.e2 = EncodBlock(in_channels=2 * base_channels, out_channels=4 * base_channels)
        # Bottleneck
        self.b = ConvBlock(in_channels=4 * base_channels, out_channels=4 * base_channels)
        # Decoder a
        self.da0 = DecodBlock(in_channels=8 * base_channels, out_channels=2 * base_channels)
        self.da1 = DecodBlock(in_channels=4 * base_channels, out_channels=base_channels)
        self.da2 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels)
        # Decoder b
        self.db0 = DecodBlock(in_channels=8 * base_channels, out_channels=2 * base_channels)
        self.db1 = DecodBlock(in_channels=4 * base_channels, out_channels=base_channels)
        self.db2 = DecodBlock(in_channels=2 * base_channels, out_channels=base_channels)
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
        # Output
        self.oa = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        if mode == "seg":
            self.ob = nn.Conv2d(in_channels=con_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "hv":
            self.ob = nn.Conv2d(in_channels=con_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError(f"Mode should be seg or hv. Got instead {mode}.")
        self.mode = mode

    def forward(self, x) -> Tuple[Tensor, Tensor]:
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
        xa = self.ca(xa)
        # Connector b
        xb = self.cb(xb)
        # Output
        return self.oa(xa), self.ob(xb)
