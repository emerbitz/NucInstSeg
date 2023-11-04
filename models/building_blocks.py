from typing import Dict, Literal, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1,
                 bias: bool = True, padding: Union[str, int] = "same"):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      stride=stride, bias=bias, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )


class ConvBNReLUMaxPool(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pooling_factor: int = 2):
        super().__init__(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=pooling_factor, stride=pooling_factor)
        )


class BLUpConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, up_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=up_factor),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: Optional[int] = None, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, bias: bool = True, repeats: int = 2, last_bn_relu: bool = True):
        if inter_channels is None:
            inter_channels = out_channels

        block = []
        for i in range(repeats):
            in_c = in_channels if i == 0 else inter_channels
            out_c = out_channels if (i + 1) == repeats else inter_channels
            bn_relu = last_bn_relu if (i + 1) == repeats else True
            if bn_relu:
                conv = ConvBNReLU(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride,
                                  dilation=dilation, bias=bias)
            else:
                conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride,
                                 dilation=dilation, bias=bias, padding="same")
            block.append(conv)
        super().__init__(*block)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, scale_factor: int = 2,
                 kernel_size: int = 2, mode: str = "bilinear"):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        if mode == "bilinear":
            self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        elif mode == "transposed":
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=scale_factor,
                                         stride=scale_factor)
        elif mode == "up-convolution":
            # The original U-Net (Ronneberger el al. 2015) used upsampling followed by a 2x2 convolution.
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
            )
        else:
            raise ValueError(f"Mode should be 'bilinear', 'transposed' or 'up-convolution'. Got instead {mode}.")

    def forward(self, x):
        return self.up(x)


class VolUp(nn.Module):
    """
    Volumetric upsampling to adapt both the channel number and the size.
    """

    def __init__(self, channel_factor: Union[int, float] = 0.5, spatial_factor: Union[int, float] = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(channel_factor, spatial_factor, spatial_factor), mode="trilinear")

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=0)  # Add an additional dimension
        x = self.up(x)
        return x.squeeze(dim=0)  # Remove the additional dimension


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, bias: bool = True,
                 mode: Literal["max", "avg", "conv", "conv1x1"] = "conv1x1"):
        super().__init__()
        assert mode in ["max", "avg", "conv", "conv1x1"], f"Mode should be max, avg or conv. Got instead {mode}."
        self.down = nn.ModuleDict({
            "max": nn.Sequential(
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding="same")
            ),
            "avg": nn.Sequential(
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding="same")
            ),
            "conv": nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=scale_factor,
                              stride=scale_factor, bias=bias, padding=0),
            "conv1x1": nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=scale_factor,
                                 bias=bias, padding=0)
        })[mode]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class AG(nn.Module):
    """
    (Additive) Attention Gate (AG) for 2D images.

    Adapted from "Attention U-Net: Learning where to look for the pancreas" by Oktay et al. 2018.
    """

    def __init__(self, in_channels: int, gate_channels: Optional[int] = None, inter_channels: Optional[int] = None,
                 actv_att: nn.Module = nn.Sigmoid(), scale_factor: int = 2, up_mode: str = "bilinear",
                 down_mode: Literal["max", "avg", "conv", "conv1x1"] = "conv1x1"):
        super().__init__()
        if gate_channels is None:
            gate_channels = in_channels

        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        assert not scale_factor % 2, f"Scale factor should be even. Got instead {scale_factor}."
        self.down_in = Down(in_channels=in_channels, out_channels=inter_channels, scale_factor=scale_factor,
                            mode=down_mode)
        self.conv_gate = nn.Conv2d(in_channels=gate_channels, out_channels=inter_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.conv_att = nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=1)
        self.actv_att = actv_att
        self.resampler = nn.Upsample(scale_factor=scale_factor, mode=up_mode)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        a = self.relu(self.down_in(x) + self.conv_gate(g))
        a = self.actv_att(self.conv_att(a))
        a = self.resampler(a)
        return x * a


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) with an additional global average pooling layer.

    See for more information:
    "DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs."
    by Chen et al. 2017.
    """

    def __init__(self, in_channels: int, in_size: int, inter_channels: int = 1, out_channels: Optional[int] = None,
                 base_dilation: int = 4, num_dilation_layers: int = 3, up_mode: str = "bilinear"):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=in_size),
                ConvBNReLU(in_channels=in_channels, out_channels=inter_channels, kernel_size=1),
                nn.Upsample(scale_factor=in_size, mode=up_mode)
            ),
            ConvBNReLU(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
        ])
        for i in range(1, num_dilation_layers + 1):
            self.layers.append(
                ConvBNReLU(in_channels=in_channels, dilation=base_dilation * i, out_channels=inter_channels)
            )

        self.out = ConvBNReLU(in_channels=inter_channels * (num_dilation_layers + 2), out_channels=out_channels,
                              kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(tuple([l(x) for l in self.layers]), dim=1)
        return self.out(x)


class EncodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_con: bool = True, num_conv: int = 2):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels, repeats=num_conv)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.skip_con = skip_con

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.conv_block(x)
        if self.skip_con:
            return self.pool(x), x  # -> Tuple[Tensor, Tensor]
        else:
            return self.pool(x)  # -> Tensor


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 3, num_conv_per_layer: int = 2):
        super().__init__()
        self.encoder = nn.ModuleList(
            [EncodBlock(in_channels=in_channels if i == 0 else 2 ** (i - 1) * base_channels,
                        out_channels=2 ** i * base_channels, num_conv=num_conv_per_layer) for i in range(depth)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        output = []
        for layer in self.encoder:
            x, s = layer(x)
            output.append(s)
        return x, tuple(output[::-1])


class DecodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, up_mode: str = "bilinear", skip_con: bool = True,
                 gate: bool = False, last_bn_relu: bool = True, num_conv: int = 2,
                 inter_channels_equal_out_channels: bool = True,
                 down_mode: Literal["max", "avg", "conv", "conv1x1"] = "conv",
                 min_inter_channels: Optional[int] = None):
        super().__init__()
        self.up = Up(in_channels=in_channels, mode=up_mode)
        if gate:
            skip_con = True
            self.gate = AG(in_channels=in_channels, actv_att=nn.Sigmoid(), down_mode=down_mode)
        else:
            self.gate = None
        in_c = 2 * in_channels if skip_con else in_channels
        inter_c = out_channels if inter_channels_equal_out_channels else in_c // 2
        inter_c = inter_c if min_inter_channels is None else max(min_inter_channels, inter_c)
        self.conv_block = ConvBlock(in_channels=in_c, inter_channels=inter_c, out_channels=out_channels,
                                    last_bn_relu=last_bn_relu, repeats=num_conv)

    def forward(self, x, skip: Optional[Tensor] = None):
        if skip is not None:
            if self.gate is not None:
                skip = self.gate(x=skip, g=x)
            x = self.up(x)
            x = torch.cat((x, skip), dim=1)
        else:
            x = self.up(x)
        return self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, depth: int = 3, gate: bool = False,
                 last_bn_relu: bool = True, num_conv_per_layer: int = 2,
                 inter_channels_equal_out_channels: bool = True,
                 down_mode: Literal["max", "avg", "conv", "conv1x1"] = "conv",
                 min_inter_channels: Optional[int] = None):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(depth):
            in_c = in_channels // 2 ** i
            out_c = out_channels if (i + 1) == depth and out_channels is not None else in_channels // 2 ** (i + 1)
            bn_relu = last_bn_relu if (i + 1) == depth else True
            self.decoder.append(
                DecodBlock(in_channels=in_c, out_channels=out_c, gate=gate, last_bn_relu=bn_relu,
                           num_conv=num_conv_per_layer, down_mode=down_mode,
                           inter_channels_equal_out_channels=inter_channels_equal_out_channels,
                           min_inter_channels=min_inter_channels)
            )

    def forward(self, x: Tensor, skip: Tuple[Tensor, ...]) -> Tensor:
        for (layer, s) in zip(self.decoder, skip):
            x = layer(x, s)
        return x


class Adapter(nn.Module):
    def __init__(self, in_channels: int, inter_channels: Optional[int] = None, mode: str = "seg"):
        super().__init__()
        # Coda
        if inter_channels is not None:
            self.coda = ConvBNReLU(in_channels=in_channels, out_channels=inter_channels)
            in_channels = inter_channels
        else:
            self.coda = None
        # Output
        if mode == "seg":
            self.o0 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "dist":
            self.o0 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "hv":
            self.o0 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.o1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError(f"Mode should be seg, dist or hv. Got instead {mode}.")
        self.mode = mode

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Coda
        if self.coda is not None:
            x = self.coda(x)
        # Output
        if self.mode == "seg":
            return {"seg_mask": self.o0(x), "cont_mask": self.o1(x)}
        elif self.mode == "dist":
            return {"dist_map": self.o0(x)}
        elif self.mode == "hv":
            return {"seg_mask": self.o0(x), "hv_map": self.o1(x)}


class SkipCompression(nn.Module):
    def __init__(self, base_channels_int: int, base_channels_ext: int, depth: int, num_skip: int):
        super().__init__()
        comp = []
        for i in range(depth - 1, -1, -1):
            channels_ext = 2 ** i * base_channels_ext
            channels_int = 2 ** i * base_channels_int
            comp.append(
                nn.Conv2d(in_channels=num_skip * channels_ext + channels_int, out_channels=channels_int, kernel_size=1)
            )
        self.comp = nn.ModuleList(comp)

    def forward(self, skip: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
        skip = [torch.cat(s, dim=1) for s in zip(*skip)]
        out = []
        for (comp, s) in zip(self.comp, skip):
            out.append(comp(s))
        return tuple(out)


class CodecBlock(nn.Module):
    """
    Codec block as a light-weighted U-Net.
    """

    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        self.e0 = ConvBNReLUMaxPool(in_channels=in_channels, out_channels=base_channels)
        self.e1 = ConvBNReLUMaxPool(in_channels=base_channels, out_channels=2 * base_channels)
        self.e2 = ConvBNReLUMaxPool(in_channels=2 * base_channels, out_channels=4 * base_channels)
        self.d2 = BLUpConvBNReLU(in_channels=4 * base_channels, out_channels=2 * base_channels)
        self.d1 = BLUpConvBNReLU(in_channels=4 * base_channels, out_channels=base_channels)
        self.d0 = BLUpConvBNReLU(in_channels=2 * base_channels, out_channels=base_channels)

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.e0(x)
        s1 = self.e1(s0)
        x = self.e2(s1)
        x = self.d2(x)
        x = self.d1(torch.cat((x, s1), dim=1))
        return self.d0(torch.cat((x, s0), dim=1))


class ALModule(nn.Module):
    """
    Attention Learning (AL) module with codec block.
    """

    def __init__(self, in_channels: int, base_channels: int = 16, codec_depth: int = 2):
        super().__init__()
        self.al_module = nn.Sequential(
            BLUpConvBNReLU(in_channels=in_channels, out_channels=in_channels),
            CodecBlock(in_channels=in_channels, base_channels=base_channels),
            nn.Sequential(
                nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(num_features=1),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        att = self.al_module(x)
        return att * x + x


class DACBlock(nn.Module):
    """
    Dense Atrouse Convolution (DAC) block with a variable number of cascade branches.
    """

    def __init__(self, in_channels: int, num_branches: int = 4):
        super().__init__()
        self.dac_block = nn.ModuleList()
        for branch_idx in range(num_branches):
            if branch_idx == 0:
                branch = ConvBNReLU(in_channels=in_channels, out_channels=in_channels)
            elif branch_idx == 1:
                branch = nn.Sequential(
                    ConvBNReLU(in_channels=in_channels, out_channels=in_channels, dilation=3),
                    ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
                )
            else:
                branch = nn.Sequential(
                    ConvBNReLU(in_channels=in_channels, out_channels=in_channels)
                )
                for i in range(branch_idx - 1):
                    dilation = 2 * i + 3
                    branch.append(
                        ConvBNReLU(in_channels=in_channels, out_channels=in_channels, dilation=dilation)
                    )
                branch.append(
                    ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
                )
            self.dac_block.append(branch)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape, device=x.device)
        for branch in self.dac_block:
            out += branch(x)
        return out


class SEBlockWithBN(nn.Module):
    def __init__(self, input_channels: int, squeeze_channels: Optional[int] = None):
        super().__init__()
        squeeze_channels = squeeze_channels or input_channels // 2
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=1),
            nn.Conv2d(in_channels=squeeze_channels, out_channels=input_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=input_channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale(x)


class CELayer(nn.Module):
    """
    Context Encoding (CE) layer consisting of a DAC block and a SE block.
    """

    def __init__(self, channels: int, dac_branches: int = 4, squeeze_channels: Optional[int] = None):
        super().__init__()
        # squeeze_channels = squeeze_channels or channels // 2
        self.context_encoding_layer = nn.Sequential(
            DACBlock(in_channels=channels, num_branches=dac_branches),
            SEBlockWithBN(input_channels=channels, squeeze_channels=squeeze_channels)
            # SEBlock(input_channels=channels, squeeze_channels=squeeze_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.context_encoding_layer(x) + x
