from typing import Dict, Tuple, Optional, Union

import torch.nn as nn
from torch import Tensor

from models.building_blocks import ConvBlock, Encoder, Decoder, Adapter, ASPP, SkipCompression


class AttentionUNet(nn.Module):
    """
    Attention U-Net from Oktay et al. 2018
    """

    def __init__(self, in_channels: int = 3, out_channels: Optional[int] = None, mode: Optional[str] = None,
                 net_params: Optional[dict] = None):
        super().__init__()
        # Define default hyperparameters:
        default_params = {
            "base_channels": 32,
            "depth": 4,
            "aspp": False,
            "num_ext_skip": None,
            "base_channels_ext": 32,
            "adapter": None,
            "aspp_inter_channels": 16,
            "inter_channels_equal_out_channels": bool,
            "down_mode": "conv1x1",
            "aspp_global_avg": False,
            "img_size": 256
        }
        if net_params is None:
            net_params = {}
        # Overwrite default hyperparameters:
        net_params = {**default_params, **net_params}
        net_params["min_inter_channels"] = net_params["base_channels"]

        if out_channels is None:
            out_channels = net_params["base_channels"]
        # Encoder
        self.e = Encoder(in_channels=in_channels, base_channels=net_params["base_channels"], depth=net_params["depth"])

        # Skip compression
        if net_params["num_ext_skip"] is not None:
            self.sc = SkipCompression(base_channels_int=net_params["base_channels"], depth=net_params["depth"],
                                      base_channels_ext=net_params["base_channels_ext"],
                                      num_skip=net_params["num_ext_skip"])
        else:
            self.sc = None

        # Bottleneck
        bottle_channels = 2 ** (net_params["depth"] - 1) * net_params["base_channels"]
        if net_params["aspp"]:
            if net_params["aspp_global_avg"]:
                in_size = net_params["img_size"] // 2 ** (net_params["depth"])
            else:
                in_size = 2
            self.b = ASPP(in_channels=bottle_channels, in_size=in_size,
                          inter_channels=net_params["aspp_inter_channels"])
        else:
            self.b = ConvBlock(in_channels=bottle_channels, out_channels=bottle_channels)

        if net_params["adapter"] is None:
            if mode is None:
                # Decoder
                self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels,
                                 depth=net_params["depth"], gate=True, last_bn_relu=False,
                                 down_mode=net_params["down_mode"],
                                 inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
                                 min_inter_channels=net_params["min_inter_channels"])
                # Adaptor
                self.a = None
            else:
                # Decoder
                self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels,
                                 depth=net_params["depth"], gate=True, last_bn_relu=True,
                                 down_mode=net_params["down_mode"],
                                 inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
                                 min_inter_channels=net_params["min_inter_channels"])
                # Adaptor
                self.a = Adapter(in_channels=out_channels, inter_channels=128, mode=mode)
        else:
            # Decoder
            self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels,
                             depth=net_params["depth"], gate=True, last_bn_relu=True, down_mode=net_params["down_mode"],
                             inter_channels_equal_out_channels=net_params["inter_channels_equal_out_channels"],
                             min_inter_channels=net_params["min_inter_channels"])
            # Adapter
            self.a = net_params["adapter"]
        self.mode = mode

    def forward(self, x: Tensor, ext_skip: Optional[Tuple[Tuple[Tensor, ...], ...]] = None) \
            -> Union[Dict[str, Tensor], Tuple[Tensor, Tuple[Tensor, ...]]]:
        # Encoder
        x, skip = self.e(x)
        # Skip compression
        if self.sc is not None:
            ext_skip += (skip,)
            skip = self.sc(ext_skip)
        # Bottleneck
        x = self.b(x)
        # Decoder
        x = self.d(x, skip)
        # Adapter
        if self.mode is None:
            if self.a is None:
                return x, skip  # -> Tuple[Tensor, Tuple[Tensor, ...]]
            else:
                return self.a(x), skip  # -> Tuple[Tensor, Tuple[Tensor, ...]]
        else:
            return self.a(x)  # -> Dict[str, Tensor]
