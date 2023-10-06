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
            "depth": 3,
            "aspp": False,
            "num_ext_skip": None,
            "base_channels_ext": 32,
            "adapter": None,
            "aspp_inter_channels": 16,
            "inter_channels_equal_out_channels": bool,
            "down_mode": "max",
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
            self.b = ASPP(in_channels=bottle_channels, inter_channels=net_params["aspp_inter_channels"])
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

# class AttentionUNet(nn.Module):
#     def __init__(self, in_channels: int = 3, out_channels: Optional[int] = None, base_channels: int = 32,
#                  bottle_channels: int = 128, depth: int = 3, aspp: bool = False, num_ext_skip: Optional[int] = None,
#                  mode: Optional[str] = None, adapter: Optional[nn.Module] = None, aspp_inter_channels: int = 1,
#                  inter_channels_equal_out_channels: bool = True, down_mode: Literal["max", "avg", "conv"] = "conv",
#                  min_inter_channels: Optional[int] = None):
#         super().__init__()
#         if out_channels is None:
#             out_channels = base_channels
#         # Encoder
#         self.e = Encoder(in_channels=in_channels, base_channels=base_channels, depth=depth)
#         # Skip compression
#         if num_ext_skip is not None:
#             self.sc = SkipCompression(base_channels=base_channels, depth=depth, num_skip=num_ext_skip+1)
#         else:
#             self.sc = None
#         # Bottleneck
#         if aspp:
#             self.b = ASPP(in_channels=bottle_channels, inter_channels=aspp_inter_channels)
#         else:
#             # bottle_channels = 2 ** (depth - 1) * base_channels
#             self.b = ConvBlock(in_channels=bottle_channels, out_channels=bottle_channels)
#
#         if adapter is None:
#             if mode is None:
#                 # Decoder
#                 self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels, depth=depth, gate=True,
#                                  last_bn_relu=False, down_mode=down_mode,
#                                  inter_channels_equal_out_channels=inter_channels_equal_out_channels,
#                                  min_inter_channels=min_inter_channels)
#                 # Adaptor
#                 self.a = None
#             else:
#                 # Decoder
#                 self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels, depth=depth, gate=True,
#                                  last_bn_relu=True, down_mode=down_mode,
#                                  inter_channels_equal_out_channels=inter_channels_equal_out_channels,
#                                  min_inter_channels=min_inter_channels)
#                 # Adaptor
#                 self.a = Adapter(in_channels=out_channels, inter_channels=128, mode=mode)
#         else:
#             # Decoder
#             self.d = Decoder(in_channels=bottle_channels, out_channels=out_channels, depth=depth, gate=True,
#                              last_bn_relu=True, down_mode=down_mode,
#                              inter_channels_equal_out_channels=inter_channels_equal_out_channels,
#                              min_inter_channels=min_inter_channels)
#             # Adapter
#             self.a = adapter
#         self.mode = mode
#
#     def forward(self, x: Tensor, ext_skip: Optional[Tuple[Tuple[Tensor, ...], ...]] = None) \
#             -> Union[Dict[str, Tensor], Tuple[Tensor, Tuple[Tensor, ...]]]:
#         # Encoder
#         x, skip = self.e(x)
#         # Skip compression
#         if self.sc is not None:
#             ext_skip += (skip,)
#             skip = self.sc(ext_skip)
#         # Bottleneck
#         x = self.b(x)
#         # Decoder
#         x = self.d(x, skip)
#         # Adapter
#         if self.mode is None:
#             if self.a is None:
#                 return x, skip  # -> Tuple[Tensor, Tuple[Tensor, ...]]
#             else:
#                 return self.a(x), skip  # -> Tuple[Tensor, Tuple[Tensor, ...]]
#         else:
#             return self.a(x)  # -> Dict[str, Tensor]


# class ParaAttUNet(nn.Module):
#     """
#     An independent U-Net for the learning of each ground truth.
#     """
#     def __init__(self, in_channels: int = 3, mode: Literal["seg", "dist", "hv"] = "seg"):
#         super().__init__()
#         adapter = nn.Sequential(
#             ConvBNReLU(in_channels=32, out_channels=128),
#             nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
#         )
#         # Main branch
#         self.mb = AttentionUNet(in_channels=in_channels, out_channels=32, aspp=True, adapter=adapter)
#
#         if mode == "seg":
#             # Auxiliary branch
#             self.ab = AttentionUNet(in_channels=in_channels, out_channels=32, aspp=True, adapter=adapter)
#             # # Compression
#             # self.comp = None
#             # # Final branch
#             # self.fb = AttentionUNet(in_channels=3*in_channels, out_channels=1, aspp=True, num_ext_skip=2)
#         elif mode == "dist":
#             # Auxiliary branch
#             self.ab = None
#             # # Final branch
#             # self.fb = AttentionUNet(in_channels=2*in_channels, out_channels=1, aspp=True, num_ext_skip=1)
#         elif mode == "hv":
#             # Auxiliary branch
#             self.ab = AttentionUNet(in_channels=in_channels, out_channels=32, aspp=True)
#             # # Compression
#             # self.comp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
#             # # Final branch
#             # self.fb = AttentionUNet(in_channels=3*in_channels, out_channels=1, aspp=True, num_ext_skip=2)
#         self.mode = mode   # Fetched by NetModel: Selects loss term and postprocessing scheme
#         self.additional_loss_term = False  # Fetched by NetModel: Adapts loss term
#
#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         x_main, _ = self.mb(x)
#         if self.ab is not None:
#             x_aux, _ = self.ab(x)
#
#         if self.mode == "seg":
#             return {"seg_mask": x_main, "cont_mask": x_aux}
#         elif self.mode == "dist":
#             return {"dist_map": x_main}
#         elif self.mode == "hv":
#             return {"seg_mask": x_main, "hv_map": x_aux}
