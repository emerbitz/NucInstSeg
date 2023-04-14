from typing import NoReturn, Tuple
import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation as SEBlock
import torchvision.models as models


class DownLayer(nn.Module):
    """Conv 3x3 + MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, act_fn=nn.ReLU) -> NoReturn:
        super(DownLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),  # Not mentioned in paper
            act_fn(),  # Activation function not specified
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UpLayer(nn.Module):
    """Upsample + Conv 3x3"""

    def __init__(self, in_channels: int, out_channels: int, act_fn=nn.ReLU, up_mode: str = "bilinear") -> NoReturn:
        super(UpLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode=up_mode),  # Mode not specified
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),  # Not mentioned in paper
            act_fn()  # Activation function not specified
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class CodecBlock(nn.Module):
    """Codec block as a two-layered U-Net"""

    def __init__(self, in_channels: int, feature_num: int, act_fn=nn.ReLU, up_mode: str = "bilinear") -> NoReturn:
        super(CodecBlock, self).__init__()
        self.e0 = DownLayer(in_channels=in_channels, out_channels=feature_num, act_fn=act_fn)
        self.e1 = DownLayer(in_channels=feature_num, out_channels=2 * feature_num, act_fn=act_fn)
        self.e2 = DownLayer(in_channels=2 * feature_num, out_channels=4 * feature_num, act_fn=act_fn)
        self.d0 = UpLayer(in_channels=4 * feature_num, out_channels=2 * feature_num, act_fn=act_fn, up_mode=up_mode)
        self.d1 = UpLayer(in_channels=4 * feature_num, out_channels=feature_num, act_fn=act_fn, up_mode=up_mode)
        self.d2 = UpLayer(in_channels=2 * feature_num, out_channels=feature_num, act_fn=act_fn, up_mode=up_mode)
        self.d3 = nn.Sequential(
            nn.Upsample(scale_factor=1.0, mode=up_mode),  # Mode not specified
            nn.Conv2d(in_channels=feature_num, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1),  # Not mentioned in paper
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_e0 = self.e0(x)  # x = (4, 64, 448, 448)
        x_e1 = self.e1(x_e0)  # x_e0 = (4, 16, 224, 224)
        x = self.e2(x_e1)  # x_e1 = (4, 32, 112, 112)
        # Decoder
        x = self.d0(x)  # x = (4, 64, 56, 56)
        x = self.d1(torch.cat([x, x_e1], dim=1))  # x = (4, 32, 112, 112), x_e1 = (4, 32, 112, 112)
        x = self.d2(torch.cat([x, x_e0], dim=1))
        return self.d3(x)


class ALModule(nn.Module):
    """Attention Learning (AL) module with codec block"""

    def __init__(self, channels: int, act_fn=nn.ReLU, up_mode: str = "bilinear") -> NoReturn:
        super(ALModule, self).__init__()
        self.attention = nn.Sequential(
            UpLayer(in_channels=channels, out_channels=channels, act_fn=act_fn, up_mode=up_mode),
            CodecBlock(in_channels=channels, feature_num=16, act_fn=act_fn, up_mode=up_mode),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x) * x + x


class DACBlock(nn.Module):
    """Dense Atrouse Convolution (DAC) block with four cascade branches"""

    def __init__(self, channels: int, act_fn=nn.ReLU) -> NoReturn:
        super(DACBlock, self).__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn()
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),  # Not mentioned in paper
            act_fn()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b0 = self.b0(x)
        x_b1 = self.b1(x)
        x_b2 = self.b2(x)
        x_b3 = self.b3(x)
        return x_b0 + x_b1 + x_b2 + x_b3


class CELayer(nn.Module):
    """Context Encoding (CE) layer with DAC block and SE block"""

    def __init__(self, channels: int, act_fn=nn.ReLU) -> NoReturn:
        super(CELayer, self).__init__()
        self.ce = nn.Sequential(
            DACBlock(channels=channels, act_fn=act_fn),
            SEBlock(input_channels=channels, squeeze_channels=channels, activation=act_fn)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ce(x)


class Encoder(nn.Module):
    """Consists of a pretrained ResNet34 with a CE layer before every pooling (except for the first pooling XD)"""

    def __init__(self, act_fn=nn.ReLU) -> NoReturn:
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.resnet.conv1 = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resnet.layer1[0].conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.ce_layer0 = CELayer(channels=32, act_fn=act_fn)
        self.ce_layer1 = CELayer(channels=64, act_fn=act_fn)
        self.ce_layer2 = CELayer(channels=128, act_fn=act_fn)
        self.ce_layer3 = CELayer(channels=256, act_fn=act_fn)
        # self.resnet.avgpool = None
        # self.resnet.fc = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        s0 = self.ce_layer0(x)

        x = self.resnet.maxpool(s0)
        x = self.resnet.layer1(x)
        s1 = self.ce_layer1(x)

        x = self.resnet.layer2(s1)
        s2 = self.ce_layer2(x)

        x = self.resnet.layer3(s2)
        s3 = self.ce_layer3(x)

        x = self.resnet.layer4(s3)
        return x, s0, s1, s2, s3


class Bridge(nn.Module):
    """Applies the Attention Learning (AL) module to the skip connections"""

    def __init__(self, act_fn=nn.ReLU, up_mode: str = "bilinear") -> NoReturn:
        super(Bridge, self).__init__()
        self.al_module0 = ALModule(channels=32, act_fn=act_fn, up_mode=up_mode)
        self.al_module1 = ALModule(channels=64, act_fn=act_fn, up_mode=up_mode)
        self.al_module2 = ALModule(channels=128, act_fn=act_fn, up_mode=up_mode)
        self.al_module3 = ALModule(channels=256, act_fn=act_fn, up_mode=up_mode)

    def forward(self, skip0: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor, skip3: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip0 = self.al_module0(skip0)
        skip1 = self.al_module1(skip1)
        skip2 = self.al_module2(skip2)
        skip3 = self.al_module3(skip3)
        return skip0, skip1, skip2, skip3


class DecoderMain(nn.Module):
    """AL-Net decoder for the main task"""

    def __init__(self, act_fn=nn.ReLU, up_mode: str = "bilinear") -> NoReturn:
        super(DecoderMain, self).__init__()
        self.up_mode = up_mode
        self.d1 = UpLayer(in_channels=512, out_channels=128, act_fn=act_fn, up_mode=up_mode)
        self.d2 = UpLayer(in_channels=256, out_channels=64, act_fn=act_fn, up_mode=up_mode)
        self.d3 = UpLayer(in_channels=128, out_channels=32, act_fn=act_fn, up_mode=up_mode)
        self.d4 = UpLayer(in_channels=64, out_channels=1, act_fn=act_fn, up_mode=up_mode)

    def forward(self, x: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor,
                skip3: torch.Tensor) -> torch.Tensor:
        *_, height, width = x.size()
        x = x.view((1, -1, 512, height, width))  # Add an additional dimension
        x = nn.Upsample(size=(256, 2*height, 2*width), mode=self.up_mode)(x)
        x = x.view((-1, 256, 2*height, 2*width))  # Remove the additional dimension
        x = torch.cat([x, skip3], dim=1)

        x = self.d1(x)
        x = torch.cat([x, skip2], dim=1)

        x = self.d2(x)
        x = torch.cat([x, skip1], dim=1)

        x = self.d3(x)
        x = torch.cat([x, skip0], dim=1)

        return self.d4(x)


class DecoderAuxiliary(nn.Module):
    """AL-Net decoder for the auxiliary task"""

    def __init__(self, up_mode: str = "bilinear") -> NoReturn:
        super(DecoderAuxiliary, self).__init__()
        self.branch0 = self.up_branch(num_upsampling=4, up_mode=up_mode)
        self.branch1 = self.up_branch(num_upsampling=3, up_mode=up_mode)
        self.branch2 = self.up_branch(num_upsampling=2, up_mode=up_mode)
        self.branch3 = self.up_branch(num_upsampling=1, up_mode=up_mode)

    @staticmethod
    def up_branch(num_upsampling: int, up_mode: str = "bilinear") -> nn.Sequential:
        return nn.Sequential(
            num_upsampling * nn.Sequential(nn.Upsample(scale_factor=2.0, mode=up_mode)),
            nn.Conv2d(in_channels=16 * 2 ** num_upsampling, out_channels=1, kernel_size=1)
        )

    def forward(self, skip0: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor,
                skip3: torch.Tensor) -> torch.Tensor:
        return self.branch0(skip3) + self.branch1(skip2) + self.branch2(skip1) + self.branch3(skip0)


class ALNet(nn.Module):
    """Attention Learning Network (AL-Net) as introduced by Zhao et al. 2022"""

    def __init__(self, act_fn=nn.ReLU, up_mode: str = "bilinear"):
        super(ALNet, self).__init__()
        self.encoder = Encoder(act_fn=act_fn)
        self.bridge = Bridge(act_fn=act_fn, up_mode=up_mode)
        self.decoder_main = DecoderMain(act_fn=act_fn, up_mode=up_mode)
        self.decoder_aux = DecoderAuxiliary(up_mode=up_mode)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, s0, s1, s2, s3 = self.encoder(x)
        s0, s1, s2, s3 = self.bridge(s0, s1, s2, s3)
        y_main = self.decoder_main(x, s0, s1, s2, s3)
        y_aux = self.decoder_aux(s0, s1, s2, s3)
        return y_main, y_aux
