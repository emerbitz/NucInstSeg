import torch
import torch.nn as nn


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


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=(2., 2.))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class EncodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv_block(x)
        return self.pool(x), x


class DecodBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = UpConv(in_channels=in_channels, out_channels=out_channels)
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, feature_num: int):
        super().__init__()
        # Encoder
        self.e1 = EncodBlock(in_channels=in_channels, out_channels=feature_num)
        self.e2 = EncodBlock(in_channels=feature_num, out_channels=2 * feature_num)
        self.e3 = EncodBlock(in_channels=2 * feature_num, out_channels=4 * feature_num)
        self.e4 = EncodBlock(in_channels=4 * feature_num, out_channels=8 * feature_num)
        # Bottleneck
        self.b = ConvBlock(in_channels=8 * feature_num, out_channels=16 * feature_num)
        # Decoder
        self.d1 = DecodBlock(in_channels=16 * feature_num, out_channels=8 * feature_num)
        self.d2 = DecodBlock(in_channels=8 * feature_num, out_channels=4 * feature_num)
        self.d3 = DecodBlock(in_channels=4 * feature_num, out_channels=2 * feature_num)
        self.d4 = DecodBlock(in_channels=2 * feature_num, out_channels=feature_num)
        # Classifier
        self.c = nn.Conv2d(in_channels=feature_num, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

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