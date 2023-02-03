from abc import abstractmethod
from configs import configs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class BasicConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        normalization=True,
        activation=nn.ReLU(True),
        act_first=True,
    ):
        super(BasicConv, self).__init__()

        basic_conv = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if activation and act_first:
            basic_conv.append(activation)
        if normalization:
            basic_conv.append(nn.BatchNorm2d(channels_out))
        if activation and not act_first:
            basic_conv.append(activation)
        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        normalization=True,
        activation=nn.ReLU(),
    ):
        super(BasicResidualBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = BasicConv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            normalization=normalization,
            activation=False,
        )

        self.act = activation
        self.conv2 = BasicConv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            normalization=normalization,
            activation=False,
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


class SpectralAttentionBlock(nn.Module):
    def __init__(self, inplanes, reduction=2):
        super(SpectralAttentionBlock, self).__init__()

        self.inplanes = inplanes

        mid_planes = inplanes // reduction
        self.fc1 = nn.Linear(inplanes, mid_planes)
        self.fc2 = nn.Linear(mid_planes, inplanes)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = torch.mean(x, dim=(2, 3))
        out = self.fc1(out)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = out[:, :, None, None]
        out = out * identity
        return out


class ConvNet(BaseModel):
    def __init__(
        self,
        layers,
        img_channels,  # number of spectral bands
        res_channels=64,
        num_classes=1,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        normalization=True,
    ):
        super(ConvNet, self).__init__()

        self.inplanes = res_channels
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.normalization = normalization

        self.spectral = SpectralAttentionBlock(img_channels)
        self.conv1 = BasicConv(
            img_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            normalization=self.normalization,
            act_first=False,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = self._make_layer(BasicResidualBlock, res_channels, layers[0])
        self.layer2 = self._make_layer(BasicResidualBlock, res_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicResidualBlock, res_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicResidualBlock, res_channels * 8, layers[3], stride=2)
        self.fc = nn.Sequential(
            nn.Linear(res_channels * 8 * BasicResidualBlock.expansion, num_classes), nn.Sigmoid()
        )
        self._init_weights(zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = BasicConv(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                normalization=self.normalization,
                activation=False,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                normalization=self.normalization,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    normalization=self.normalization,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # TODO
            elif isinstance(m, SpectralAttentionBlock):
                nn.init.constant_(m.fc1.weight, 0.1)
                nn.init.constant_(m.fc1.bias, 0)
                nn.init.constant_(m.fc2.weight, 0.1)
                nn.init.constant_(m.fc2.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.spectral(x)
        x = self.conv1(x)  # torch.Size([32, 64, 32, 32])
        x = self.maxpool(x)  # torch.Size([32, 64, 16, 16])

        x = self.layer1(x)  # torch.Size([32, 64, 16, 16])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
