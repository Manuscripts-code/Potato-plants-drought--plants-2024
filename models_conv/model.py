from abc import abstractmethod

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


class BasicBlock(nn.Module):
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
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

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


class ConvNet(BaseModel):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        img_channels=373,  # number of spectral bands
        res_channels=64,
        num_classes=1,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        normalization=True,
        activation=nn.ReLU(),
    ):
        super(ConvNet, self).__init__()

        self.inplanes = res_channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.normalization = normalization
        self.activation = activation

        self.conv1 = BasicConv(
            img_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            normalization=self.normalization,
        )

        self.act = activation
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.layer1 = self._make_layer(block, res_channels, layers[0])
        self.layer2 = self._make_layer(
            block, res_channels * 2, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, res_channels * 4, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, res_channels * 8, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.fc = nn.Sequential(
            nn.Linear(res_channels * 8 * block.expansion, num_classes),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(
                m,
                (
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.GroupNorm,
                        nn.LayerNorm,
                        nn.InstanceNorm1d,
                        nn.InstanceNorm2d,
                        nn.InstanceNorm3d,
                    )
                ),
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
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
                previous_dilation,
                normalization=self.normalization,
                activation=self.activation,
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
                    activation=self.activation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
