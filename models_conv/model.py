from abc import abstractmethod

import numpy as np
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



class ConvNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = self._conv_block(373, 128)
        self.conv2 = self._conv_block(128, 64)
        self.conv3 = self._conv_block(64, 32)
        self.conv4 = self._conv_block(32, 16)

        self.flatten = nn.Flatten(1)

        self.fc1 = self._fc_block(256, 64, activation="relu")
        self.fc2 = self._fc_block(64, 1, activation="sigmoid")

    def _conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),
        )
        return conv_block

    def _fc_block(self, in_channels, out_channels, activation="relu"):
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        fc_block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # nn.Dropout(0.1),
            activation,
        )
        return fc_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
