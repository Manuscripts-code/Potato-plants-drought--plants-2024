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
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class ConvAutoencoderGroupsHyp(BaseModel):
    def __init__(self):
        super().__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(373*1, 373*2, 3, padding=1, groups=373),
            nn.BatchNorm2d(373*2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(373*2, 373*3, 3, padding=1, groups=373),
            nn.BatchNorm2d(373*3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(373*3, 373*4, 3, padding=1, groups=373),
            nn.BatchNorm2d(373*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(373*4, 373*3, 2, stride=2, groups=373),
            nn.BatchNorm2d(373*3),
            nn.ReLU(),
        )
        t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(373*3, 373*2, 2, stride=2, groups=373),
            nn.BatchNorm2d(373*2),
            nn.ReLU(),
        )
        t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(373*2, 373*1, 2, stride=2, groups=373),
            nn.BatchNorm2d(373*1),
            nn.Sigmoid(),
        )
        self.encoder = nn.Sequential(conv1, conv2, conv3)
        self.decoder = nn.Sequential(t_conv1, t_conv2, t_conv3)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(373*4, 373*1, 3, padding=1, groups=373),
            nn.BatchNorm2d(373*1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(8,8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(373, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x) # encode input hyp image
        decoded = self.decoder(encoded) # decode to get decoded hyp image
        encoded2 = self.encoder2(encoded) # further reduce dimension to 448 dim vector
        pred_class = self.classifier(encoded2) # predict from 448 dim vector
        return decoded, encoded2, pred_class



class ConvNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = self._conv_block(373, 128)
        self.conv2 = self._conv_block(128, 64)
        self.conv3 = self._conv_block(64, 32)
        self.conv4 = self._conv_block(32, 16)
        self.conv5 = self._conv_block(16, 8)

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
        )
        return conv_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x