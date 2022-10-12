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
            nn.Conv2d(448*1, 448*2, 3, padding=1, groups=448),
            nn.BatchNorm2d(448*2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(448*2, 448*3, 3, padding=1, groups=448),
            nn.BatchNorm2d(448*3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(448*3, 448*4, 3, padding=1, groups=448),
            nn.BatchNorm2d(448*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(448*4, 448*3, 2, stride=2, groups=448),
            nn.BatchNorm2d(448*3),
            nn.ReLU(),
        )
        t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(448*3, 448*2, 2, stride=2, groups=448),
            nn.BatchNorm2d(448*2),
            nn.ReLU(),
        )
        t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(448*2, 448*1, 2, stride=2, groups=448),
            nn.BatchNorm2d(448*1),
            nn.Sigmoid(),
        )
        self.encoder = nn.Sequential(conv1, conv2, conv3)
        self.decoder = nn.Sequential(t_conv1, t_conv2, t_conv3)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(448*4, 448*1, 3, padding=1, groups=448),
            nn.BatchNorm2d(448*1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(8,8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(448, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x) # encode input hyp image
        decoded = self.decoder(encoded) # decode to get decoded hyp image
        encoded2 = self.encoder2(encoded) # further reduce dimension to 448 dim vector
        pred_class = self.classifier(encoded2) # predict from 448 dim vector
        return decoded, encoded2, pred_class
