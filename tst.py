import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# _ = torch.manual_seed (2021)
# output = torch.rand(5, 2, 10, 10)
# target = torch.rand(5, 2, 10, 10)
# out_loss = F.binary_cross_entropy(output, target)
# out_loss = F.l1_loss(output, target, reduction="none")
# print(out_loss)


class ConvNet(nn.Module):
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

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


# network = ConvNet()
# x = torch.rand(32, 373, 32, 32)
# x = network.conv1(x)
# x = network.conv2(x)
# x = network.conv3(x)
# x = network.conv4(x)
# x = network.conv5(x)
# # x = network.fc(x)
# print(x.shape)
# # print(network)


class AreaNormalization:
    def __call__(self, image):
        image = self._image_normalization(image, self._signal_normalize)
        return image.astype(dtype="float32")

    @staticmethod
    def _image_normalization(image, func1d):
        return np.apply_along_axis(func1d, axis=2, arr=image)

    @staticmethod
    def _signal_normalize(signal):
        area = np.trapz(signal)
        if area == 0:
            return signal
        return signal / area

x = np.random.rand(5, 5, 2).astype(dtype="float32")

rand = AreaNormalization()
print(x)
# print("")
print(rand(x))


from numba import njit, prange


@njit()
def _signal_normalize(signal):
    area = np.float32(np.trapz(signal))
    if area == 0:
        return signal
    return signal / area

@njit(parallel=True)
def AreaNormalization(image):
    image_out = np.zeros_like(image)
    n_rows, n_cols, _ = image.shape
    for v in prange(n_rows):
        for u in prange(n_cols):
            image_out[v, u, :] = _signal_normalize(image[v, u, :])
    return image_out


x = np.random.rand(5, 5, 20).astype(dtype="float32")

print(x.shape)
x = AreaNormalization(x)
print("a")
print(x.shape)
