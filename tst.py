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


class ConvAutoencoder1(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 3
        params_conv = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 0,
            "groups": num_channels
        }
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, **params_conv)  
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, **params_conv)  
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=12, **params_conv)  
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=15, **params_conv)  
        self.pool = nn.MaxPool2d(3, 3)
        params_conv["kernel_size"] = 2
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=15, out_channels=9, **params_conv)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=9, out_channels=6, **params_conv)
        self.conv_trans3 = nn.ConvTranspose2d(in_channels=6, out_channels=3, **params_conv)
        self.conv_trans4 = nn.ConvTranspose2d(in_channels=3, out_channels=3, **params_conv)
        self.conv_trans5 = nn.ConvTranspose2d(in_channels=3, out_channels=3, **params_conv)


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.pool(x)
        # x = F.relu(self.conv_trans1(x))
        # x = F.relu(self.conv_trans2(x))
        # x = F.relu(self.conv_trans3(x))
        # x = F.relu(self.conv_trans4(x))
        # x = F.relu(self.conv_trans5(x))
        # x = F.sigmoid(x)
        return x

# x = pool(x)
# x = F.relu(conv2(x))
# x = pool(x)
# x = F.relu(t_conv1(x))
# x = F.sigmoid(t_conv2(x))


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(160*1, 160*5, 3, padding=1, groups=160),
            nn.BatchNorm2d(160*5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(160*5, 160*7, 3, padding=1, groups=160),
            nn.BatchNorm2d(160*7),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(160*7, 160*5, 3, padding=1, groups=160),
            nn.BatchNorm2d(160*5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(160*5, 160*7, 2, stride=2, groups=160),
            nn.BatchNorm2d(160*7),
            nn.ReLU(),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(160*7, 160*5, 2, stride=2, groups=160),
            nn.BatchNorm2d(160*5),
            nn.ReLU(),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(160*5, 160*1, 2, stride=2, groups=160),
            nn.BatchNorm2d(160*1),
            nn.Sigmoid(),
        )
        # self.encoder = nn.Sequential(conv1, conv2, conv3)
        # self.decoder = nn.Sequential(t_conv1, t_conv2, t_conv3)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(160*5, 160*1, 3, padding=1, groups=160),
            nn.BatchNorm2d(160*1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(8,8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(160, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.t_conv1(x)
        x = self.t_conv2(x)
        x = self.t_conv3(x)
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # encoded2 = self.encoder2(encoded)
        # pred_class = self.classifier(encoded2)
        return decoded, pred_class

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

network = ConvAutoencoder()
x = torch.rand(32, 160, 64, 64)
decoded, pred_class = network.forward(x)
print(decoded.shape)
print(network)

# from data_loader.data_loaders import PotatosDataset

# root_dir = "C:\\Users\\janezla\\Documents\\DATA\\slikanje_3_images_sliced_5thresh"
# data = PotatosDataset(root_dir, train=False)
# pass


# input = torch.randn(5, 160, 4, 4)
# # With default parameters
# decoded = input.view(-1, 2560)
# print(decoded.size())



