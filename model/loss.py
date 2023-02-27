import torch
import torch.nn.functional as F


# classification loss
def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


# loss for attention map
def l1_vector_loss(attention):
    # calculate l1 loss on 1d band scaler vector (to emphasize sparsity)
    # and normalize by number of bands
    return torch.norm(attention, 1) / torch.prod(torch.tensor(attention.shape))


# combined loss
def loss(loss_type):
    if loss_type == "bce":
        return bce_loss
    elif loss_type == "l1":
        return l1_vector_loss
