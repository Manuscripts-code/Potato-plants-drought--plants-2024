import torch
import torch.nn.functional as F


# classification loss
def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


# loss for attention map
def l1_vector_loss(attention):
    # calculate l1 loss on 1d band scaler vector (to emphasize sparsity)
    # first calculate the mean of attentions from all the batches
    attention_norm = attention.mean(0)
    # apply transformation function
    attention_norm = 1 - (attention_norm / 0.95 - 1) ** 2
    # calculate l1 penalty on 1d vector and divide by batch size
    attention_norm = torch.sum(attention_norm) / attention.shape[1]
    return attention_norm


# combined loss
def loss(loss_type):
    if loss_type == "bce":
        return bce_loss
    elif loss_type == "l1":
        return l1_vector_loss
