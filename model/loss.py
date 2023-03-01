from typing import Callable, Protocol

import torch
import torch.nn.functional as F


class BaseLoss(Protocol):
    def __call__(self, loss_type: str) -> Callable[..., torch.Tensor]:
        ...


class Loss(BaseLoss):
    def __init__(self, l1_lambda):
        self.l1_lambda = l1_lambda

    # classification loss
    def _bce_loss(self, output, target):
        return F.binary_cross_entropy(output, target)

    # loss for attention map
    def _l1_vector_loss(self, attention):
        # calculate l1 loss on 1d band scaler vector (to emphasize sparsity)
        # first calculate the mean of attentions from all the batches
        attention_norm = attention.mean(0)
        # apply transformation function
        attention_norm = 1 - (attention_norm / 0.95 - 1) ** 2
        # calculate l1 penalty on 1d vector and divide by batch size
        attention_norm = torch.sum(attention_norm) / attention.shape[1]
        return attention_norm * self.l1_lambda

    # combined loss
    def __call__(self, loss_type):
        if loss_type == "bce":
            return self._bce_loss
        elif loss_type == "l1":
            return self._l1_vector_loss
