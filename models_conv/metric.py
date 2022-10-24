import torch
import torch.nn.functional as F


def accuracy_(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy(output, target):
    output = output.view(-1)
    target = target.view(-1)
    assert target.ndim == 1 and target.size() == output.size()
    output = output > 0.5
    return (target == output).sum().item() / target.size(0)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def l1_mean(output, target):
    with torch.no_grad():
        out_loss = F.l1_loss(output, target)
    return out_loss


def l1_max(output, target):
    with torch.no_grad():
        out_loss = F.l1_loss(output, target, reduction="none").max()
    return out_loss
