import torch
from sklearn.metrics import f1_score


def accuracy(output, target):
    with torch.no_grad():
        output = output.view(-1).round()
        target = target.view(-1)
        assert target.ndim == 1 and target.size() == output.size()
    return (target == output).sum().item() / target.size(0)


def f1(output, target):
    with torch.no_grad():
        output = output.view(-1).round()
        target = target.view(-1)
        assert target.ndim == 1 and target.size() == output.size()
        score = f1_score(output.cpu(), target.cpu(), average="weighted")
    return score
