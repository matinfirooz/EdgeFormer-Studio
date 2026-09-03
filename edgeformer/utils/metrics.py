import torch


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()
