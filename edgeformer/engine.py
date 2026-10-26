"""Training and evaluation with sample-weighted metrics and mode restoration."""

import torch
from torch.nn import functional as F


def evaluate(model, loader, device="cpu", max_batches=None):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    modes = {m: m.training for m in model.modules()}
    total = correct = top5 = 0
    loss_sum = 0.0
    try:
        model.eval()
        with torch.inference_mode():
            for index, (images, targets) in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                correct += (logits.argmax(-1) == targets).sum().item()
                top5 += (
                    logits.topk(min(5, logits.shape[-1]), dim=-1)
                    .indices.eq(targets[:, None])
                    .any(-1)
                    .sum()
                    .item()
                )
                total += targets.numel()
    finally:
        for module, mode in modes.items():
            module.training = mode
    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return dict(
        loss=loss_sum / total, accuracy=correct / total, top5_accuracy=top5 / total, samples=total
    )


def train_epoch(
    model, loader, optimizer, device="cpu", *, grad_clip=1.0, label_smoothing=0.0, max_batches=None
):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * targets.numel()
        total += targets.numel()
        correct += (logits.detach().argmax(-1) == targets).sum().item()
    if not total:
        raise ValueError("Cannot train on an empty loader")
    return dict(loss=loss_sum / total, accuracy=correct / total, samples=total)
