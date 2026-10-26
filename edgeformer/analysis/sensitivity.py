"""One layer at a time, with fixed evaluation batches for fair comparisons."""

import torch
from torch import nn
from edgeformer.engine import evaluate
from edgeformer.quantization import quantize_model_weights
from edgeformer.analysis.metrics import output_distortion


def cache_batches(loader, max_batches=4):
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        batches.append((x.cpu(), y.cpu()))
    if not batches:
        raise ValueError("No calibration/evaluation batches")
    return batches


def layer_sensitivity(model, batches, bits=4, device="cpu"):
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        baseline = evaluate(model, batches, device)
        with torch.inference_mode():
            ref = torch.cat([model(x.to(device)).cpu() for x, _ in batches])
        rows = []
        names = [n for n, m in model.named_modules() if type(m) in {nn.Linear, nn.Conv2d}]
        for name in names:
            candidate = quantize_model_weights(model, 32, layer_bits={name: bits}).to(device).eval()
            metrics = evaluate(candidate, batches, device)
            with torch.inference_mode():
                pred = torch.cat([candidate(x.to(device)).cpu() for x, _ in batches])
            rows.append(
                dict(
                    layer=name,
                    bits=bits,
                    accuracy=metrics["accuracy"],
                    accuracy_drop=baseline["accuracy"] - metrics["accuracy"],
                    **output_distortion(ref, pred),
                )
            )
        return sorted(rows, key=lambda r: r["logit_rmse"], reverse=True)
    finally:
        for module, mode in modes.items():
            module.training = mode
