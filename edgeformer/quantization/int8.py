"""Weight-only PTQ simulation, including selective mixed-precision plans."""

import copy
import torch
from torch import nn
from .fake_quant import fake_quantize_tensor, validate_bits


def quantize_model_weights(
    model: nn.Module, bits=8, *, per_channel=True, layer_bits=None
) -> nn.Module:
    validate_bits(bits)
    plan = dict(layer_bits or {})
    modules = dict(model.named_modules())
    for name, width in plan.items():
        validate_bits(width)
        if name not in modules or type(modules[name]) not in {nn.Linear, nn.Conv2d}:
            raise ValueError(f"Unknown or unsupported quantization layer: {name}")
    qmodel = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in qmodel.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if type(module) not in {nn.Linear, nn.Conv2d}:
                    raise ValueError(
                        "PTQ expects a floating model, not an existing QAT/error wrapper"
                    )
                module.weight.copy_(
                    fake_quantize_tensor(
                        module.weight, plan.get(name, bits), per_channel=per_channel
                    )
                )
                # Bias, LayerNorm, class/position tokens deliberately stay floating.
    qmodel._edgeformer_transform = {
        "kind": "ptq",
        "bits": bits,
        "per_channel": per_channel,
        "layer_bits": plan,
    }
    return qmodel
