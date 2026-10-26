import copy
import math
import torch
from torch import nn


class ApproxLinear(nn.Linear):
    """Inference-only statistical output noise; not a circuit error model."""

    def __init__(self, in_features, out_features, bias=True, noise_scale=0.01, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        self.noise_scale = float(noise_scale)

    def forward(self, x):
        out = nn.functional.linear(x, self.weight, self.bias)
        if not self.training and self.noise_scale > 0:
            sigma = out.detach().abs().mean().clamp_min(1e-8) * self.noise_scale
            out = out + torch.randn_like(out) * sigma
        return out


def replace_linear_with_approx(model: nn.Module, noise_scale=0.01, *, layers=None):
    """Copy and wrap all or selected named layers, including a root Linear."""
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    selected = None if layers is None else set(layers)
    modules = dict(model.named_modules())
    for name in selected or ():
        if name not in modules or type(modules[name]) is not nn.Linear:
            raise ValueError(f"Unknown or unsupported approximation layer: {name}")
    result = copy.deepcopy(model)

    def convert(module, prefix=""):
        if isinstance(module, nn.Linear) and (selected is None or prefix in selected):
            if type(module) is not nn.Linear:
                raise ValueError("Approximation cannot replace QAT/custom Linear modules")
            out = ApproxLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                noise_scale=noise_scale,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            out.load_state_dict(module.state_dict())
            out.weight.requires_grad_(module.weight.requires_grad)
            if out.bias is not None:
                out.bias.requires_grad_(module.bias.requires_grad)
            return out.train(module.training)
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            setattr(module, name, convert(child, path))
        return module

    return convert(result)
