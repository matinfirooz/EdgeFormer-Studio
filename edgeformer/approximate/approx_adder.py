import math
import torch


def approximate_add(a, b, noise_scale=0.0):
    """Differentiable statistical output error; no circuit correspondence implied."""
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    out = a + b
    if noise_scale == 0:
        return out
    sigma = out.detach().abs().mean().clamp_min(1e-8) * noise_scale
    return out + torch.randn_like(out) * sigma
