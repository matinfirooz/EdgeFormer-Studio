import torch


def approximate_add(a: torch.Tensor, b: torch.Tensor, noise_scale: float = 0.0):
    """Differentiable statistical approximation model for an adder."""
    out = a + b
    if noise_scale <= 0:
        return out
    sigma = out.detach().abs().mean().clamp_min(1e-8) * noise_scale
    return out + torch.randn_like(out) * sigma
