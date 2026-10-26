"""Symmetric narrow-range quantization with optional straight-through gradients."""

import torch


def validate_bits(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {*range(2, 17), 32}:
        raise ValueError("bits must be an integer in 2..16, or 32 for identity")


def quantization_scale(x, bits=8, per_channel=False, channel_axis=0):
    validate_bits(bits)
    if not x.is_floating_point() or x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("quantization requires a nonempty, finite floating tensor")
    if per_channel:
        if x.ndim == 0 or not -x.ndim <= channel_axis < x.ndim:
            raise ValueError("invalid channel_axis")
        dims = tuple(d for d in range(x.ndim) if d != channel_axis % x.ndim)
        peak = x.detach().abs().amax(dim=dims, keepdim=True) if dims else x.detach().abs()
    else:
        peak = x.detach().abs().amax()
    qmax = 2 ** (bits - 1) - 1
    return (peak / qmax).clamp_min(torch.finfo(x.dtype).tiny)


def fake_quantize_tensor(
    x: torch.Tensor, bits: int = 8, *, per_channel=False, channel_axis=0, ste=False
) -> torch.Tensor:
    """Return floating Q/DQ; ste=True gives an identity surrogate gradient.

    Weights use one scale per output channel; activations use a dynamic scale
    per input tensor. No observer calibration or integer kernel is implied.
    """
    validate_bits(bits)
    if bits == 32:
        return x.clone()
    scale = quantization_scale(x, bits, per_channel, channel_axis)
    qmax = 2 ** (bits - 1) - 1
    dequant = (x / scale).round().clamp(-qmax, qmax) * scale
    return x + (dequant - x).detach() if ste else dequant
