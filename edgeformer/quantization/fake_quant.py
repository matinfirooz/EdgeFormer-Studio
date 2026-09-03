import torch


def fake_quantize_tensor(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric fake quantization with dequantized floating output."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = 2 ** (bits - 1) - 1
    max_abs = x.detach().abs().max()
    if max_abs == 0:
        return x.clone()
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale
