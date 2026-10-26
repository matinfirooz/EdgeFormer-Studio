import torch
from edgeformer.quantization.fake_quant import fake_quantize_tensor


def test_fake_quant_shape_and_finite():
    x = torch.randn(16, 16)
    y = fake_quantize_tensor(x, 8)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
