import torch
from edgeformer.approximate import ApproxLinear


def test_approx_linear_shape():
    layer = ApproxLinear(8, 4, noise_scale=0.01).eval()
    x = torch.randn(3, 8)
    assert layer(x).shape == (3, 4)
