import torch
from edgeformer.models import TinyViT


def test_vit_forward():
    model = TinyViT(embed_dim=96, depth=2, num_heads=3)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
