import torch
from edgeformer.models.attention import MultiHeadSelfAttention


def test_attention_shapes():
    layer = MultiHeadSelfAttention(64, 4)
    x = torch.randn(2, 17, 64)
    y, attn = layer(x, return_attention=True)
    assert y.shape == (2, 17, 64)
    assert attn.shape == (2, 4, 17, 17)
