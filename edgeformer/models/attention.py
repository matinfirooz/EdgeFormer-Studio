import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """Small readable multi-head self-attention implementation.

    Set ``return_attention=True`` in forward to retrieve attention maps.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, backend: str = "sdpa"):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if backend not in {"manual", "sdpa"}:
            raise ValueError("backend must be manual or sdpa")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.backend = backend
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if return_attention or self.backend == "manual":
            scores = (q @ k.transpose(-2, -1)) * self.scale
            attn = scores.softmax(dim=-1)
            out = self.dropout(attn) @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        if return_attention:
            # Return probabilities before dropout so each row sums to one.
            return out, attn
        return out
