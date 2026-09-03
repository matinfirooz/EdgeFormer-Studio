def estimate_vit_macs(model) -> dict:
    """Analytic MAC estimate for this repository's TinyViT model."""
    n = model.num_patches + 1
    d = model.embed_dim
    depth = len(model.blocks)
    h = model.blocks[0].attn.num_heads
    head_dim = d // h
    qkv = 3 * n * d * d
    attention = h * n * n * head_dim * 2
    proj = n * d * d
    mlp_hidden = model.blocks[0].mlp.net[0].out_features
    mlp = 2 * n * d * mlp_hidden
    per_block = qkv + attention + proj + mlp
    patch = model.num_patches * (model.patch_size ** 2 * 3) * d
    head = d * model.head.out_features
    total = patch + depth * per_block + head
    return {
        "patch_embed": int(patch),
        "attention": int(depth * (qkv + attention + proj)),
        "mlp": int(depth * mlp),
        "classifier": int(head),
        "total": int(total),
    }
