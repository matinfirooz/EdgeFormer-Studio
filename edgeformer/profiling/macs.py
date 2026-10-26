"""Per-image multiply-accumulate counts, excluding elementwise/nonlinear ops."""


def estimate_vit_macs(model):
    n, d = model.num_patches + 1, model.embed_dim
    patch = model.num_patches * model.patch_embed.weight[0].numel() * d
    qkv = sum(3 * n * d * d for _ in model.blocks)
    qk = sum(n * n * d for _ in model.blocks)
    av = qk
    proj = sum(n * d * d for _ in model.blocks)
    mlp = sum(2 * n * d * b.mlp.net[0].out_features for b in model.blocks)
    head = d * model.head.out_features
    attention = qkv + qk + av + proj
    return dict(
        patch_embed=patch,
        qkv=qkv,
        qk=qk,
        av=av,
        attention_projection=proj,
        attention=attention,
        mlp=mlp,
        classifier=head,
        total=patch + attention + mlp + head,
    )
