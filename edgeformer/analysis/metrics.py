"""Numerical distortion and exact nondomination over finite objectives."""

import math
import torch
from torch.nn import functional as F


def output_distortion(reference, candidate):
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("outputs must have matching nonempty shapes")
    delta = candidate.float() - reference.float()
    return dict(
        logit_mae=delta.abs().mean().item(),
        logit_rmse=delta.square().mean().sqrt().item(),
        agreement=reference.argmax(-1).eq(candidate.argmax(-1)).float().mean().item(),
        logit_cosine=F.cosine_similarity(reference.float(), candidate.float(), dim=-1)
        .mean()
        .item(),
    )


def attention_distortion(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("attention lists must have equal positive length")
    maes, jss = [], []
    for p, q in zip(reference, candidate):
        if p.shape != q.shape:
            raise ValueError("attention maps must have matching shapes")
        p, q = p.float().clamp_min(1e-12), q.float().clamp_min(1e-12)
        p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
        m = 0.5 * (p + q)
        maes.append((p - q).abs().mean())
        jss.append(
            0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1)).mean()
        )
    return dict(
        attention_mae=torch.stack(maes).mean().item(),
        attention_js_divergence=max(0.0, torch.stack(jss).mean().item()),
    )


def pareto_mask(rows, *, minimize=("ideal_packed_total_bytes",), maximize=("accuracy",)):
    """Keep duplicates; dominance requires no worse everywhere and better somewhere."""
    if not minimize and not maximize:
        raise ValueError("At least one objective is required")
    vectors = [[*(float(r[k]) for k in minimize), *(-float(r[k]) for k in maximize)] for r in rows]
    if any(not math.isfinite(x) for v in vectors for x in v):
        raise ValueError("Pareto objectives must be finite")
    return [
        not any(
            all(y <= x for x, y in zip(v, other)) and any(y < x for x, y in zip(v, other))
            for other in vectors
        )
        for v in vectors
    ]
