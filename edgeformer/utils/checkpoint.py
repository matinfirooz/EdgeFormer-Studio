"""Versioned state-dict checkpoints; compatible with the original archive."""

import copy
import os
from pathlib import Path
import torch
from edgeformer.models import TinyViT
from edgeformer.quantization import prepare_qat


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(model, config, path, **state):
    cfg = copy.deepcopy(config)
    cfg["model"] = dict(model.config)
    payload = dict(
        format_version=2,
        model=model.state_dict(),
        config=cfg,
        transform=getattr(model, "_edgeformer_transform", {"kind": "fp32"}),
        **state,
    )
    atomic_save(payload, path)
    return payload


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise ValueError("Expected a checkpoint with model and config mappings")
    if ckpt.get("format_version", 1) > 2:
        raise ValueError("This checkpoint requires a newer EdgeFormer version")
    model = TinyViT(**ckpt["config"]["model"])
    transform = ckpt.get(
        "transform",
        {"kind": "ptq", "bits": ckpt["bits"], "per_channel": False}
        if "bits" in ckpt
        else {"kind": "fp32"},
    )
    if transform["kind"] == "qat":
        model = prepare_qat(model, transform["bits"])
    elif transform["kind"] not in {"ptq", "fp32"}:
        raise ValueError(f"Unsupported transform {transform['kind']}")
    model.load_state_dict(ckpt["model"], strict=True)
    model._edgeformer_transform = transform
    return model.to(device), ckpt
