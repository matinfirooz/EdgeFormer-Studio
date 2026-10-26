from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def attention_rollout(maps):
    """Head-mean attention with identity residuals, multiplied in layer order."""
    if not maps:
        raise ValueError("No attention maps")
    tokens = maps[0].shape[-1]
    joint = torch.eye(tokens, device=maps[0].device, dtype=maps[0].dtype)
    for attn in maps:
        mean = attn[0].mean(0)
        transition = mean + torch.eye(tokens, device=mean.device, dtype=mean.dtype)
        transition = transition / transition.sum(-1, keepdim=True)
        joint = transition @ joint
    return joint[0, 1:]


def _extract(model, image):
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        with torch.inference_mode():
            _, maps = model(image.unsqueeze(0), return_attention=True)
        return maps
    finally:
        for module, mode in modes.items():
            module.training = mode


def save_cls_attention(model, image, path="results/attention.png", layer=-1, head=0):
    maps = _extract(model, image)
    attn = maps[layer][0, head, 0, 1:]
    side = model.image_size // model.patch_size
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(attn.reshape(side, side).cpu().numpy(), cmap="magma")
    ax.set(title=f"CLS attention: layer {layer}, head {head}", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_attention_report(model, image, path="results/attention.png"):
    maps = _extract(model, image)
    side = model.image_size // model.patch_size
    panels = len(maps) + 2
    columns = min(4, panels)
    fig, axes = plt.subplots(
        math.ceil(panels / columns),
        columns,
        figsize=(3.2 * columns, 3.2 * math.ceil(panels / columns)),
        squeeze=False,
    )
    axes = axes.ravel()
    image_np = image.detach().float().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    if image_np.shape[-1] == 1:
        image_np = image_np[..., 0]
    elif image_np.shape[-1] != 3:
        image_np = image_np.mean(-1)
    axes[0].imshow(image_np.clip(0, 1))
    axes[0].set_title("Input image")
    for i, attn in enumerate(maps):
        patch = attn[0].mean(0)[0, 1:].reshape(side, side).cpu().numpy()
        axes[i + 1].imshow(patch, cmap="magma")
        axes[i + 1].set_title(f"Block {i}: head-mean CLS")
    rollout = attention_rollout(maps).reshape(1, 1, side, side)
    heat = (
        F.interpolate(rollout, size=(model.image_size,) * 2, mode="bilinear", align_corners=False)[
            0, 0
        ]
        .cpu()
        .numpy()
    )
    axes[len(maps) + 1].imshow(image_np.clip(0, 1))
    axes[len(maps) + 1].imshow(heat, alpha=0.6, cmap="magma")
    axes[len(maps) + 1].set_title("Attention rollout")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("EdgeFormer Studio | attention inspection", fontsize=14)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
