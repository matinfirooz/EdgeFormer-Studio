from pathlib import Path
import matplotlib.pyplot as plt
import torch


def save_cls_attention(model, image, path="results/attention.png", layer=-1, head=0):
    model.eval()
    with torch.no_grad():
        _, maps = model(image.unsqueeze(0), return_attention=True)
    attn = maps[layer][0, head, 0, 1:]
    side = int(attn.numel() ** 0.5)
    attn = attn.reshape(side, side).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(attn)
    plt.axis("off")
    plt.title(f"CLS attention: layer {layer}, head {head}")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path
