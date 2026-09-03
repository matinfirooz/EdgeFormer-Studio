import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT
from edgeformer.utils.dataset import get_loaders
from edgeformer.quantization import quantize_model_weights
from edgeformer.approximate import replace_linear_with_approx


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = TinyViT(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt["config"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--bits", type=int, default=32)
    p.add_argument("--approximate", action="store_true")
    p.add_argument("--noise-scale", type=float, default=0.01)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    if args.bits < 32:
        model = quantize_model_weights(model, args.bits).to(device)
    if args.approximate:
        model = replace_linear_with_approx(model, args.noise_scale).to(device)
    _, loader = get_loaders(args.dataset, cfg["training"]["batch_size"], cfg["training"]["num_workers"])
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    print(f"accuracy={100*correct/total:.2f}%")


if __name__ == "__main__":
    main()
