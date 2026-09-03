import argparse
from pathlib import Path
import sys
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT
from edgeformer.utils.dataset import get_loaders
from edgeformer.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cifar10.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", default="checkpoints/vit_cifar10.pt")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    dataset = args.dataset or ("cifar100" if cfg["model"]["num_classes"] == 100 else "cifar10")
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_loader, test_loader = get_loaders(dataset, batch_size, cfg["training"]["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyViT(**cfg["model"]).to(device)
    opt = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train(); total = correct = 0; running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            running += loss.item() * y.size(0)
            total += y.size(0); correct += (logits.argmax(1) == y).sum().item()
        print(f"train loss={running/total:.4f} acc={100*correct/total:.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, args.output)
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
