"""CIFAR, original/prepared Tiny ImageNet, and deterministic offline smoke data."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from .seed import seed_worker


class TinyImageNetValidation(Dataset):
    def __init__(self, root, classes, transform):
        self.transform = transform
        self.class_to_idx = {n: i for i, n in enumerate(classes)}
        root = Path(root)
        annotation = root / "val_annotations.txt"
        if not annotation.exists():
            raise FileNotFoundError(f"Missing {annotation}")
        self.samples = []
        for line in annotation.read_text().splitlines():
            name, label, *_ = line.split()
            if label not in self.class_to_idx:
                raise ValueError(f"Unknown validation class: {label}")
            self.samples.append((root / "images" / name, self.class_to_idx[label]))
        if not self.samples:
            raise ValueError("Tiny ImageNet validation set is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), target


def synthetic_dataset(samples, image_size, in_channels, num_classes, seed):
    """Small learnable patterns with seeded noise; never a dataset accuracy claim."""
    if samples < 1:
        raise ValueError("synthetic sample count must be positive")
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % num_classes
    images = torch.randn(samples, in_channels, image_size, image_size, generator=g) * 0.1
    for i, target in enumerate(labels.tolist()):
        channel = target % in_channels
        column = (target // in_channels * 3) % image_size
        images[i, channel, :, column : min(column + 2, image_size)] += 1
    return TensorDataset(images, labels)


def get_loaders(
    name="cifar10",
    batch_size=128,
    num_workers=0,
    data_dir="./data",
    *,
    image_size=32,
    in_channels=3,
    num_classes=None,
    download=False,
    seed=42,
    train_samples=128,
    val_samples=64,
    eval_only=False,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    name = name.lower()
    expected = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}
    if name in expected and num_classes is not None and num_classes != expected[name]:
        raise ValueError(f"{name} requires num_classes={expected[name]}")
    if name == "synthetic":
        kw = dict(image_size=image_size, in_channels=in_channels, num_classes=num_classes or 10)
        train = None if eval_only else synthetic_dataset(train_samples, seed=seed, **kw)
        val = synthetic_dataset(val_samples, seed=seed + 100000, **kw)
    else:
        # Keep torchvision optional for core model and offline workflows.
        from torchvision import datasets, transforms

        if in_channels != 3:
            raise ValueError("Real image datasets require in_channels=3")
        normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        test_tf = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        )
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        if name in {"cifar10", "cifar100"}:
            cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
            train = (
                None
                if eval_only
                else cls(data_dir, train=True, download=download, transform=train_tf)
            )
            val = cls(data_dir, train=False, download=download, transform=test_tf)
        elif name == "tiny_imagenet":
            root = Path(data_dir)
            classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
            if len(classes) != 200:
                raise ValueError("Tiny ImageNet expects 200 class directories in train/")
            train = None if eval_only else datasets.ImageFolder(root / "train", transform=train_tf)
            if (root / "val" / "val_annotations.txt").exists():
                val = TinyImageNetValidation(root / "val", classes, test_tf)
            else:
                val = datasets.ImageFolder(root / "val", transform=test_tf)
                if val.classes != classes:
                    raise ValueError("Train and validation class folders must match")
        else:
            raise ValueError("Supported datasets: cifar10, cifar100, tiny_imagenet, synthetic")
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = (
        DataLoader(train, shuffle=True, generator=generator, **kwargs)
        if train is not None
        else None
    )
    val_loader = DataLoader(
        val, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **kwargs
    )
    return train_loader, val_loader


def loaders_from_config(cfg, eval_only=False):
    data = dict(cfg["data"])
    name = data.pop("dataset")
    return get_loaders(
        name=name,
        **data,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        image_size=cfg["model"]["image_size"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        seed=cfg["seed"],
        eval_only=eval_only,
    )
