from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loaders(name="cifar10", batch_size=128, num_workers=2, data_dir="./data"):
    name = name.lower()
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if name == "cifar10":
        cls = datasets.CIFAR10
    elif name == "cifar100":
        cls = datasets.CIFAR100
    else:
        raise ValueError("Supported datasets: cifar10, cifar100")
    train_ds = cls(data_dir, train=True, download=True, transform=train_tf)
    test_ds = cls(data_dir, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
