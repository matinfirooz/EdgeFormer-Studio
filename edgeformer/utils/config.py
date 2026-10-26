"""Small explicit configuration loader shared by every command."""

import copy
from pathlib import Path
import yaml
from edgeformer.models import TinyViT

DEFAULT = dict(
    seed=42,
    model={},
    data=dict(
        dataset="cifar10", data_dir="data", download=False, train_samples=128, val_samples=64
    ),
    training=dict(
        epochs=20,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=0.05,
        num_workers=0,
        grad_clip=1.0,
        label_smoothing=0.0,
    ),
    quantization=dict(bits=8),
    approximation=dict(enabled=False, noise_scale=0.0),
)


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULT)
    if path:
        with Path(path).open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError("configuration must be a mapping")
        unknown = set(loaded) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        for key, value in loaded.items():
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a mapping")
                if key != "model" and set(value) - set(cfg[key]):
                    raise ValueError(f"Unknown {key} options: {sorted(set(value) - set(cfg[key]))}")
                cfg[key].update(value)
            else:
                cfg[key] = value
    # Resolve default model fields without constructing a model or consuming RNG.
    import inspect

    defaults = {n: p.default for n, p in inspect.signature(TinyViT).parameters.items()}
    if set(cfg["model"]) - set(defaults):
        raise ValueError("Unknown model configuration options")
    cfg["model"] = {**defaults, **cfg["model"]}
    return cfg


def validate_training_config(cfg):
    t = cfg["training"]
    for key in ["epochs", "batch_size"]:
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if t["num_workers"] < 0 or t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid workers, learning rate or weight decay")
    if t["grad_clip"] < 0 or not 0 <= t["label_smoothing"] < 1:
        raise ValueError("Invalid gradient clip or label smoothing")
