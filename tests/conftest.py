import pytest
import torch
from edgeformer.models import TinyViT


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def small_model():
    torch.manual_seed(7)
    return TinyViT(
        image_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
