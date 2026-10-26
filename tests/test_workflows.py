import json
from pathlib import Path
import pytest
import torch
from torch import nn
from edgeformer.models import TinyViT
from edgeformer.profiling import estimate_vit_macs, storage_report, benchmark_latency
from edgeformer.quantization import quantize_model_weights, prepare_qat
from edgeformer.engine import evaluate
from edgeformer.analysis import output_distortion, attention_distortion, pareto_mask
from edgeformer.utils.config import load_config
from edgeformer.utils.dataset import get_loaders, TinyImageNetValidation
from edgeformer.utils.checkpoint import save_checkpoint, load_checkpoint
from edgeformer.utils.seed import capture_rng, restore_rng, set_seed
from edgeformer.cli import main


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0},
        {"num_heads": 0},
        {"patch_size": 0},
        {"embed_dim": 14},
        {"dropout": 1.0},
        {"image_size": 15},
    ],
)
def test_model_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TinyViT(**kwargs)


def test_model_rejects_wrong_image(small_model):
    with pytest.raises(ValueError, match="Expected"):
        small_model(torch.randn(1, 3, 32, 32))


def test_macs_against_forward_hooks_and_grayscale():
    model = TinyViT(
        image_size=8,
        patch_size=4,
        in_channels=1,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_classes=3,
    ).eval()
    observed = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            observed.append(output.numel() * module.weight[0].numel())
        else:
            observed.append(output.numel() * module.in_features)

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    model(torch.randn(1, 1, 8, 8))
    for h in handles:
        h.remove()
    report = estimate_vit_macs(model)
    assert report["total"] == sum(observed) + report["qk"] + report["av"]
    assert report["patch_embed"] == 4 * 16 * 8


def test_storage_reflects_float_biases_tokens_and_scales(small_model):
    fp = storage_report(small_model, 32)
    q = storage_report(small_model, 4)
    assert fp["runtime_parameter_bytes"] == q["runtime_parameter_bytes"]
    assert fp["ideal_packed_total_bytes"] == fp["runtime_parameter_bytes"]
    assert q["scale_bytes"] > 0 and q["ideal_packed_total_bytes"] < fp["ideal_packed_total_bytes"]
    assert q["ideal_packed_total_bytes"] > fp["runtime_parameter_bytes"] / 8


def test_benchmark_restores_heterogeneous_modes_and_rng(small_model):
    small_model.train()
    small_model.blocks[0].attn.eval()
    modes = [m.training for m in small_model.modules()]
    sample = torch.randn(2, 3, 16, 16)
    rng = torch.get_rng_state().clone()
    stats = benchmark_latency(small_model, sample, 1, 3, return_stats=True)
    assert [m.training for m in small_model.modules()] == modes
    assert torch.equal(rng, torch.get_rng_state())
    assert stats["p95_ms"] >= stats["median_ms"] > 0
    assert stats["images_per_second"] > 0
    with pytest.raises(ValueError):
        benchmark_latency(small_model, sample, runs=0)


def test_evaluation_is_sample_weighted():
    class Identity(nn.Module):
        def forward(self, x):
            return x

    x = torch.tensor([[3.0, -2.0], [3.0, -2.0], [-2.0, 3.0]])
    y = torch.tensor([0, 0, 0])
    model = Identity().train()
    result = evaluate(model, [(x[:2], y[:2]), (x[2:], y[2:])])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["loss"] == pytest.approx(nn.functional.cross_entropy(x, y).item())
    assert model.training
    with pytest.raises(ValueError):
        evaluate(model, [])


@pytest.mark.parametrize("kind", ["fp32", "ptq", "qat", "legacy"])
def test_checkpoint_numerics_roundtrip(tmp_path, small_model, kind):
    cfg = load_config()
    cfg["model"] = small_model.config
    model = small_model
    if kind == "ptq":
        model = quantize_model_weights(model, 4)
    if kind == "qat":
        model = prepare_qat(model, 4)
    path = tmp_path / "model.pt"
    if kind == "legacy":
        torch.save({"model": model.state_dict(), "config": cfg}, path)
    else:
        save_checkpoint(model, cfg, path)
    restored, state = load_checkpoint(path)
    x = torch.randn(2, 3, 16, 16)
    torch.testing.assert_close(model.eval()(x), restored.eval()(x), rtol=0, atol=0)
    assert state["config"]["model"] == cfg["model"]


def test_rng_roundtrip():
    import random
    import numpy as np

    set_seed(5)
    state = capture_rng()
    reference = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng(state)
    assert random.random() == reference[0]
    assert float(np.random.rand()) == reference[1]
    assert torch.equal(torch.rand(3), reference[2])


def test_synthetic_loaders_reproducible_and_no_eval_train_access():
    kw = dict(
        name="synthetic", batch_size=3, image_size=8, num_classes=3, train_samples=9, val_samples=5
    )
    a, av = get_loaders(**kw)
    b, bv = get_loaders(**kw)
    xa, ya = next(iter(a))
    xb, yb = next(iter(b))
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert len(av.dataset) == len(bv.dataset) == 5
    no_train, _ = get_loaders(**kw, eval_only=True)
    assert no_train is None
    with pytest.raises(ValueError):
        get_loaders(name="cifar10", num_classes=100)


def test_original_tiny_imagenet_annotation_mapping(tmp_path):
    from PIL import Image
    from torchvision.transforms import ToTensor

    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "a.JPEG")
    (tmp_path / "val_annotations.txt").write_text("a.JPEG n2 0 0 4 4\n")
    dataset = TinyImageNetValidation(tmp_path, ["n1", "n2"], ToTensor())
    image, label = dataset[0]
    assert label == 1 and image.shape == (3, 4, 4)


def test_distortion_and_pareto_duplicates():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert output_distortion(x, x)["logit_rmse"] == 0
    m = [torch.softmax(torch.randn(1, 2, 4, 4), -1)]
    d = attention_distortion(m, m)
    assert d["attention_mae"] == 0 and d["attention_js_divergence"] == 0
    rows = [
        {"accuracy": 0.8, "ideal_packed_total_bytes": 8},
        {"accuracy": 0.7, "ideal_packed_total_bytes": 8},
        {"accuracy": 0.6, "ideal_packed_total_bytes": 4},
        {"accuracy": 0.8, "ideal_packed_total_bytes": 8},
    ]
    assert pareto_mask(rows) == [True, False, True, True]
    with pytest.raises(ValueError):
        pareto_mask([{"accuracy": float("nan"), "ideal_packed_total_bytes": 4}])


def test_cli_end_to_end(tmp_path):
    smoke = str(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    run = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--config",
                smoke,
                "--epochs",
                "1",
                "--max-batches",
                "1",
                "--output",
                str(run),
            ]
        )
        == 0
    )
    checkpoint = str(run / "best.pt")
    evaluation = tmp_path / "evaluation.json"
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--bits",
                "4",
                "--max-batches",
                "1",
                "--output",
                str(evaluation),
            ]
        )
        == 0
    )
    assert json.loads(evaluation.read_text())["synthetic"]
    assert (
        main(
            [
                "quantize",
                "--checkpoint",
                checkpoint,
                "--bits",
                "4",
                "--output",
                str(tmp_path / "q.pt"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(tmp_path / "q.pt"),
                "--runs",
                "2",
                "--warmup",
                "0",
                "--output",
                str(tmp_path / "bench.json"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "bench.json").read_text())["transform"]["bits"] == 4
    assert (
        main(
            [
                "sweep",
                "--checkpoint",
                checkpoint,
                "--bits",
                "32",
                "4",
                "--noise-scales",
                "0",
                "--max-batches",
                "1",
                "--runs",
                "1",
                "--warmup",
                "0",
                "--output",
                str(tmp_path / "sweep.csv"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "sensitivity",
                "--checkpoint",
                checkpoint,
                "--max-batches",
                "1",
                "--output",
                str(tmp_path / "sensitivity.csv"),
            ]
        )
        == 0
    )
    assert (
        main(["attention", "--checkpoint", checkpoint, "--output", str(tmp_path / "attention.png")])
        == 0
    )
    assert main(["evaluate", "--checkpoint", checkpoint, "--bits", "1"]) == 2


def test_cli_qat_checkpoint_inference_and_resume(tmp_path):
    smoke = str(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    run = tmp_path / "qat"
    assert (
        main(
            [
                "train",
                "--config",
                smoke,
                "--qat-bits",
                "4",
                "--epochs",
                "1",
                "--max-batches",
                "1",
                "--output",
                str(run),
            ]
        )
        == 0
    )
    checkpoint = str(run / "latest.pt")
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--max-batches",
                "1",
                "--output",
                str(tmp_path / "qat.json"),
            ]
        )
        == 0
    )
    assert main(["train", "--resume", checkpoint, "--max-batches", "1", "--output", str(run)]) == 0
    assert main(["train", "--resume", checkpoint, "--epochs", "2", "--output", str(run)]) == 2


@pytest.mark.parametrize("qat_bits", [None, 4])
def test_epoch_resume_matches_continuous_training(tmp_path, monkeypatch, qat_bits):
    import edgeformer.cli as cli

    smoke = str(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    base = ["train", "--config", smoke, "--max-batches", "2", "--deterministic"]
    if qat_bits is not None:
        base += ["--qat-bits", str(qat_bits)]
    assert main([*base, "--output", str(full)]) == 0
    original = cli.train_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated interruption after a completed epoch")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "train_epoch", interrupted)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        main([*base, "--output", str(resumed)])
    monkeypatch.setattr(cli, "train_epoch", original)
    assert (
        main(
            [
                "train",
                "--resume",
                str(resumed / "latest.pt"),
                "--max-batches",
                "2",
                "--output",
                str(resumed),
            ]
        )
        == 0
    )
    a = torch.load(full / "latest.pt", weights_only=True)
    b = torch.load(resumed / "latest.pt", weights_only=True)
    assert a["history"] == b["history"]
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
