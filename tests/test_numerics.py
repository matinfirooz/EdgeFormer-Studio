import copy
import pytest
import torch
from torch import nn
from edgeformer.quantization import fake_quantize_tensor, quantize_model_weights, prepare_qat
from edgeformer.quantization.qat import FakeQuantLinear, FakeQuantConv2d
from edgeformer.approximate import replace_linear_with_approx
from edgeformer.models.attention import MultiHeadSelfAttention


@pytest.mark.parametrize("bits", [2, 4, 6, 8, 16, 32])
def test_quantization_zero_identity_and_error_bound(bits):
    x = torch.linspace(-1, 1, 97)
    q = fake_quantize_tensor(x, bits)
    assert torch.equal(fake_quantize_tensor(torch.zeros(3), bits), torch.zeros(3))
    if bits == 32:
        assert torch.equal(x, q) and q.data_ptr() != x.data_ptr()
    else:
        assert (q - x).abs().max() <= 0.5 / (2 ** (bits - 1) - 1) + 1e-6
        assert q.min() >= -1 and q.max() <= 1


@pytest.mark.parametrize("bits", [0, 1, 17, 31, 33, True, 4.5])
def test_invalid_bits(bits):
    with pytest.raises(ValueError):
        fake_quantize_tensor(torch.ones(3), bits)


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.empty(0),
        torch.ones(2, dtype=torch.int64),
    ],
)
def test_invalid_quantization_input(x):
    with pytest.raises(ValueError):
        fake_quantize_tensor(x, 8)


def test_per_channel_preserves_small_channels():
    x = torch.tensor([[1.0, 0.45, -0.3], [0.01, 0.0045, -0.003]])
    per_tensor = fake_quantize_tensor(x, 4)
    per_channel = fake_quantize_tensor(x, 4, per_channel=True)
    assert (per_channel[1] - x[1]).square().sum() < (per_tensor[1] - x[1]).square().sum()
    torch.testing.assert_close(per_channel[1], per_channel[0] * 0.01)


def test_ste_and_qat_have_useful_gradients(small_model):
    x = torch.randn(4, 8, requires_grad=True)
    fake_quantize_tensor(x, 4, ste=True).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    model = prepare_qat(small_model, 4)
    model(torch.randn(3, 3, 16, 16)).square().sum().backward()
    assert isinstance(model.patch_embed, FakeQuantConv2d)
    for layer in model.modules():
        if isinstance(layer, (FakeQuantLinear, FakeQuantConv2d)):
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0


def test_transforms_preserve_mode_dtype_and_source(small_model):
    model = small_model.double().eval()
    before = copy.deepcopy(model.state_dict())
    for result in [
        prepare_qat(model, 4),
        quantize_model_weights(model, 4),
        replace_linear_with_approx(model, 0.03),
    ]:
        assert all(not m.training for m in result.modules())
        assert all(p.dtype == torch.float64 for p in result.parameters())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_approx_noise_is_inference_only_and_seeded():
    layer = nn.Linear(8, 4).double().eval()
    x = torch.ones(2, 8, dtype=torch.float64)
    exact = replace_linear_with_approx(layer, 0.0)
    torch.testing.assert_close(exact(x), layer(x), rtol=0, atol=0)
    noisy = replace_linear_with_approx(layer, 0.3)
    torch.manual_seed(5)
    first = noisy(x)
    torch.manual_seed(5)
    second = noisy(x)
    assert torch.equal(first, second) and not torch.equal(first, layer(x))
    torch.testing.assert_close(noisy.train()(x), layer(x))


def test_selected_layers_and_mixed_precision(small_model):
    q = quantize_model_weights(small_model, 32, layer_bits={"head": 4})
    assert torch.equal(q.patch_embed.weight, small_model.patch_embed.weight)
    assert not torch.equal(q.head.weight, small_model.head.weight)
    assert torch.equal(q.head.bias, small_model.head.bias)
    with pytest.raises(ValueError):
        quantize_model_weights(small_model, 4, layer_bits={"typo": 8})
    with pytest.raises(ValueError):
        replace_linear_with_approx(small_model, layers={"typo"})
    with pytest.raises(ValueError):
        replace_linear_with_approx(prepare_qat(small_model, 4))


def test_sdpa_manual_output_and_gradient_parity():
    manual = MultiHeadSelfAttention(16, 4, backend="manual").double().eval()
    sdpa = MultiHeadSelfAttention(16, 4, backend="sdpa").double().eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    y = manual(x)
    z = sdpa(other)
    torch.testing.assert_close(y, z, atol=1e-10, rtol=1e-10)
    y.square().sum().backward()
    z.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad, atol=1e-10, rtol=1e-10)


def test_maps_normalized_before_dropout_and_eval_deterministic():
    layer = MultiHeadSelfAttention(16, 4, dropout=0.5)
    x = torch.randn(1, 7, 16)
    _, maps = layer(x, return_attention=True)
    torch.testing.assert_close(maps.sum(-1), torch.ones(1, 4, 7))
    layer.eval()
    assert torch.equal(layer(x), layer(x))
