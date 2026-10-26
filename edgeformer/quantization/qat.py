"""Dynamic activation / per-channel weight QAT simulation for Linear and Conv2d."""

import copy
from torch import nn
from torch.nn import functional as F
from .fake_quant import fake_quantize_tensor, validate_bits


class FakeQuantLinear(nn.Linear):
    def __init__(self, *args, bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        validate_bits(bits)
        self.bits = bits

    def forward(self, x):
        qx = fake_quantize_tensor(x, self.bits, ste=True)
        qw = fake_quantize_tensor(self.weight, self.bits, per_channel=True, ste=True)
        return F.linear(qx, qw, self.bias)


class FakeQuantConv2d(nn.Conv2d):
    def __init__(self, *args, bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        validate_bits(bits)
        self.bits = bits

    def forward(self, x):
        qx = fake_quantize_tensor(x, self.bits, ste=True)
        qw = fake_quantize_tensor(self.weight, self.bits, per_channel=True, ste=True)
        return self._conv_forward(qx, qw, self.bias)


def prepare_qat(model, bits=8):
    """Copy a floating model; keep parameter values, device, dtype and mode."""
    validate_bits(bits)
    result = copy.deepcopy(model)

    def convert(module):
        if type(module) is nn.Linear:
            out = FakeQuantLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                bits=bits,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        elif type(module) is nn.Conv2d:
            out = FakeQuantConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                bits=bits,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            raise ValueError("QAT expects floating Linear/Conv2d modules")
        else:
            for name, child in list(module.named_children()):
                setattr(module, name, convert(child))
            return module
        out.load_state_dict(module.state_dict())
        out.weight.requires_grad_(module.weight.requires_grad)
        if out.bias is not None:
            out.bias.requires_grad_(module.bias.requires_grad)
        return out.train(module.training)

    result = convert(result)
    result._edgeformer_transform = {"kind": "qat", "bits": bits}
    return result
