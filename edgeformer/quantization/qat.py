from torch import nn
from .fake_quant import fake_quantize_tensor


class FakeQuantLinear(nn.Linear):
    """Training-friendly fake-quantized linear layer."""
    def __init__(self, *args, bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits

    def forward(self, x):
        qx = fake_quantize_tensor(x, self.bits)
        qw = fake_quantize_tensor(self.weight, self.bits)
        return nn.functional.linear(qx, qw, self.bias)
