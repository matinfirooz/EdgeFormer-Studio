from .fake_quant import fake_quantize_tensor
from .int8 import quantize_model_weights
from .qat import FakeQuantLinear, FakeQuantConv2d, prepare_qat

__all__ = [
    "fake_quantize_tensor",
    "quantize_model_weights",
    "FakeQuantLinear",
    "FakeQuantConv2d",
    "prepare_qat",
]
