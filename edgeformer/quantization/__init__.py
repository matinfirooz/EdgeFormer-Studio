from .fake_quant import fake_quantize_tensor
from .int8 import quantize_model_weights

__all__ = ["fake_quantize_tensor", "quantize_model_weights"]
