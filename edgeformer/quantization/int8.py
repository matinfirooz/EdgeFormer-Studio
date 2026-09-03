import copy
from torch import nn
from .fake_quant import fake_quantize_tensor


def quantize_model_weights(model: nn.Module, bits: int = 8) -> nn.Module:
    """Return a copy with Linear/Conv2d weights fake-quantized to N-bit precision.

    This is intentionally framework-independent research PTQ simulation rather
    than backend-specific integer kernel conversion.
    """
    qmodel = copy.deepcopy(model)
    for module in qmodel.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.copy_(fake_quantize_tensor(module.weight.data, bits))
            if module.bias is not None:
                module.bias.data.copy_(fake_quantize_tensor(module.bias.data, bits))
    return qmodel
