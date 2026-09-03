import copy
import torch
from torch import nn


class ApproxLinear(nn.Linear):
    """Linear layer with controllable output-error injection.

    The model is useful for sensitivity sweeps before mapping to a concrete
    approximate arithmetic circuit.
    """
    def __init__(self, in_features, out_features, bias=True, noise_scale=0.01):
        super().__init__(in_features, out_features, bias=bias)
        self.noise_scale = float(noise_scale)

    def forward(self, x):
        out = nn.functional.linear(x, self.weight, self.bias)
        if not self.training and self.noise_scale > 0:
            sigma = out.detach().abs().mean().clamp_min(1e-8) * self.noise_scale
            out = out + torch.randn_like(out) * sigma
        return out


def replace_linear_with_approx(model: nn.Module, noise_scale: float = 0.01):
    model = copy.deepcopy(model)
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            repl = ApproxLinear(child.in_features, child.out_features,
                                child.bias is not None, noise_scale)
            repl.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                repl.bias.data.copy_(child.bias.data)
            setattr(model, name, repl)
        else:
            setattr(model, name, replace_linear_with_approx(child, noise_scale))
    return model
