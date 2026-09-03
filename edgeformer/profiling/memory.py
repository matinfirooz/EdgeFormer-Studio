def parameter_memory_mb(model, bits_per_parameter: int = 32) -> float:
    params = sum(p.numel() for p in model.parameters())
    return params * bits_per_parameter / 8 / (1024 ** 2)
