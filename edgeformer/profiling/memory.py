"""Runtime tensors and hypothetical packed weights are separate quantities."""

from torch import nn
from edgeformer.quantization.fake_quant import validate_bits


def parameter_memory_mb(model, bits_per_parameter=32):
    """Legacy ideal UNIFORM parameter storage in MiB; excludes scales/buffers."""
    validate_bits(bits_per_parameter)
    return sum(p.numel() for p in model.parameters()) * bits_per_parameter / 8 / 2**20


def storage_report(model, bits=32, *, per_channel=True, layer_bits=None):
    validate_bits(bits)
    plan = dict(layer_bits or {})
    eligible = {
        f"{n}.weight" if n else "weight": (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    }
    valid_names = {n for n, _ in eligible.values()}
    if set(plan) - valid_names:
        raise ValueError(f"Unknown layers in storage plan: {sorted(set(plan) - valid_names)}")
    for b in plan.values():
        validate_bits(b)
    payload_bits = scales = quantized = 0
    for name, p in model.named_parameters():
        width = p.element_size() * 8
        if name in eligible:
            layer_name, module = eligible[name]
            width = plan.get(layer_name, bits)
            if width < 32:
                quantized += p.numel()
                scales += module.weight.shape[0] if per_channel else 1
        payload_bits += p.numel() * width
    return dict(
        parameters=sum(p.numel() for p in model.parameters()),
        quantized_weight_parameters=quantized,
        runtime_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        runtime_buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
        ideal_packed_parameter_bytes=(payload_bits + 7) // 8,
        scale_bytes=scales * 4,
        ideal_packed_total_bytes=(payload_bits + 7) // 8 + scales * 4,
        storage_note="Ideal contiguous bit packing with FP32 scales; runtime remains floating point.",
    )


def attention_storage_report(model, batch_size=1, element_bytes=4):
    if batch_size < 1 or element_bytes < 1:
        raise ValueError("batch_size and element_bytes must be positive")
    n = model.num_patches + 1
    h = model.blocks[0].attn.num_heads
    matrix = batch_size * h * n * n * element_bytes
    return dict(
        tokens=n,
        heads=h,
        single_attention_matrix_bytes=matrix,
        score_plus_probability_bytes=2 * matrix,
        row_score_plus_probability_bytes=2 * batch_size * h * n * element_bytes,
        note="Analytical score/probability storage only, excluding Q/K/V, training and backend workspace.",
    )
