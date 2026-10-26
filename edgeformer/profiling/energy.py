"""Uncalibrated analytical compute-cost proxy, never measured device energy."""

import math
from edgeformer.quantization.fake_quant import validate_bits


def estimate_energy_mj(macs, bits=8, pj_per_mac_int8=0.25):
    validate_bits(bits)
    if (
        not math.isfinite(macs)
        or macs < 0
        or not math.isfinite(pj_per_mac_int8)
        or pj_per_mac_int8 < 0
    ):
        raise ValueError("MAC count and coefficient must be finite and nonnegative")
    return macs * pj_per_mac_int8 * (bits / 8) ** 2 / 1e9
