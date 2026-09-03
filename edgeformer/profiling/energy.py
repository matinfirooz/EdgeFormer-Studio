def estimate_energy_mj(macs: int, bits: int = 8, pj_per_mac_int8: float = 0.25) -> float:
    """Simple first-order compute-energy proxy.

    Default coefficient is deliberately configurable and is NOT a silicon
    measurement. Replace with numbers from your target technology/tool flow.
    """
    scale = (bits / 8) ** 2
    return macs * pj_per_mac_int8 * scale / 1e9
