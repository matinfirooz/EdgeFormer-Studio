import argparse
from pathlib import Path
import sys
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT
from edgeformer.quantization import quantize_model_weights
from edgeformer.approximate import replace_linear_with_approx
from edgeformer.profiling.latency import benchmark_latency
from edgeformer.profiling.macs import estimate_vit_macs
from edgeformer.profiling.memory import parameter_memory_mb
from edgeformer.profiling.energy import estimate_energy_mj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default="results/design_comparison.csv")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        base = TinyViT(**ckpt["config"]["model"]); base.load_state_dict(ckpt["model"])
    else:
        base = TinyViT()
    variants = [
        ("FP32", base, 32),
        ("INT8-sim", quantize_model_weights(base, 8), 8),
        ("INT8+Approx", replace_linear_with_approx(quantize_model_weights(base, 8), 0.01), 8),
    ]
    rows = []
    for name, model, bits in variants:
        model = model.to(device).eval()
        sample = torch.randn(1, 3, model.image_size, model.image_size, device=device)
        macs = estimate_vit_macs(model)["total"]
        rows.append({
            "name": name,
            "bits": bits,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "macs_m": macs / 1e6,
            "memory_mb": parameter_memory_mb(model, bits),
            "latency_ms": benchmark_latency(model, sample, warmup=3, runs=10),
            "energy_proxy_mj": estimate_energy_mj(macs, bits),
        })
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
