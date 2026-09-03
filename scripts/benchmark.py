import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT
from edgeformer.profiling.macs import estimate_vit_macs
from edgeformer.profiling.memory import parameter_memory_mb
from edgeformer.profiling.energy import estimate_energy_mj
from edgeformer.profiling.latency import benchmark_latency
from edgeformer.visualization.dashboard import print_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--bits", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = TinyViT(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
    else:
        model = TinyViT()
    model = model.to(device)
    macs = estimate_vit_macs(model)
    sample = torch.randn(args.batch_size, 3, model.image_size, model.image_size, device=device)
    latency = benchmark_latency(model, sample, runs=args.runs)
    report = {
        "Device": str(device),
        "Parameters": f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M",
        "Estimated MACs": f"{macs['total']/1e6:.2f} M",
        "Attention MACs": f"{macs['attention']/1e6:.2f} M",
        "MLP MACs": f"{macs['mlp']/1e6:.2f} M",
        "Parameter Memory": f"{parameter_memory_mb(model, args.bits):.2f} MB @ {args.bits}-bit",
        "Measured Latency": f"{latency:.3f} ms/batch",
        "Energy Proxy": f"{estimate_energy_mj(macs['total'], args.bits):.5f} mJ",
    }
    print_report(report)


if __name__ == "__main__":
    main()
