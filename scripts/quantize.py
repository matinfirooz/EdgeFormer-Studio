import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT
from edgeformer.quantization import quantize_model_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--precision", choices=["int8", "int4"], default="int8")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    bits = 8 if args.precision == "int8" else 4
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TinyViT(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    qmodel = quantize_model_weights(model, bits)
    output = args.output or f"checkpoints/vit_fake_int{bits}.pt"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": qmodel.state_dict(), "config": ckpt["config"], "bits": bits}, output)
    print(f"Saved fake-quantized checkpoint: {output}")


if __name__ == "__main__":
    main()
