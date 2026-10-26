"""Optional local Gradio inference. Requires an explicitly selected checkpoint."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.utils.checkpoint import load_checkpoint

CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def preprocess(image: Image.Image, size=32, channels=3):
    if image is None:
        raise ValueError("Choose an image first")
    if channels not in {1, 3}:
        raise ValueError("Demo supports one- or three-channel images")
    image = image.convert("RGB" if channels == 3 else "L").resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1
    if channels == 1:
        arr = arr[..., None]
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)


@lru_cache(maxsize=2)
def _cached_load(path, mtime_ns):
    model, state = load_checkpoint(path)
    model._edgeformer_dataset = state["config"].get("data", {}).get("dataset")
    return model.eval()


def load(checkpoint):
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first.")
    return _cached_load(str(path), path.stat().st_mtime_ns)


def predict(image, checkpoint):
    model = load(checkpoint)
    with torch.inference_mode():
        probs = model(preprocess(image, model.image_size, model.in_channels)).softmax(-1)[0]
    values, indices = probs.topk(min(5, probs.numel()))
    # Label only known CIFAR10 checkpoints; avoid relabeling arbitrary 10-class models.
    dataset = model._edgeformer_dataset
    labels = CIFAR10 if dataset == "cifar10" else [f"class {i}" for i in range(probs.numel())]
    return {labels[int(i)]: float(v) for v, i in zip(values, indices)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    load(args.checkpoint)
    import gradio as gr

    with gr.Blocks(title="EdgeFormer Studio") as demo:
        gr.Markdown("# EdgeFormer Studio\nLocal checkpoint inference. Confidence is a model score.")
        image = gr.Image(type="pil", label="Image")
        output = gr.Label(num_top_classes=5)
        gr.Button("Predict").click(lambda img: predict(img, args.checkpoint), [image], [output])
    demo.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
