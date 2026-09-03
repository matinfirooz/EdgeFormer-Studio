from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.models import TinyViT

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def preprocess(image: Image.Image):
    image = image.convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load(checkpoint):
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        model = TinyViT(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
        return model.eval()
    return TinyViT().eval()


def predict(image, checkpoint="checkpoints/vit_cifar10.pt"):
    model = load(checkpoint)
    with torch.no_grad():
        probs = model(preprocess(image)).softmax(-1)[0]
    values, indices = probs.topk(5)
    return {CLASSES[i] if i < len(CLASSES) else str(i): float(v) for v, i in zip(values, indices)}


def main():
    import gradio as gr
    with gr.Blocks(title="EdgeFormer Studio") as demo:
        gr.Markdown("# ⚡ EdgeFormer Studio Demo\nUpload a CIFAR-like image and inspect TinyViT predictions.")
        image = gr.Image(type="pil", label="Input image")
        checkpoint = gr.Textbox(value="checkpoints/vit_cifar10.pt", label="Checkpoint")
        output = gr.Label(num_top_classes=5)
        gr.Button("Run inference").click(predict, [image, checkpoint], output)
    demo.launch()


if __name__ == "__main__":
    main()
