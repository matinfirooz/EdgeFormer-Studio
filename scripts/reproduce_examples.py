"""Run the complete offline workflow and regenerate bundled smoke evidence."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="examples/results")
    parser.add_argument("--work-dir", default="runs/example")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    work = (ROOT / args.work_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run(*command):
        print("Running:", " ".join(command), flush=True)
        subprocess.run([sys.executable, "-m", "edgeformer", *command], cwd=ROOT, check=True)

    run(
        "train",
        "--config",
        "configs/smoke.yaml",
        "--deterministic",
        "--device",
        "cpu",
        "--output",
        str(work / "fp32"),
    )
    reference = str(work / "fp32" / "best.pt")
    run(
        "evaluate",
        "--checkpoint",
        reference,
        "--device",
        "cpu",
        "--output",
        str(output / "evaluation.json"),
    )
    run(
        "quantize",
        "--checkpoint",
        reference,
        "--bits",
        "4",
        "--device",
        "cpu",
        "--output",
        str(work / "ptq_w4.pt"),
    )
    for bits in [32, 8, 4]:
        run(
            "benchmark",
            "--checkpoint",
            reference,
            "--bits",
            str(bits),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--runs",
            "20",
            "--warmup",
            "5",
            "--output",
            str(output / f"benchmark_w{bits}.json"),
        )
    run(
        "sweep",
        "--checkpoint",
        reference,
        "--bits",
        "32",
        "8",
        "6",
        "4",
        "--noise-scales",
        "0",
        ".01",
        "--repeats",
        "2",
        "--device",
        "cpu",
        "--runs",
        "10",
        "--warmup",
        "3",
        "--output",
        str(output / "sweep.csv"),
    )
    run(
        "sensitivity",
        "--checkpoint",
        reference,
        "--bits",
        "4",
        "--device",
        "cpu",
        "--max-batches",
        "2",
        "--output",
        str(output / "sensitivity.csv"),
    )
    run(
        "attention",
        "--checkpoint",
        reference,
        "--device",
        "cpu",
        "--output",
        str(output / "attention.png"),
    )
    run(
        "train",
        "--checkpoint",
        reference,
        "--qat-bits",
        "4",
        "--epochs",
        "3",
        "--deterministic",
        "--device",
        "cpu",
        "--output",
        str(work / "qat_w4"),
    )
    run(
        "evaluate",
        "--checkpoint",
        str(work / "qat_w4" / "best.pt"),
        "--device",
        "cpu",
        "--output",
        str(output / "evaluation_qat_w4.json"),
    )
    for variant in ["fp32", "qat_w4"]:
        for name in ["history.csv", "run.json"]:
            shutil.copyfile(work / variant / name, output / f"{variant}_{name}")
    make_plots(output)
    print("Offline evidence saved:", output)


def make_plots(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "fp32_history.csv").open() as f:
        history = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [int(r["epoch"]) for r in history]
    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(epochs, [float(r[key]) for r in history], "o-", label=label)
    for key, label in [("train_accuracy", "Train"), ("val_accuracy", "Validation")]:
        axes[1].plot(epochs, [100 * float(r[key]) for r in history], "o-", label=label)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Synthetic patterns only | offline training smoke test")
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=170)
    plt.close(fig)
    with (output / "sensitivity.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = list(reversed(rows))
    ax.barh([r["layer"] for r in rows], [float(r["logit_rmse"]) for r in rows], color="#2563eb")
    ax.set_xlabel("Logit RMSE vs floating reference")
    ax.set_title("One-layer W4 PTQ | synthetic validation images")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "sensitivity.png", dpi=170)
    plt.close(fig)
    manifest = {
        "dataset": "synthetic",
        "precision_execution": "floating simulation",
        "warning": "These runs verify software behavior; they are not CIFAR/ImageNet accuracy or ASIC PPA results.",
        "reproduce": "python scripts/reproduce_examples.py",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
