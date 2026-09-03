from pathlib import Path
import matplotlib.pyplot as plt


def save_tradeoff(rows, path="results/tradeoff.png"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = [r["latency_ms"] for r in rows]
    y = [r["accuracy"] for r in rows]
    labels = [r["name"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy–Latency Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
