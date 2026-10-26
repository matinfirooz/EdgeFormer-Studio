from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_tradeoff(rows, path="results/tradeoff.png", title=None):
    if not rows:
        raise ValueError("No design points")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bits = sorted({r["bits"] for r in rows}, reverse=True)
    palette = ["#334155", "#2563eb", "#0f766e", "#b45309", "#9333ea"]
    colors = {b: palette[i % len(palette)] for i, b in enumerate(bits)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for row in rows:
        color = colors[row["bits"]]
        marker = "o" if row.get("noise_scale", 0) == 0 else "X"
        for ax, x in zip(axes, [row["ideal_packed_total_bytes"] / 1024, row["latency_ms"]]):
            ax.scatter(x, 100 * row["accuracy"], color=color, marker=marker, s=55, alpha=0.85)
            if row.get("pareto_accuracy_storage"):
                ax.scatter(
                    x,
                    100 * row["accuracy"],
                    facecolors="none",
                    edgecolors="#2563eb",
                    s=145,
                    linewidths=1.5,
                )
    for ax in axes:
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(alpha=0.2)
    axes[0].set_xlabel("Ideal packed weights + scales (KiB)")
    axes[0].set_title("Accuracy / storage (ring: Pareto set)")
    axes[1].set_xlabel("Measured floating forward latency (ms/batch)")
    axes[1].set_title("Host runtime; precision is simulated")
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=colors[b], label=f"W{b}") for b in bits
        ],
        loc="lower right",
        fontsize=9,
    )
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker=m, linestyle="", color="#475569", label=label)
            for m, label in [("o", "No output noise"), ("X", "Statistical output noise")]
        ],
        loc="lower right",
        fontsize=9,
    )
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
