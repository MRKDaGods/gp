from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent


def ensure_output_dir() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_association_saturation() -> None:
    """Create a placeholder scatter showing saturated association tuning."""
    # TODO: Replace this synthetic series with the full 10c v14-v53 experiment history.
    rng = np.random.default_rng(42)
    config_ids = np.arange(1, 226)
    scores = 77.55 + rng.normal(0.0, 0.07, size=config_ids.size)
    scores = np.clip(scores, 77.35, 77.65)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(config_ids, scores, s=18, alpha=0.75, color="#1f77b4", edgecolors="none")
    ax.axhspan(77.35, 77.65, color="#ffcc80", alpha=0.35, label="0.3pp saturation band")
    ax.axhline(77.5, color="#d62728", linestyle="--", linewidth=1.5, label="Current reproducible best")
    ax.scatter(config_ids[best_idx], best_score, color="#111111", s=45, zorder=3)
    ax.annotate(
        f"Best placeholder config\nIDF1={best_score:.2f}%",
        xy=(config_ids[best_idx], best_score),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title("Association Tuning Saturation (Placeholder Data)")
    ax.set_xlabel("Configuration index")
    ax.set_ylabel("MTMC IDF1 (%)")
    ax.set_ylim(77.2, 77.8)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "association_saturation.png", dpi=200)
    plt.close(fig)


def generate_map_vs_mtmc() -> None:
    """Create a placeholder scatter for the mAP versus MTMC mismatch."""
    # TODO: Replace with exact points and citations from the verified experiment log.
    points = [
        ("Baseline ViT 256px", 80.14, 77.5, "#1f77b4"),
        ("Augoverhaul", 81.59, 72.2, "#d62728"),
        ("DMT", 87.30, 75.8, "#2ca02c"),
        ("384px deployment", 80.00, 75.6, "#9467bd"),
        ("Weak ResNet fusion", 52.77, 77.4, "#8c564b"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label, map_score, mtmc_idf1, color in points:
        ax.scatter(map_score, mtmc_idf1, s=70, color=color)
        ax.annotate(label, (map_score, mtmc_idf1), textcoords="offset points", xytext=(6, 6), fontsize=9)

    x_vals = np.array([point[1] for point in points])
    y_vals = np.array([point[2] for point in points])
    coeffs = np.polyfit(x_vals, y_vals, deg=1)
    x_line = np.linspace(x_vals.min() - 2, x_vals.max() + 2, 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, linestyle="--", color="#444444", linewidth=1.2, label="Placeholder trend")

    ax.set_title("Single-Camera mAP vs. MTMC IDF1 (Placeholder Data)")
    ax.set_xlabel("ReID mAP (%)")
    ax.set_ylabel("MTMC IDF1 (%)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "map_vs_mtmc.png", dpi=200)
    plt.close(fig)


def generate_dead_end_waterfall() -> None:
    """Create a placeholder sorted bar chart for dead-end impacts."""
    # TODO: Replace with the final curated dead-end table once exact values are frozen.
    methods = [
        ("CSLS", -34.7),
        ("SAM2 masking", -8.7),
        ("Augoverhaul + CircleLoss", -5.3),
        ("mtmc_only=true", -5.0),
        ("AFLink", -3.8),
        ("Global optimal tracker", -3.5),
        ("CID_BIAS", -3.3),
        ("384px deployment", -2.8),
        ("Denoise", -2.7),
        ("FAC", -2.5),
        ("DMT", -1.4),
        ("Network flow", -0.24),
    ]
    methods.sort(key=lambda item: item[1])

    labels = [item[0] for item in methods]
    deltas = [item[1] for item in methods]
    colors = ["#c62828" if delta < -5 else "#ef6c00" if delta < -2 else "#f9a825" for delta in deltas]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(range(len(labels)), deltas, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MTMC IDF1 delta (pp)")
    ax.set_title("Dead-End Impact Summary (Placeholder Data)")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    for bar, delta in zip(bars, deltas):
        ax.annotate(
            f"{delta:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, delta),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#111111",
        )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dead_end_waterfall.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    generate_association_saturation()
    generate_map_vs_mtmc()
    generate_dead_end_waterfall()


if __name__ == "__main__":
    main()