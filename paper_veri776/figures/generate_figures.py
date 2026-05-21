"""
paper_veri776/figures/generate_figures.py
Generates all paper figures from verified numeric data.
Run from the paper_veri776/figures/ directory:
    python generate_figures.py

Requires: matplotlib, numpy
All numbers are sourced from docs/findings.md and the 14t Kaggle kernel.
Mark TODO lines with the data that must be filled in after ablation re-runs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# ------------------------------------------------------------------ #
# Verified data                                                        #
# ------------------------------------------------------------------ #
# All values sourced from docs/findings.md (canonical) or literature
# References marked [V]=verified, [P]=paper-reported (not our run)

METHODS = {
    # (mAP, R1, source_type)
    # source_type: "ours" | "literature" | "ours_component"
    "TransReID (ours)":          (89.97, 98.33, "ours_component"),
    "CLIP-SENet (ours re-impl)": (91.54, 97.32, "ours_component"),
    "CLIP-SENet (paper)":        (92.90, 98.70, "literature"),  # [P] arXiv:2502.16815
    "MBR4B-LAI (w/ RK)":        (92.10, None,  "literature"),  # [P] leaderboard, R1 not reported
    "RPTM":                      (88.00, 97.30, "literature"),  # [P]
    "VehicleNet":                (83.41, 96.78, "literature"),  # [P]
    "TransReID (ICCV 2021)":     (82.30, 97.10, "literature"),  # [P]
    "Ours (fusion)":             (93.30, 98.45, "ours"),
}

COLOR_MAP = {
    "ours":           "#1f77b4",   # blue
    "ours_component": "#aec7e8",   # light blue
    "literature":     "#555555",   # dark grey
}

# TODO (after ablation re-runs): fill in rows A0, A2, A6, A7, A8, A9
ABLATION_ROWS = [
    # (label, mAP, verified)
    ("A1  TransReID + AQE + RK",      89.97, True),
    ("A3  CLIP-SENet + AQE + RK",     91.54, True),
    ("A4  Concat + AQE + RK",         93.19, True),
    ("A5  Score fusion + AQE + RK\n(proposed)",  93.30, True),
]

ABLATION_TODO_LABEL = "A6–A9 pending re-run"


# ------------------------------------------------------------------ #
# Figure 1: mAP comparison bar chart                                   #
# ------------------------------------------------------------------ #
def fig_map_bar():
    labels = [
        "TransReID\n(ours)",
        "CLIP-SENet\n(ours re-impl)",
        "CLIP-SENet\n(paper$^\\dagger$)",
        "MBR4B-LAI\n(w/ RK$^\\dagger$)",
        "Ours\n(fusion)",
    ]
    maps = [89.97, 91.54, 92.90, 92.10, 93.30]
    colors = [
        COLOR_MAP["ours_component"],
        COLOR_MAP["ours_component"],
        COLOR_MAP["literature"],
        COLOR_MAP["literature"],
        COLOR_MAP["ours"],
    ]
    hatches = ["", "", "//", "//", ""]

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(labels, maps, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.6, width=0.6)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if val == 93.30 else "normal")

    ax.set_ylabel("mAP (%)")
    ax.set_title("VeRi-776 mAP Comparison")
    ax.set_ylim(85, 94.5)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor=COLOR_MAP["ours_component"],
                       edgecolor="black", label="Our components"),
        mpatches.Patch(facecolor=COLOR_MAP["ours"],
                       edgecolor="black", label="Our fusion (headline)"),
        mpatches.Patch(facecolor=COLOR_MAP["literature"], hatch="//",
                       edgecolor="black", label="Literature ($^\\dagger$author-reported)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
    ax.text(0.01, -0.18,
            "$^\\dagger$Not reproduced by us. CLIP-SENet is also a component of our fusion.",
            transform=ax.transAxes, fontsize=6, color="grey")

    plt.tight_layout()
    plt.savefig("map_bar.pdf", bbox_inches="tight")
    plt.savefig("map_bar.png", bbox_inches="tight")
    print("Saved: map_bar.pdf / map_bar.png")
    plt.close()


# ------------------------------------------------------------------ #
# Figure 2: R1 vs mAP Pareto scatter                                   #
# ------------------------------------------------------------------ #
def fig_pareto():
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for name, (m, r1, stype) in METHODS.items():
        if r1 is None:
            continue
        marker = "o" if stype == "literature" else ("*" if stype == "ours" else "^")
        size   = 90 if stype == "ours" else 55
        ax.scatter(m, r1, c=COLOR_MAP[stype], marker=marker,
                   s=size, zorder=3, edgecolors="black", linewidths=0.5)

        # Label placement — nudge to avoid overlap
        dx, dy = 0.10, 0.05
        short = name.split("(")[0].strip()
        ax.annotate(short, (m, r1), xytext=(m + dx, r1 + dy),
                    fontsize=6.5, color="black")

    # Pareto frontier (mAP-dominant) for illustration
    pareto_pts = sorted(
        [(m, r1) for n, (m, r1, _) in METHODS.items() if r1 is not None],
        key=lambda x: x[0]
    )
    # Simple upper-right Pareto filter
    frontier = []
    max_r1 = -1
    for pt in sorted(pareto_pts, reverse=True):
        if pt[1] >= max_r1:
            frontier.append(pt)
            max_r1 = pt[1]
    frontier.sort()
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "g--", linewidth=0.9, alpha=0.7, label="Pareto frontier (mAP)")

    ax.set_xlabel("mAP (%)")
    ax.set_ylabel("R1 (%)")
    ax.set_title("VeRi-776 R1 vs. mAP")
    ax.set_xlim(80, 94.5)
    ax.set_ylim(96.5, 99.2)
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor=COLOR_MAP["ours"], label="Ours (fusion)"),
        mpatches.Patch(facecolor=COLOR_MAP["ours_component"], label="Our components"),
        mpatches.Patch(facecolor=COLOR_MAP["literature"], label="Literature"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax.text(0.01, -0.17,
            "Note: our R1 (98.45) is below CLIP-SENet paper's R1 (98.70). "
            "We do not claim R1 leadership.",
            transform=ax.transAxes, fontsize=6, color="grey")

    plt.tight_layout()
    plt.savefig("pareto.pdf", bbox_inches="tight")
    plt.savefig("pareto.png", bbox_inches="tight")
    print("Saved: pareto.pdf / pareto.png")
    plt.close()


# ------------------------------------------------------------------ #
# Figure 3: Ablation bar chart                                         #
# ------------------------------------------------------------------ #
def fig_ablation():
    labels = [r[0] for r in ABLATION_ROWS]
    maps   = [r[1] for r in ABLATION_ROWS]
    colors = [COLOR_MAP["ours"] if "proposed" in r[0].lower()
              else COLOR_MAP["ours_component"]
              for r in ABLATION_ROWS]

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bars = ax.barh(labels, maps, color=colors, edgecolor="black",
                   linewidth=0.6, height=0.55)

    for bar, val in zip(bars, maps):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5)

    ax.set_xlabel("mAP (%)")
    ax.set_title("Ablation Study (VeRi-776)")
    ax.set_xlim(87, 94.5)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.18,
            f"Note: {ABLATION_TODO_LABEL}. "
            "Full table in text.",
            transform=ax.transAxes, fontsize=6, color="grey")

    plt.tight_layout()
    plt.savefig("ablation.pdf", bbox_inches="tight")
    plt.savefig("ablation.png", bbox_inches="tight")
    print("Saved: ablation.pdf / ablation.png")
    plt.close()


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    fig_map_bar()
    fig_pareto()
    fig_ablation()
    print("All figures generated.")
