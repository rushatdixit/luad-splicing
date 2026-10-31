"""Shared figure style for the project (matplotlib).

Colours come from the dataviz reference palette (validated categorical order, single-hue sequential,
blue<->red diverging with a grey midpoint). Class identity is never encoded by colour alone: tumour =
circle, normal = triangle. Every figure is saved as PNG (300 dpi) and PDF in results/figures/.
"""
import os

import matplotlib

matplotlib.use("Agg")  # headless: figures are only ever saved, never shown
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = "results/figures"

TUMOUR, NORMAL = "#2a78d6", "#eb6834"        # categorical slots 1 and 2
THIRD = "#1baf7a"                            # categorical slot 3 (aqua)
MUTED = "#b9b8b2"                            # de-emphasised points / context
INK, INK2 = "#0b0b0b", "#52514e"             # text tokens
CLASS_COLOR = {"tumour": TUMOUR, "normal": NORMAL}
CLASS_MARKER = {"tumour": "o", "normal": "^"}

# Sequential: one hue (blue), light -> dark.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Diverging: blue <-> red with a neutral grey midpoint.
DIV = LinearSegmentedColormap.from_list("div_bluered", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52a2a"])
NAN_COLOR = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def savefig(fig, name):
    """Save a figure as PNG + PDF under results/figures/ and close it."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def scatter_by_class(ax, x, y, cls, s=12, alpha=0.75):
    """Tumour (circle, blue) first, then normal (triangle, orange) on top, with a thin white ring for overlap."""
    for c in ["tumour", "normal"]:
        m = cls == c
        ax.scatter(x[m], y[m], s=s if c == "tumour" else s * 1.8, c=CLASS_COLOR[c], marker=CLASS_MARKER[c],
                   alpha=alpha, linewidths=0.4, edgecolors="white", label=f"{c} (n={int(m.sum())})")
