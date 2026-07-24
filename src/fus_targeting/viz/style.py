"""Shared visual style for every figure in this project.

Every pipeline stage (MRI prep, pseudo-CT conversion, simulation, feature
extraction, modeling) saves at least one figure through this module, so the
whole project reads as one coherent, paper-quality visual system rather than
ad-hoc plots -- including figures that never make it into the paper itself
(they're also source material for articles/videos later).

Palette values are the validated reference instance from the project's
dataviz design skill (categorical order is CVD-safe by construction; do not
reorder the CATEGORICAL list).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Palette (light-mode values; figures are print/paper-oriented) ---------

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

DIVERGING_BLUE_RED = ["#0d366b", "#2a78d6", "#f0efec", "#eb6834", "#7a1616"]

CHROME = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "good": "#006300",
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# Anatomical images (MRI/pseudo-CT slices) use grayscale, not the categorical
# palette -- that's the correct, standard convention for radiological images.
ANATOMY_CMAP = "gray"

sequential_cmap = LinearSegmentedColormap.from_list("fus_sequential_blue", SEQUENTIAL_BLUE)
diverging_cmap = LinearSegmentedColormap.from_list("fus_diverging_blue_red", DIVERGING_BLUE_RED)


def apply_style() -> None:
    """Set matplotlib rcParams for the whole project. Call once, at import time
    of any script that produces a figure."""
    mpl.rcParams.update({
        "figure.facecolor": CHROME["surface"],
        "axes.facecolor": CHROME["surface"],
        "savefig.facecolor": CHROME["surface"],
        "axes.edgecolor": CHROME["baseline"],
        "axes.labelcolor": CHROME["ink_primary"],
        "axes.titlecolor": CHROME["ink_primary"],
        "axes.grid": True,
        "grid.color": CHROME["gridline"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.color": CHROME["ink_muted"],
        "ytick.color": CHROME["ink_muted"],
        "text.color": CHROME["ink_primary"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def save_figure(fig, relative_path: str, project_root: Path | None = None, formats=("png",)) -> list[Path]:
    """Save a figure into the tracked results/figures/ tree.

    relative_path is stage/name, e.g. "kwave_smoke_test/pressure_field"
    (no extension). Saves paper-quality raster (300dpi PNG) by default; pass
    formats=("png", "pdf") for a vector copy too.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    out_base = project_root / "results" / "figures" / relative_path
    out_base.parent.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved.append(out_path)
    return saved


apply_style()
