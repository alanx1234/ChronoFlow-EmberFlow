import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "set_paper_style", "set_age_axis",
    "COLOR", "COLOR_ACCENT", "COLOR_ALT", "CMAP",
    "FIG_SINGLE", "FIG_WIDE", "LABEL_FS", "TICK_FS",
]

COLOR = "#8C7FC7"        # lavender (plasma family)
COLOR_ACCENT = "#E8785A"  # warm coral, for reference/median lines
COLOR_ALT = "#5C1A72"
CMAP = "magma"

FIG_SINGLE = (7.4, 4.8)
FIG_WIDE = (13.2, 4.6)
LABEL_FS = 15
TICK_FS = 13


def set_paper_style() -> None:
    plt.style.use(["seaborn-v0_8-paper", "tableau-colorblind10"])
    plt.rcParams.update({
        "text.usetex":          False,
        "font.family":          "serif",
        "font.serif":           ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset":     "cm",
        "font.size":            12,
        "axes.labelsize":       13,
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "legend.fontsize":      11,
        "figure.dpi":           120,
        "savefig.dpi":          300,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        "xtick.top":            True,
        "ytick.right":          True,
        "axes.spines.top":      True,
        "axes.spines.right":    True,
    })


def set_age_axis(ax, axis: str = "x", unit: str = "gyr") -> None:
    gyr = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    labels = ["1 Myr", "10 Myr", "100 Myr", "1 Gyr", "10 Gyr"]
    ticks = gyr if unit == "gyr" else gyr * 1000

    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_ticks(ticks)
    a.set_ticklabels(labels, fontsize=TICK_FS)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)("Age", fontsize=LABEL_FS)
