"""Build the figures used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FEATURES, Sample  # noqa: E402

COLORS = {"setosa": "#2a9d8f", "versicolor": "#e9c46a", "virginica": "#e76f51"}


def pair_grid(samples: list[Sample], out: Path) -> Path:
    """4x4 grid: histograms on the diagonal, scatter plots elsewhere."""
    n = len(FEATURES)
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    species = sorted({s.species for s in samples})
    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            for sp in species:
                pts = [s.features for s in samples if s.species == sp]
                color = COLORS.get(sp, None)
                if r == c:
                    ax.hist([p[c] for p in pts], bins=12, alpha=0.6, color=color, label=sp)
                else:
                    ax.scatter([p[c] for p in pts], [p[r] for p in pts], s=10, alpha=0.7, color=color, label=sp)
            if r == n - 1:
                ax.set_xlabel(FEATURES[c].replace("_", " "))
            if c == 0:
                ax.set_ylabel(FEATURES[r].replace("_", " "))
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Iris features by species")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def k_curve(results: dict[int, float], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(results)
    ax.plot(ks, [results[k] for k in ks], marker="o", color="#264653")
    ax.set_xlabel("k (neighbors)")
    ax.set_ylabel("5-fold accuracy")
    ax.set_title("Picking k with cross-validation")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
