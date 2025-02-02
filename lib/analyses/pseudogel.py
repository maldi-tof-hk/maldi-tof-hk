from matplotlib import pyplot as plt
import matplotlib.gridspec as grid
import matplotlib
import numpy as np
from lib.path import AnalysisPath


def analyze_pseudogel(
    sensitive_samples,
    resistant_samples,
    center,
    path: AnalysisPath,
    sample_n=500,
    random_state=812,
):
    path = path.get_path(f"pseudogel_{center}.png")

    plot_min = center - 100
    plot_max = center + 100

    rng = np.random.default_rng(random_state)
    sensitive_samples = rng.choice(sensitive_samples, size=sample_n, replace=False)
    rng = np.random.default_rng(random_state)
    resistant_samples = rng.choice(resistant_samples, size=sample_n, replace=False)

    plt.clf()
    plt.figure(figsize=(8, 6))

    gs = grid.GridSpec(
        2,
        2,
        height_ratios=[len(sensitive_samples), len(resistant_samples)],
        width_ratios=[15, 1],
    )

    ax2 = plt.subplot(gs[0])
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_title("MSSA")
    ax2.set_ylabel("Sample")
    p2 = ax2.imshow(
        sensitive_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5)
    )

    colorAx = plt.subplot(gs[1])
    cb = plt.colorbar(p2, cax=colorAx)

    ax3 = plt.subplot(gs[2], sharex=ax2)
    ax3.set_xlim(plot_min, plot_max)
    ax3.set_title("MRSA")
    ax3.set_xlabel("Feature")
    ax3.set_ylabel("Sample")
    p3 = ax3.imshow(
        resistant_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
