from matplotlib import pyplot as plt
import matplotlib.gridspec as grid
import matplotlib
import numpy as np
from lib.path import AnalysisPath
from lib.utils import random_choice


def analyze_pseudogel(
    top_samples,
    bottom_samples,
    center,
    path: AnalysisPath,
    sample_n=500,
    random_state=812,
    top_label="MSSA",
    bottom_label="MRSA",
):
    path = path.get_path(f"pseudogel_{center}.png")
    print(f"Pseudogel plot saved to {path}")

    plot_min = center - 50
    plot_max = center + 50

    top_samples = random_choice(top_samples, sample_n, random_state)
    bottom_samples = random_choice(bottom_samples, sample_n, random_state)

    plt.clf()
    plt.figure(figsize=(12, 9))

    gs = grid.GridSpec(
        2,
        2,
        height_ratios=[len(top_samples), len(bottom_samples)],
        width_ratios=[15, 1],
    )

    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_xticks(list(ax1.get_xticks()) + [center])
    ax1.set_title(top_label)
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Sample ID")
    p2 = ax1.imshow(top_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5))

    colorAx = plt.subplot(gs[1])
    cb = plt.colorbar(p2, cax=colorAx, label="Normalized Intensity")

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_xticks(list(ax2.get_xticks()) + [center])
    ax2.set_title(bottom_label)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Sample ID")
    p3 = ax2.imshow(
        bottom_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300)


def analyze_pseudogel_three_pane(
    top_samples,
    middle_samples,
    bottom_samples,
    center,
    path: AnalysisPath,
    sample_n=500,
    random_state=812,
    top_label="MSSA",
    middle_label="True-positive MRSA",
    bottom_label="False-negative MRSA",
):
    path = path.get_path(f"pseudogel_3_{center}.png")
    print(f"3-pane pseudogel plot saved to {path}")

    plot_min = center - 50
    plot_max = center + 50

    top_samples = random_choice(top_samples, sample_n, random_state)
    middle_samples = random_choice(middle_samples, sample_n, random_state)
    bottom_samples = random_choice(bottom_samples, sample_n, random_state)

    plt.clf()
    plt.figure(figsize=(12, 12))

    gs = grid.GridSpec(
        3,
        2,
        height_ratios=[len(top_samples), len(middle_samples), len(bottom_samples)],
        width_ratios=[15, 1],
    )

    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_xticks(list(ax1.get_xticks()) + [center])
    ax1.set_title(top_label)
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Sample ID")
    p2 = ax1.imshow(top_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5))

    colorAx = plt.subplot(gs[1])
    cb = plt.colorbar(p2, cax=colorAx, label="Normalized Intensity")

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_xticks(list(ax2.get_xticks()) + [center])
    ax2.set_title(middle_label)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Sample ID")
    p3 = ax2.imshow(
        middle_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5)
    )

    ax3 = plt.subplot(gs[4], sharex=ax1)
    ax3.set_xlim(plot_min, plot_max)
    ax3.set_xticks(list(ax3.get_xticks()) + [center])
    ax3.set_title(bottom_label)
    ax3.set_xlabel("Features")
    ax3.set_ylabel("Sample ID")
    p3 = ax3.imshow(
        bottom_samples, aspect="auto", norm=matplotlib.colors.PowerNorm(0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
