from matplotlib import pyplot as plt
import numpy as np

from lib.path import AnalysisPath


def aggregate_intensities(samples, start, end):
    sum = None
    for arg in range(start, end + 1):
        if sum is None:
            sum = samples[:, arg] ** 2
        else:
            sum += samples[:, arg] ** 2
    return (f"Feature {start}-{end}", np.sqrt(sum))


def analyze_scatter(x, y, c, path: AnalysisPath, suffix=None):
    x_name, x_data = x
    y_name, y_data = y

    path = path.get_path(
        f"scatter_{x_name}_{y_name}{'' if suffix is None else f'_{suffix}'}.png"
    )
    print(f"Scatter plot saved to {path}")
    plt.clf()
    plt.scatter(x_data, y_data, c=c, s=3)
    plt.title(f"{x_name} against {y_name} {'' if suffix is None else f'({suffix})'}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(path, dpi=300)
