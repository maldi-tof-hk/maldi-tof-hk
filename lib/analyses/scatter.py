from matplotlib import pyplot as plt

from lib.path import AnalysisPath


def analyze_scatter(x, y, c, x_name, y_name, path: AnalysisPath, suffix=None):

    path = path.get_path(
        f"scatter_{x_name}_{y_name}{'' if suffix is None else f'_{suffix}'}.png"
    )
    print(f"Scatter plot saved to {path}")
    plt.clf()
    plt.scatter(x, y, c=c, s=3)
    plt.title(f"{x_name} against {y_name} {'' if suffix is None else f'({suffix})'}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(path, dpi=300)
