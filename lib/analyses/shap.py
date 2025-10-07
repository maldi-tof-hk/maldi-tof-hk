from matplotlib import pyplot as plt
import numpy as np
import shap
from lib.models.base import BaseClassifier
from lib.path import ModelAnalysisPath


def analyze_shap_summary(shap_values, X, path: ModelAnalysisPath):
    output = path.get_path(f"shap_summary.png")
    print(f"SHAP summary plot saved to {output}")

    plt.clf()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        title=f"SHAP Summary - {path.model.name if path.model is not None else path.model_id}",
        plot_size=0.25,
    )
    plt.savefig(output, dpi=300)


def analyze_shap_spectrum(shap_values, X, path: ModelAnalysisPath):
    output = path.get_path(f"shap_spectrum.png")
    print(f"SHAP spectrum plot saved to {output}")

    mean_shap = np.transpose(shap_values[0])
    mean_shap = np.array([np.mean(np.absolute(x)) for x in mean_shap])

    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.plot(mean_shap)
    plt.title(
        f"Mean SHAP Values - {path.model.name if path.model is not None else path.model_id}"
    )
    plt.xlabel("Features")
    plt.ylabel("mean(|SHAP value|)")
    plt.savefig(output, dpi=300)


def analyze_all_shap(shap_values, X, path: ModelAnalysisPath):
    analyze_shap_summary(shap_values, X, path)
    analyze_shap_spectrum(shap_values, X, path)
