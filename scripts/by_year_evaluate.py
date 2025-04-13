from lightgbm import train
from lib.data import load_spectra, xy_split
from lib.evaluations.metrics import evaluate_all_metrics, evaluate_metrics, save_metrics
from lib.evaluations.predictions import save_predictions
from lib.models.base import BaseClassifier
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


init_libraries()

df = load_spectra()

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

years = [2022, 2023, 2024]

def plot_by_year(model: BaseClassifier, metric: str, year_metrics: np.ndarray, path: MetricsPath):
    path = path.get_path(f"{metric.lower()}.png")
    print(f"{metric} by year saved to {path}")

    plt.clf()
    plt.plot(years, year_metrics)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.title(f"{metric} by year - {model.name}")
    plt.legend(loc="best")
    plt.savefig(path, dpi=300)

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    year_metrics = []

    if model != ensemble:
        model.load(ModelPath(Phase.BY_YEAR, fold=2021))

    for year in years:
        print(
            f"=================== Evaluating {model.id} year {year} ==================="
        )

        train = df[df['year'] == 2021]
        val = df[df['year'] == year]
        X_train, y_train, X_val, y_val = xy_split(train, val)

        y_pred = model.predict_proba(X_val)[:, 1]

        save_predictions(
            y_val, y_pred, PredictionPath(model, Phase.BY_YEAR, fold=year)
        )

        evaluate_all_metrics(y_val, y_pred, MetricsPath(model, Phase.BY_YEAR, fold=year))

        year_metrics.append(
            evaluate_metrics(y_val, y_pred, MetricsPath(model, Phase.BY_YEAR, fold=year))
        )

        tf.keras.backend.clear_session()

    year_metrics = np.column_stack(year_metrics)

    plot_by_year(model, "Loss", year_metrics[0], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "Accuracy", year_metrics[1], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "Precision", year_metrics[2], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "Recall", year_metrics[3], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "F1", year_metrics[4], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "AUPRC", year_metrics[5], MetricsPath(model, Phase.BY_YEAR, fold=0))
    plot_by_year(model, "AUROC", year_metrics[6], MetricsPath(model, Phase.BY_YEAR, fold=0))