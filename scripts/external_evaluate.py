from lib.data import df_to_xy, load_spectra
from lib.evaluations.metrics import (
    evaluate_all_metrics,
    evaluate_multiplex_arc,
    evaluate_multiplex_prc,
)
from lib.evaluations.predictions import save_predictions
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries
import time

init_libraries()

df = load_spectra(path="data/external_processed")

X, y = df_to_xy(df)

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

model_predictions = []


for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    if model != ensemble:
        # Loading not required for ensemble
        model.load(ModelPath(Phase.TRAINING))

    start = time.perf_counter()
    y_pred = model.predict_proba(X)[:, 1]
    end = time.perf_counter()
    print(f"Prediction time for {model.id}: {end - start:.4f} seconds")

    save_predictions(y, y_pred, PredictionPath(model, Phase.EXTERNAL))

    evaluate_all_metrics(y, y_pred, MetricsPath(model, Phase.EXTERNAL))

    model_predictions.append((model, y, y_pred))

evaluate_multiplex_prc(
    model_predictions, MetricsPath("top_models", Phase.EXTERNAL), enable_inset=False
)
evaluate_multiplex_arc(model_predictions, MetricsPath("top_models", Phase.EXTERNAL))
