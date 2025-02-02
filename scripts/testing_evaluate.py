from lib.evaluations.metrics import (
    evaluate_all_metrics,
    evaluate_multiplex_prc,
    evaluate_multiplex_tac,
)
from lib.evaluations.predictions import load_predictions
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import MetricsPath, Phase, PredictionPath
from lib.utils import init_libraries


init_libraries()

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

model_predictions = []


for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    y_true, y_pred = load_predictions(PredictionPath(model, Phase.TESTING))

    evaluate_all_metrics(y_true, y_pred, MetricsPath(model, Phase.TESTING))

    model_predictions.append((model, y_true, y_pred))

evaluate_multiplex_prc(model_predictions, MetricsPath("top_models", Phase.TESTING))
evaluate_multiplex_tac(model_predictions, MetricsPath("top_models", Phase.TESTING))
